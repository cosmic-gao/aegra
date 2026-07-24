"""Unit tests for the MCP server: each graph is exposed as its own typed tool.

Mirrors LangGraph Platform's MCP model (one tool per agent, input schema from the
graph) rather than the old generic ``list_assistants``/``run_assistant`` pair.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
import structlog
from fastapi import HTTPException
from langgraph_sdk import Auth
from mcp import types

import aegra_api.services.mcp_server as mcp_server
from aegra_api.models.auth import User
from aegra_api.observability.span_enrichment import _trace_attrs

pytestmark = pytest.mark.unit

_USER = User(identity="u1", display_name="U")


class _FakeGraph:
    def __init__(self, schema: dict | None = None):
        self._schema = schema if schema is not None else {"type": "object", "properties": {"messages": {}}}

    def get_input_jsonschema(self) -> dict:
        return self._schema

    async def ainvoke(self, input: dict, config: dict) -> dict:
        return {"echo": input, "messages": ["done"]}


class _FakeService:
    def __init__(self, graphs: tuple[str, ...] = ("weather", "chat"), *, schema_error: bool = False):
        self._graphs = list(graphs)
        self._schema_error = schema_error
        self._graph = _FakeGraph()

    def list_graphs(self) -> dict[str, str]:
        return dict.fromkeys(self._graphs, "path")

    async def get_graph_for_validation(self, graph_id: str, user: User | None = None) -> _FakeGraph:
        if self._schema_error:
            raise RuntimeError("cannot compile")
        return self._graph

    @asynccontextmanager
    async def get_graph(
        self, name: str, *, config: dict | None = None, user: User | None = None, context: dict | None = None
    ) -> AsyncIterator[_FakeGraph]:
        yield self._graph


@pytest.fixture(autouse=True)
def _clear_schema_cache():
    mcp_server._SCHEMA_CACHE.clear()
    yield
    mcp_server._SCHEMA_CACHE.clear()


@pytest.fixture(autouse=True)
def _reset_trace_context():
    """_call_tool now binds trace context in-place; reset so it never bleeds."""
    _trace_attrs.set(None)
    structlog.contextvars.clear_contextvars()
    yield
    _trace_attrs.set(None)
    structlog.contextvars.clear_contextvars()


@pytest.fixture
def auth():
    with patch.object(mcp_server, "_authenticate", return_value=_USER):
        yield


async def test_list_tools_one_per_graph_with_input_schema(auth) -> None:
    with patch.object(mcp_server, "get_langgraph_service", return_value=_FakeService(("weather", "chat"))):
        tools = await mcp_server._list_tools()
    assert {t.name for t in tools} == {"weather", "chat"}
    assert all(isinstance(t, types.Tool) and t.inputSchema.get("type") == "object" for t in tools)
    assert all("Run the" in (t.description or "") for t in tools)


async def test_list_tools_falls_back_to_permissive_schema(auth) -> None:
    with patch.object(mcp_server, "get_langgraph_service", return_value=_FakeService(("agent",), schema_error=True)):
        tools = await mcp_server._list_tools()
    assert tools[0].inputSchema == {"type": "object", "additionalProperties": True}


async def test_input_schema_is_cached_per_graph(auth) -> None:
    """A second tools/list must not recompile graphs — schema comes from cache."""
    svc = _FakeService(("weather", "chat"))
    calls: dict[str, int] = {"n": 0}
    original = svc.get_graph_for_validation

    async def counting(graph_id: str, user=None):
        calls["n"] += 1
        return await original(graph_id, user=user)

    svc.get_graph_for_validation = counting
    with patch.object(mcp_server, "get_langgraph_service", return_value=svc):
        await mcp_server._list_tools()
        await mcp_server._list_tools()
    assert calls["n"] == 2  # one compile per graph, not 4


async def test_invalidate_schema_cache_forces_recompute(auth) -> None:
    svc = _FakeService(("weather",))
    calls: dict[str, int] = {"n": 0}
    original = svc.get_graph_for_validation

    async def counting(graph_id: str, user=None):
        calls["n"] += 1
        return await original(graph_id, user=user)

    svc.get_graph_for_validation = counting
    with patch.object(mcp_server, "get_langgraph_service", return_value=svc):
        await mcp_server._list_tools()
        mcp_server.invalidate_schema_cache()
        await mcp_server._list_tools()
    assert calls["n"] == 2  # recomputed after invalidation


async def test_call_tool_runs_graph_and_returns_serialized_output(auth) -> None:
    with patch.object(mcp_server, "get_langgraph_service", return_value=_FakeService(("weather",))):
        content = await mcp_server._call_tool("weather", {"messages": ["hi"]})
    assert content[0].type == "text"
    assert json.loads(content[0].text)["echo"] == {"messages": ["hi"]}


async def test_call_tool_binds_langfuse_trace_context(auth) -> None:
    """The graph runs with Langfuse enrichment active — user/session/name/source set.

    Without bind_run_trace_context, MCP-invoked graphs export un-attributed traces
    (no user, no session grouping, generic trace name).
    """
    captured: dict[str, object] = {}

    class _CapturingGraph:
        def get_input_jsonschema(self) -> dict:
            return {"type": "object"}

        async def ainvoke(self, input: dict, config: dict) -> dict:
            captured["attrs"] = _trace_attrs.get()
            return {"messages": ["done"]}

    class _CapturingService(_FakeService):
        @asynccontextmanager
        async def get_graph(
            self, name: str, *, config: dict | None = None, user: User | None = None, context: dict | None = None
        ) -> AsyncIterator[_CapturingGraph]:
            yield _CapturingGraph()

    with patch.object(mcp_server, "get_langgraph_service", return_value=_CapturingService(("weather",))):
        await mcp_server._call_tool("weather", {"messages": ["hi"]})

    attrs = captured["attrs"]
    assert isinstance(attrs, dict)
    assert attrs["langfuse.user.id"] == "u1"
    assert attrs["langfuse.trace.name"] == "weather"
    assert attrs["langfuse.trace.metadata.source"] == "mcp"
    assert "langfuse.session.id" in attrs


async def test_call_tool_rejects_unknown_assistant(auth) -> None:
    with (
        patch.object(mcp_server, "get_langgraph_service", return_value=_FakeService(("weather",))),
        pytest.raises(ValueError, match="Unknown assistant"),
    ):
        await mcp_server._call_tool("nope", {})


async def test_transport_dispatch_uses_per_graph_handler(auth) -> None:
    """The low-level handler override reaches the transport dispatch table."""
    handler = mcp_server.mcp._mcp_server.request_handlers[types.ListToolsRequest]
    with patch.object(mcp_server, "get_langgraph_service", return_value=_FakeService(("weather",))):
        result = await handler(types.ListToolsRequest(method="tools/list"))
    assert [t.name for t in result.root.tools] == ["weather"]


async def test_authenticate_without_http_context_raises() -> None:
    with pytest.raises(ValueError, match="no HTTP context"):
        await mcp_server._authenticate()


async def test_call_tool_denies_when_auth_handler_rejects(auth) -> None:
    """A denying @auth.on.threads.create_run handler blocks MCP before the graph runs.

    Before this fix the MCP tool path only authenticated, so create_run authorization
    (enforced on the REST path) was silently bypassed here.
    """
    denier = Auth()

    @denier.on.threads.create_run
    async def _deny(*, ctx: Any, value: Any) -> bool:
        return False

    with (
        patch.object(mcp_server, "get_langgraph_service", return_value=_FakeService(("weather",))),
        patch("aegra_api.core.auth_handlers.get_auth_instance", return_value=denier),
        pytest.raises(HTTPException) as exc_info,
    ):
        await mcp_server._call_tool("weather", {"messages": ["hi"]})

    assert exc_info.value.status_code == 403
    assert _trace_attrs.get() is None  # rejected before trace binding


async def test_call_tool_applies_auth_handler_config_and_context(auth) -> None:
    """An allowing handler's config/context reach the inline MCP graph run (was bypassed).

    Mirrors api/runs.create_run: injected config merges into the run config and the
    server-generated thread_id stays authoritative.
    """
    injector = Auth()

    @injector.on.threads.create_run
    async def _inject(*, ctx: Any, value: Any) -> dict[str, Any]:
        return {"config": {"configurable": {"tenant": ctx.user.identity}}, "context": {"scope": "team"}}

    seen: dict[str, Any] = {}

    class _CapturingGraph:
        def get_input_jsonschema(self) -> dict:
            return {"type": "object"}

        async def ainvoke(self, input: dict, config: dict) -> dict:
            seen["invoke_config"] = config
            return {"messages": ["done"]}

    class _CapturingService(_FakeService):
        @asynccontextmanager
        async def get_graph(
            self, name: str, *, config: dict | None = None, user: User | None = None, context: dict | None = None
        ) -> AsyncIterator[_CapturingGraph]:
            seen["graph_config"] = config
            seen["graph_context"] = context
            yield _CapturingGraph()

    with (
        patch.object(mcp_server, "get_langgraph_service", return_value=_CapturingService(("weather",))),
        patch("aegra_api.core.auth_handlers.get_auth_instance", return_value=injector),
    ):
        await mcp_server._call_tool("weather", {"messages": ["hi"]})

    invoke_config = seen["invoke_config"]
    assert invoke_config["configurable"]["tenant"] == "u1"
    assert invoke_config["configurable"]["thread_id"]  # server thread_id preserved
    assert seen["graph_config"]["configurable"]["tenant"] == "u1"
    assert seen["graph_context"] == {"scope": "team"}
