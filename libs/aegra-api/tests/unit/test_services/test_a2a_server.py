"""Unit tests for the A2A server executor.

Focus: A2A runs each target graph inline via ``ainvoke`` (bypassing the run
executor), so it must bind Langfuse/OTEL trace context itself — otherwise the
exported trace carries no user, session, or trace name.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from fastapi import HTTPException
from langgraph_sdk import Auth

import aegra_api.services.a2a_server as a2a_server
from aegra_api.models.auth import User
from aegra_api.observability.span_enrichment import _trace_attrs

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_trace_context():
    """execute() binds trace context in-place; reset so it never bleeds."""
    _trace_attrs.set(None)
    structlog.contextvars.clear_contextvars()
    yield
    _trace_attrs.set(None)
    structlog.contextvars.clear_contextvars()


async def test_execute_binds_langfuse_trace_context() -> None:
    """The graph runs with enrichment active — user/session/name/source all set."""
    captured: dict[str, object] = {}

    class _CapturingGraph:
        async def ainvoke(self, input: dict, config: dict) -> dict:
            captured["attrs"] = _trace_attrs.get()
            return {"messages": [{"role": "assistant", "content": "done"}]}

    @asynccontextmanager
    async def _fake_get_graph(
        graph_id: str, *, config: dict | None = None, user: User | None = None, context: dict | None = None
    ) -> AsyncIterator[_CapturingGraph]:
        yield _CapturingGraph()

    fake_service = MagicMock()
    fake_service.list_graphs.return_value = {"weather": "path"}
    fake_service.get_graph = _fake_get_graph

    task = MagicMock()
    task.id = "task-1"
    task.context_id = "ctx-1"

    context = MagicMock()
    context.call_context.state = {"graph_id": "weather", "aegra_user": User(identity="u1", display_name="U")}
    context.current_task = task
    context.get_user_input.return_value = "hi"

    updater = MagicMock()
    updater.start_work = AsyncMock()
    updater.add_artifact = AsyncMock()
    updater.complete = AsyncMock()

    with (
        patch.object(a2a_server, "get_langgraph_service", return_value=fake_service),
        patch.object(a2a_server, "TaskUpdater", return_value=updater),
    ):
        await a2a_server.AegraAgentExecutor().execute(context, MagicMock())

    attrs = captured["attrs"]
    assert isinstance(attrs, dict)
    assert attrs["langfuse.user.id"] == "u1"
    assert attrs["langfuse.trace.name"] == "weather"
    assert attrs["langfuse.trace.metadata.source"] == "a2a"
    assert attrs["langfuse.trace.metadata.a2a_context_id"] == "ctx-1"
    assert attrs["langfuse.trace.metadata.run_id"] == "task-1"
    updater.complete.assert_awaited_once()


async def test_execute_rejects_unknown_assistant() -> None:
    """An unregistered graph id raises before any trace context is bound."""
    fake_service = MagicMock()
    fake_service.list_graphs.return_value = {"weather": "path"}

    context = MagicMock()
    context.call_context.state = {"graph_id": "nope", "aegra_user": None}

    with (
        patch.object(a2a_server, "get_langgraph_service", return_value=fake_service),
        pytest.raises(ValueError, match="Unknown assistant"),
    ):
        await a2a_server.AegraAgentExecutor().execute(context, MagicMock())

    assert _trace_attrs.get() is None


async def test_execute_denies_when_auth_handler_rejects() -> None:
    """A denying @auth.on.threads.create_run handler blocks A2A before the graph runs.

    Before this fix the A2A path only authenticated, so an operator's create_run
    authorization (enforced on the REST path) was silently bypassed here.
    """
    auth = Auth()

    @auth.on.threads.create_run
    async def _deny(*, ctx: Any, value: Any) -> bool:
        return False

    ran: dict[str, bool] = {"invoked": False}

    class _Graph:
        async def ainvoke(self, input: dict, config: dict) -> dict:
            ran["invoked"] = True
            return {"messages": []}

    @asynccontextmanager
    async def _fake_get_graph(
        graph_id: str, *, config: dict | None = None, user: User | None = None, context: dict | None = None
    ) -> AsyncIterator[_Graph]:
        yield _Graph()

    fake_service = MagicMock()
    fake_service.list_graphs.return_value = {"weather": "path"}
    fake_service.get_graph = _fake_get_graph

    context = MagicMock()
    context.call_context.state = {"graph_id": "weather", "aegra_user": User(identity="u1", display_name="U")}
    context.get_user_input.return_value = "hi"

    with (
        patch.object(a2a_server, "get_langgraph_service", return_value=fake_service),
        patch("aegra_api.core.auth_handlers.get_auth_instance", return_value=auth),
        pytest.raises(HTTPException) as exc_info,
    ):
        await a2a_server.AegraAgentExecutor().execute(context, MagicMock())

    assert exc_info.value.status_code == 403
    assert ran["invoked"] is False  # graph never executed
    assert _trace_attrs.get() is None  # rejected before trace binding


async def test_execute_applies_auth_handler_config_and_context() -> None:
    """An allowing handler's config/context reach the inline graph run (was bypassed).

    Mirrors api/runs.create_run: injected config merges into the run config and the
    server-generated thread_id stays authoritative.
    """
    auth = Auth()

    @auth.on.threads.create_run
    async def _inject(*, ctx: Any, value: Any) -> dict[str, Any]:
        return {"config": {"configurable": {"tenant": ctx.user.identity}}, "context": {"scope": "team"}}

    seen: dict[str, Any] = {}

    class _Graph:
        async def ainvoke(self, input: dict, config: dict) -> dict:
            seen["invoke_config"] = config
            return {"messages": [{"role": "assistant", "content": "ok"}]}

    @asynccontextmanager
    async def _fake_get_graph(
        graph_id: str, *, config: dict | None = None, user: User | None = None, context: dict | None = None
    ) -> AsyncIterator[_Graph]:
        seen["graph_config"] = config
        seen["graph_context"] = context
        yield _Graph()

    fake_service = MagicMock()
    fake_service.list_graphs.return_value = {"weather": "path"}
    fake_service.get_graph = _fake_get_graph

    task = MagicMock()
    task.id = "task-1"
    task.context_id = "ctx-1"

    context = MagicMock()
    context.call_context.state = {"graph_id": "weather", "aegra_user": User(identity="u1", display_name="U")}
    context.current_task = task
    context.get_user_input.return_value = "hi"

    updater = MagicMock()
    updater.start_work = AsyncMock()
    updater.add_artifact = AsyncMock()
    updater.complete = AsyncMock()

    with (
        patch.object(a2a_server, "get_langgraph_service", return_value=fake_service),
        patch.object(a2a_server, "TaskUpdater", return_value=updater),
        patch("aegra_api.core.auth_handlers.get_auth_instance", return_value=auth),
    ):
        await a2a_server.AegraAgentExecutor().execute(context, MagicMock())

    invoke_config = seen["invoke_config"]
    assert invoke_config["configurable"]["tenant"] == "u1"
    assert invoke_config["configurable"]["thread_id"]  # server thread_id preserved
    assert seen["graph_config"]["configurable"]["tenant"] == "u1"
    assert seen["graph_context"] == {"scope": "team"}
    updater.complete.assert_awaited_once()
