"""Unit tests for the A2A server executor.

Focus: A2A runs each target graph inline via ``ainvoke`` (bypassing the run
executor), so it must bind Langfuse/OTEL trace context itself — otherwise the
exported trace carries no user, session, or trace name.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

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
    async def _fake_get_graph(graph_id: str, user: User | None = None):
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
