"""Integration tests for the v2 event streaming routes."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aegra_api.api import event_streaming as es_module
from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.core.orm import get_session
from aegra_api.models.auth import User
from aegra_api.services.event_streaming import capabilities as caps
from aegra_api.services.event_streaming import commands as cmd_module


class _OwnershipSession:
    """Session whose scalar() reports whether the caller owns the thread."""

    def __init__(self, *, owned: bool) -> None:
        self._owned = owned

    async def scalar(self, _stmt: Any) -> Any:
        return "t1" if self._owned else None


def _make_app(*, owned: bool = True) -> FastAPI:
    app = FastAPI()
    user = User(identity="test-user")
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_session] = lambda: _OwnershipSession(owned=owned)
    app.include_router(es_module.router)
    return app


@pytest.fixture(autouse=True)
def _v2_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Turn the flag on and clear the capability cache for each test."""
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    caps._probe_runtime_symbols.cache_clear()
    yield
    caps._probe_runtime_symbols.cache_clear()


class TestCommandRoute:
    def test_run_start_returns_success_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_prepare(*_args: Any, **_kwargs: Any) -> tuple[str, object, object]:
            return "run-1", object(), object()

        monkeypatch.setattr(cmd_module, "_prepare_run", fake_prepare)
        client = TestClient(_make_app())

        resp = client.post(
            "/threads/t1/commands",
            json={"id": 1, "method": "run.start", "params": {"assistant_id": "agent", "input": {"messages": []}}},
        )
        assert resp.status_code == 200
        assert resp.json() == {"type": "success", "id": 1, "result": {"run_id": "run-1"}}

    def test_unknown_command_returns_400_error_envelope(self) -> None:
        client = TestClient(_make_app())
        resp = client.post("/threads/t1/commands", json={"id": 1, "method": "agent.getTree", "params": {}})
        assert resp.status_code == 400
        assert resp.json()["error"] == "not_supported"

    def test_cross_tenant_thread_is_404(self) -> None:
        client = TestClient(_make_app(owned=False))
        resp = client.post("/threads/t1/commands", json={"id": 1, "method": "run.start", "params": {}})
        assert resp.status_code == 404

    def test_disabled_flag_returns_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", False)
        client = TestClient(_make_app())
        resp = client.post("/threads/t1/commands", json={"id": 1, "method": "run.start", "params": {}})
        assert resp.status_code == 503
        # Bare router app returns FastAPI's default {"detail": ...}; the real
        # app remaps this to {"message": ...} via its exception handler.
        assert "FF_V2_EVENT_STREAMING" in resp.json()["detail"]


class TestStreamRoute:
    def test_missing_channels_is_400(self) -> None:
        client = TestClient(_make_app())
        resp = client.post("/threads/t1/stream/events", json={"run_id": "run-1"})
        assert resp.status_code == 400

    def test_unknown_channel_is_400(self) -> None:
        client = TestClient(_make_app())
        resp = client.post("/threads/t1/stream/events", json={"run_id": "run-1", "channels": ["bogus"]})
        assert resp.status_code == 400

    def test_missing_run_id_is_400(self) -> None:
        client = TestClient(_make_app())
        resp = client.post("/threads/t1/stream/events", json={"channels": ["messages"]})
        assert resp.status_code == 400

    def test_cross_tenant_thread_is_404(self) -> None:
        client = TestClient(_make_app(owned=False))
        resp = client.post("/threads/t1/stream/events", json={"run_id": "r", "channels": ["messages"]})
        assert resp.status_code == 404

    def test_stream_emits_v2_frames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A seeded broker run streams content-block frames over SSE."""
        import asyncio

        from langchain_core.messages import AIMessageChunk

        from aegra_api.services.broker import broker_manager

        async def seed() -> None:
            broker = broker_manager.get_or_create_broker("run-sse")
            chunk = AIMessageChunk(content="hi", id="m1")
            chunk.chunk_position = "last"
            await broker.put("run-sse_event_1", ("messages", (chunk, {})))
            await broker.put("run-sse_event_2", ("end", {"status": "success"}))

        asyncio.run(seed())
        client = TestClient(_make_app())

        with client.stream(
            "POST", "/threads/t1/stream/events", json={"run_id": "run-sse", "channels": ["messages", "lifecycle"]}
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        assert "event: messages" in body
        assert "message-start" in body
        assert "content-block-delta" in body
        assert "event: lifecycle" in body
        assert "completed" in body
