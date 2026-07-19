"""Exception-handler tests: every error response shares the Agent Protocol envelope.

The SDK reads a string ``message`` from the error body; FastAPI's default 422
returns a list ``detail``, so without the custom handler a client sees only a
generic "422 Unprocessable Entity".
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from aegra_api.main import exception_handlers

pytestmark = pytest.mark.unit


class _Body(BaseModel):
    name: str
    count: int


def _app() -> FastAPI:
    app = FastAPI()
    for exc_type, handler in exception_handlers.items():
        app.exception_handler(exc_type)(handler)

    @app.post("/echo")
    async def echo(body: _Body) -> dict[str, bool]:
        return {"ok": True}

    return app


def test_422_uses_agent_protocol_envelope() -> None:
    client = TestClient(_app())
    resp = client.post("/echo", json={"name": "x"})  # missing count
    assert resp.status_code == 422
    data = resp.json()
    assert set(data) == {"error", "message", "details"}
    assert data["error"] == "validation_error"
    assert isinstance(data["message"], str) and "count" in data["message"]
    assert isinstance(data["details"]["errors"], list)


def test_422_message_is_a_string_not_a_list() -> None:
    """Regression: SDK reads a string message, not FastAPI's default list detail."""
    client = TestClient(_app())
    resp = client.post("/echo", json={"name": "x", "count": "not-an-int"})
    assert isinstance(resp.json()["message"], str)
