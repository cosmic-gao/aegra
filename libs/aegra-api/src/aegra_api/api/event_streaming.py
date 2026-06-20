"""Agent Protocol v2 event streaming endpoints.

* ``POST /threads/{thread_id}/stream/events`` — SSE stream filtered by
  channel. Body is an ``EventStreamRequest`` ``{channels, since?}``.
* ``POST /threads/{thread_id}/commands`` — run a thread command
  (``run.start``, ``input.respond``) and get a JSON response envelope.

Both gate on the ``FF_V2_EVENT_STREAMING`` flag + runtime capability, and
verify thread ownership before doing anything.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import get_session
from aegra_api.core.sse import format_sse_message, get_sse_headers, make_sse_response, sse_to_bytes
from aegra_api.models import User
from aegra_api.services.event_streaming.capabilities import get_v2_capabilities
from aegra_api.services.event_streaming.commands import handle_command
from aegra_api.services.event_streaming.session import ThreadEventSession, validate_channels

logger = structlog.getLogger(__name__)

router = APIRouter(tags=["Event Streaming"], dependencies=auth_dependency)


async def _verify_thread_owned(session: AsyncSession, thread_id: str, user: User) -> None:
    """404 unless the thread exists and belongs to the caller."""
    owned = await session.scalar(
        select(ThreadORM.thread_id).where(
            ThreadORM.thread_id == thread_id,
            ThreadORM.user_id == user.identity,
        )
    )
    if owned is None:
        raise HTTPException(404, f"Thread '{thread_id}' not found")


def _require_v2_enabled() -> None:
    """503 with a clear reason when v2 is off or the runtime can't serve it."""
    caps = get_v2_capabilities()
    if not caps.ok:
        raise HTTPException(503, caps.error_message)


@router.post("/threads/{thread_id}/stream/events")
async def stream_thread_events(
    thread_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    """Open a channel-filtered SSE stream of v2 events for the thread's run.

    Body ``{channels: [...], since?: int}``. Each SSE frame is a protocol
    event envelope: ``event:`` is the channel, ``data:`` the JSON envelope,
    ``id:`` the session ``seq`` a client echoes back as ``since`` on resume.
    """
    _require_v2_enabled()
    await _verify_thread_owned(session, thread_id, user)

    body = await _json_body(request)
    channels, invalid = validate_channels(body.get("channels"))
    if invalid:
        raise HTTPException(400, f"Unsupported or missing channels: {', '.join(invalid)}")

    run_id = body.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise HTTPException(400, "run_id is required to stream a thread's events")

    since = body.get("since")
    since = since if isinstance(since, int) and not isinstance(since, bool) and since >= 0 else None

    session_stream = ThreadEventSession(run_id, channels=channels, since=since)
    return make_sse_response(sse_to_bytes(_frame_events(session_stream)), headers=get_sse_headers())


@router.post("/threads/{thread_id}/commands")
async def post_thread_command(
    thread_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    """Run a single v2 command on the thread and return its response envelope."""
    _require_v2_enabled()
    await _verify_thread_owned(session, thread_id, user)

    payload = await _json_body(request)
    response, _run_id = await handle_command(payload, session=session, thread_id=thread_id, user=user)
    status_code = 200 if response.get("type") == "success" else 400
    return JSONResponse(response, status_code=status_code)


async def _json_body(request: Request) -> dict[str, Any]:
    """Parse a JSON object body or raise 400."""
    try:
        body = await request.json()
    except (ValueError, TypeError):
        raise HTTPException(400, "Request body must be valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")
    return body


async def _frame_events(session_stream: ThreadEventSession) -> AsyncGenerator[str, None]:
    """Frame v2 event envelopes as SSE messages (event=method, data=envelope, id=seq)."""
    async for envelope in session_stream.stream():
        yield format_sse_message(envelope["method"], envelope, str(envelope["seq"]))
