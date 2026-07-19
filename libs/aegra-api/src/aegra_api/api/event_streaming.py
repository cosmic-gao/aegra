"""Agent Protocol v2 event streaming endpoints.

* ``POST /threads/{thread_id}/stream/events`` — SSE stream of a run's
  events, filtered by channel. Body is an ``EventStreamRequest``.
* ``POST /threads/{thread_id}/commands`` — run a thread command
  (``run.start``, ``input.respond``) and get a JSON response envelope.
* ``WS /threads/{thread_id}/stream/events`` — full-duplex transport:
  command envelopes in, event/response envelopes out (LangGraph Platform
  WebSocket parity).

All gate on the ``FF_V2_EVENT_STREAMING`` flag + runtime capability, and
verify thread (and run) ownership before doing anything.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse
from starlette.authentication import AuthenticationError

from aegra_api.core.auth_deps import _to_user_model, auth_dependency, get_current_user
from aegra_api.core.auth_middleware import get_auth_backend
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.core.sse import (
    _decode_literal_unicode_escapes,
    format_sse_message,
    get_sse_headers,
    make_sse_response,
    sse_to_bytes,
)
from aegra_api.models import User
from aegra_api.models.event_streaming import EventStreamRequest, ThreadCommand
from aegra_api.services.assistant_service import AssistantService
from aegra_api.services.event_streaming.capabilities import get_v2_capabilities
from aegra_api.services.event_streaming.commands import _status_to_error_code, _thread_assistant_id, handle_command
from aegra_api.services.event_streaming.protocol import build_error, build_success
from aegra_api.services.event_streaming.session import RunLister, ThreadEventSession, validate_channels
from aegra_api.services.langgraph_service import get_langgraph_service

logger = structlog.getLogger(__name__)

# WS routes can't take the router-level ``require_auth`` (it needs an HTTP
# Request), so the HTTP routes carry the auth dependency individually.
router = APIRouter(tags=["Event Streaming"])


async def _verify_thread_owned_or_new(session: AsyncSession, thread_id: str, user: User) -> None:
    """Allow a not-yet-existing thread; block one owned by someone else.

    The SDK mints the thread id client-side and expects ``run.start`` to
    create it (run preparation does, owned by the caller). So a missing
    thread is fine here; an existing thread owned by another user is 404.
    """
    existing_owner = await session.scalar(select(ThreadORM.user_id).where(ThreadORM.thread_id == thread_id))
    if existing_owner is not None and existing_owner != user.identity:
        raise HTTPException(404, f"Thread '{thread_id}' not found")


def _thread_run_lister(thread_id: str, user: User) -> RunLister:
    """Async callable returning the thread's (run_id, status, graph_name) rows (oldest first), user-scoped.

    Called repeatedly while a stream is live so a run started after the stream
    opened is picked up. Status lets the session drain historical runs whose
    broker events expired instead of tailing them forever; graph_name feeds the
    run's root lifecycle events. Each tick uses a short-lived session so a
    connection is not held for the whole SSE lifetime (see #423).
    """

    async def list_run_ids() -> list[tuple[str, str | None, str | None]]:
        maker = _get_session_maker()
        async with maker() as session:
            rows = await session.execute(
                select(RunORM.run_id, RunORM.status, AssistantORM.graph_id)
                .outerjoin(AssistantORM, RunORM.assistant_id == AssistantORM.assistant_id)
                .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
                .order_by(RunORM.created_at.asc())
            )
            return [(run_id, status, graph_id) for run_id, status, graph_id in rows.all()]

    return list_run_ids


def _require_v2_enabled() -> None:
    """503 with a clear reason when v2 is off or the runtime can't serve it."""
    caps = get_v2_capabilities()
    if not caps.ok:
        raise HTTPException(503, caps.error_message)


# Error responses both v2 routes can emit, for an accurate OpenAPI contract.
_V2_ERROR_RESPONSES: dict[int | str, dict[str, str]] = {
    400: {"description": "Invalid request (unsupported channels, malformed command)"},
    404: {"description": "Thread owned by another user"},
    503: {"description": "v2 disabled by flag, or runtime too old for native v3 events"},
}


@router.post(
    "/threads/{thread_id}/stream/events",
    response_class=EventSourceResponse,
    dependencies=auth_dependency,
    responses={
        200: {"description": "SSE stream of protocol event envelopes", "content": {"text/event-stream": {}}},
        **_V2_ERROR_RESPONSES,
    },
)
async def stream_thread_events(
    thread_id: str,
    body: EventStreamRequest,
    user: User = Depends(get_current_user),
) -> EventSourceResponse:
    """Open a channel-filtered SSE stream of the thread's run events.

    Thread-scoped: events for any run on the thread flow through, so a
    client can open the stream then issue ``run.start``. Each SSE frame's
    ``data:`` is a protocol event envelope; ``id:`` is the ``seq`` a client
    echoes back as ``since`` on resume.

    DB work runs in short-lived sessions (the upfront ownership check here, and
    each run-lister poll) so no connection is held for the SSE lifetime (#423).
    """
    _require_v2_enabled()

    channels, invalid = validate_channels(body.channels)
    if invalid:
        raise HTTPException(400, f"Unsupported channels: {', '.join(invalid)}")

    # The SDK opens the stream before run.start, against a thread it minted
    # client-side — a not-yet-existing thread is allowed; one owned by another
    # user is not.
    maker = _get_session_maker()
    async with maker() as session:
        await _verify_thread_owned_or_new(session, thread_id, user)

    session_stream = ThreadEventSession(
        thread_id,
        channels=channels,
        list_run_ids=_thread_run_lister(thread_id, user),
        since=body.since,
        namespaces=body.namespaces,
        depth=body.depth,
    )
    return make_sse_response(sse_to_bytes(_frame_events(session_stream)), headers=get_sse_headers())


@router.post(
    "/threads/{thread_id}/commands",
    dependencies=auth_dependency,
    responses={
        200: {"description": "Protocol response envelope (success or error)"},
        404: {"description": "Thread owned by another user"},
        503: {"description": "v2 disabled by flag, or runtime too old for native v3 events"},
    },
)
async def post_thread_command(
    thread_id: str,
    body: ThreadCommand,
    user: User = Depends(get_current_user),
) -> JSONResponse:
    """Run a single v2 command on the thread and return its response envelope.

    Uses a short-lived session (the run created by ``run.start`` executes on
    the worker; the response returns before streaming begins).
    """
    _require_v2_enabled()

    maker = _get_session_maker()
    async with maker() as session:
        # run.start may target a not-yet-created thread (the SDK mints the id and
        # expects run preparation to create it); other ownership is enforced there.
        await _verify_thread_owned_or_new(session, thread_id, user)
        response, _run_id = await handle_command(body.model_dump(), session=session, thread_id=thread_id, user=user)

    # Protocol error envelopes ride HTTP 200 — a client treating non-2xx as a
    # transport failure would throw before parsing the envelope's error code.
    return JSONResponse(response, status_code=200)


async def _frame_events(session_stream: ThreadEventSession) -> AsyncGenerator[str, None]:
    """Frame v2 event envelopes as SSE messages (event=method, data=envelope, id=seq)."""
    async for envelope in session_stream.stream():
        yield format_sse_message(envelope["method"], envelope, str(envelope["seq"]))


# --- WebSocket transport -----------------------------------------------------

_WS_INVALID_JSON_MESSAGE = "Protocol commands must be valid JSON."
_WS_INVALID_ENVELOPE_MESSAGE = "Protocol commands must include an integer id and string method."
_WS_WATCHDOG_REASON = "thread stream ended"

# App close codes mirroring the HTTP statuses of the sibling routes.
_WS_CLOSE_UNAUTHORIZED = 4401
_WS_CLOSE_NOT_FOUND = 4404
_WS_CLOSE_INTERNAL_ERROR = 1011

_frame_serializer = GeneralSerializer()


@router.websocket("/threads/{thread_id}/stream/events")
async def stream_thread_events_ws(websocket: WebSocket, thread_id: str) -> None:
    """Full-duplex v2 protocol socket on the same path as the SSE stream.

    The client sends command envelopes (``{id, method, params}``) as JSON
    text frames: ``subscription.subscribe`` / ``subscription.unsubscribe``
    manage which event channels this socket receives, ``run.start`` /
    ``input.respond`` ride the shared command handler, and ``agent.getTree``
    returns the assistant's drawable graph. The server pushes subscribed
    event envelopes (the SSE ``data:`` payloads, including ``seq``) and
    command responses as JSON text frames. When the underlying thread event
    stream ends the socket closes with 1011 ``thread stream ended``.
    """
    user = await _authenticate_ws(websocket)
    if user is None:
        await websocket.close(code=_WS_CLOSE_UNAUTHORIZED, reason="Authentication required")
        return

    caps = get_v2_capabilities()
    if not caps.ok:
        await websocket.close(code=_WS_CLOSE_INTERNAL_ERROR, reason=_trim_close_reason(caps.error_message))
        return

    maker = _get_session_maker()
    try:
        async with maker() as session:
            await _verify_thread_owned_or_new(session, thread_id, user)
    except HTTPException as exc:
        await websocket.close(code=_WS_CLOSE_NOT_FOUND, reason=_trim_close_reason(str(exc.detail)))
        return

    await websocket.accept()
    await _ProtocolSocket(websocket, thread_id=thread_id, user=user).run()


async def _authenticate_ws(websocket: WebSocket) -> User | None:
    """Authenticate a WS handshake with the same backend ``require_auth`` uses.

    A ``WebSocket`` is an ``HTTPConnection`` and the backend reads only its
    headers, so the HTTP dependency's flow applies unchanged. ``None`` means
    reject (the backend refused, or produced no usable identity).
    """
    backend = get_auth_backend()
    try:
        result = await backend.authenticate(websocket)
    except AuthenticationError as exc:
        logger.info("WebSocket authentication failed", error=str(exc))
        return None
    if result is None:
        return None

    credentials, user = result
    # Mirror require_auth's scope side effects for downstream compatibility.
    websocket.scope["user"] = user
    websocket.scope["auth"] = credentials
    try:
        return _to_user_model(user)
    except HTTPException as exc:
        logger.info("WebSocket authentication failed", error=str(exc.detail))
        return None


class _ProtocolSocket:
    """One accepted WS connection speaking the v2 command/event protocol."""

    def __init__(self, websocket: WebSocket, *, thread_id: str, user: User) -> None:
        self._websocket = websocket
        self._thread_id = thread_id
        self._user = user
        self._send_lock = asyncio.Lock()
        self._subscriptions: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    async def run(self) -> None:
        """Serve command frames until the client disconnects or a close fires."""
        try:
            while True:
                frame = await _receive_ws_text(self._websocket)
                if frame is None:
                    return
                await self._handle_frame(frame)
        except WebSocketDisconnect:
            return
        finally:
            await self._shutdown()

    async def _handle_frame(self, frame: str) -> None:
        """Parse one inbound frame, dispatch it, and answer with an envelope."""
        try:
            payload = json.loads(frame)
        except json.JSONDecodeError:
            await self._send(build_error(None, "invalid_argument", _WS_INVALID_JSON_MESSAGE))
            return

        if not isinstance(payload, dict):
            await self._send(build_error(None, "invalid_argument", _WS_INVALID_ENVELOPE_MESSAGE))
            return
        command_id = payload.get("id")
        method = payload.get("method")
        if not isinstance(command_id, int) or not isinstance(method, str):
            fallback_id = command_id if isinstance(command_id, int) else None
            await self._send(build_error(fallback_id, "invalid_argument", _WS_INVALID_ENVELOPE_MESSAGE))
            return
        params = payload.get("params", {})
        if not isinstance(params, dict):
            await self._send(build_error(command_id, "invalid_argument", "params must be an object."))
            return

        # An RPC response must stay an envelope — mirror handle_command's
        # backstop so a dispatch bug never tears the socket down.
        try:
            response = await self._dispatch(command_id, method, params)
        except Exception:
            logger.exception("Unhandled error dispatching WebSocket command", method=method, thread_id=self._thread_id)
            response = build_error(command_id, "unknown_error", f"Command {method!r} failed unexpectedly.")
        await self._send(response)

    async def _dispatch(self, command_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "subscription.subscribe":
            return self._subscribe(command_id, params)
        if method == "subscription.unsubscribe":
            return self._unsubscribe(command_id, params)
        if method == "agent.getTree":
            return await self._agent_tree(command_id, params)
        return await self._delegate(command_id, method, params)

    def _subscribe(self, command_id: int, params: dict[str, Any]) -> dict[str, Any]:
        """Attach a channel-filtered event subscription to this socket."""
        try:
            request = EventStreamRequest.model_validate(params)
        except ValidationError as exc:
            return build_error(command_id, "invalid_argument", str(exc.errors()[0].get("msg", "invalid params")))

        channels, invalid = validate_channels(request.channels)
        if invalid:
            return build_error(command_id, "invalid_argument", f"Unsupported channels: {', '.join(invalid)}")

        session_stream = ThreadEventSession(
            self._thread_id,
            channels=channels,
            list_run_ids=_thread_run_lister(self._thread_id, self._user),
            since=request.since,
            namespaces=request.namespaces,
            depth=request.depth,
        )
        subscription_id = uuid.uuid4().hex
        self._subscriptions[subscription_id] = asyncio.create_task(self._pump(session_stream))
        return build_success(command_id, {"subscription_id": subscription_id})

    def _unsubscribe(self, command_id: int, params: dict[str, Any]) -> dict[str, Any]:
        subscription_id = params.get("subscription_id")
        task = self._subscriptions.pop(subscription_id, None) if isinstance(subscription_id, str) else None
        if task is None:
            return build_error(command_id, "no_such_subscription", f"Unknown subscription id {subscription_id!r}.")
        task.cancel()
        return build_success(command_id, {})

    async def _delegate(self, command_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run one command through the shared handler (run.start, input.respond, unknown).

        Re-checks thread ownership per command like the HTTP route does; over
        WS the resulting HTTPException becomes a protocol error envelope.
        """
        envelope = {"id": command_id, "method": method, "params": params}
        maker = _get_session_maker()
        try:
            async with maker() as session:
                await _verify_thread_owned_or_new(session, self._thread_id, self._user)
                response, _run_id = await handle_command(
                    envelope, session=session, thread_id=self._thread_id, user=self._user
                )
        except HTTPException as exc:
            return build_error(command_id, _status_to_error_code(exc.status_code), str(exc.detail))
        return response

    async def _agent_tree(self, command_id: int, params: dict[str, Any]) -> dict[str, Any]:
        """Resolve the assistant's drawable graph (the REST ``/assistants/{id}/graph`` payload).

        ``assistant_id`` defaults to the assistant of the thread's most recent
        run, matching how ``input.respond`` recovers it.
        """
        xray = params.get("xray", False)
        if not isinstance(xray, (bool, int)):
            return build_error(command_id, "invalid_argument", "xray must be a boolean or integer.")

        assistant_id = params.get("assistant_id")
        maker = _get_session_maker()
        try:
            async with maker() as session:
                if not isinstance(assistant_id, str) or not assistant_id:
                    assistant_id = await _thread_assistant_id(session, self._thread_id, self._user)
                if not assistant_id:
                    return build_error(
                        command_id, "no_such_run", "No run on this thread to resolve an assistant from; pass assistant_id."
                    )
                service = AssistantService(session, self._user, get_langgraph_service())
                graph = await service.get_assistant_graph(assistant_id, xray)
        except HTTPException as exc:
            return build_error(command_id, _status_to_error_code(exc.status_code), str(exc.detail))
        return build_success(command_id, graph)

    async def _pump(self, session_stream: ThreadEventSession) -> None:
        """Push one subscription's envelopes; watchdog-close when the thread stream ends."""
        try:
            async for envelope in session_stream.stream():
                await self._send(envelope)
        except (WebSocketDisconnect, RuntimeError) as exc:
            # The peer vanished or another task already closed the socket.
            logger.debug("WebSocket event pump stopped", error=str(exc))
            return
        await self._close(_WS_CLOSE_INTERNAL_ERROR, _WS_WATCHDOG_REASON)

    async def _send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._websocket.send_text(_dump_ws_frame(payload))

    async def _close(self, code: int, reason: str) -> None:
        """Close once; racing closers and an already-gone peer are no-ops."""
        if self._closed:
            return
        self._closed = True
        try:
            async with self._send_lock:
                await self._websocket.close(code=code, reason=_trim_close_reason(reason))
        except (WebSocketDisconnect, RuntimeError) as exc:
            logger.debug("WebSocket already closed", error=str(exc))

    async def _shutdown(self) -> None:
        """Cancel every subscription pump and await their teardown."""
        tasks = list(self._subscriptions.values())
        self._subscriptions.clear()
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("WebSocket event pump failed", error=str(result))


async def _receive_ws_text(websocket: WebSocket) -> str | None:
    """Next data frame as text, or ``None`` once the client disconnected.

    Binary frames decode as UTF-8; undecodable bytes become replacement
    characters and then fail JSON parsing, yielding the protocol error.
    """
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        return None
    text = message.get("text")
    if text is not None:
        return text
    data = message.get("bytes")
    return "" if data is None else data.decode("utf-8", errors="replace")


def _dump_ws_frame(payload: dict[str, Any]) -> str:
    """Serialize one protocol frame exactly like the SSE ``data:`` payload."""
    text = json.dumps(payload, default=_frame_serializer.serialize, separators=(",", ":"), ensure_ascii=False)
    return _decode_literal_unicode_escapes(text)


def _trim_close_reason(reason: str) -> str:
    """Close-frame reasons cap at 123 UTF-8 bytes (RFC 6455 control frame)."""
    return reason.encode("utf-8")[:123].decode("utf-8", errors="ignore")
