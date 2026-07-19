"""Run endpoints for Agent Protocol"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, MutableMapping
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from redis import RedisError
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette import EventSourceResponse

from aegra_api.core.active_runs import active_runs
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.database import db_manager
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker, get_session
from aegra_api.core.sse import create_end_event, get_sse_headers, make_sse_response, sse_to_bytes
from aegra_api.models import Run, RunCreate, RunStatus, User
from aegra_api.models.enums import CancelAction
from aegra_api.models.errors import CONFLICT, NOT_FOUND, SSE_RESPONSE
from aegra_api.models.runs import RunsCancelRequest
from aegra_api.services.broker import broker_manager
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.services.run_waiters import TERMINAL_STATES, encode_output, heartbeat_wait_body, wrap_run_result
from aegra_api.services.streaming_service import streaming_service
from aegra_api.settings import settings
from aegra_api.utils.status_compat import validate_run_status

router = APIRouter(tags=["Thread Runs"], dependencies=auth_dependency)

logger = structlog.getLogger(__name__)


# active_runs is imported from aegra_api.core.active_runs (dependency-free module)

# Default stream modes for background run execution
DEFAULT_STREAM_MODES = ["values"]


@router.post("/threads/{thread_id}/runs", response_model=Run, responses={**NOT_FOUND, **CONFLICT})
async def create_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Create and execute a new run.

    Starts graph execution asynchronously and returns the run record
    immediately with status `pending`. Poll the run or use the stream
    endpoint to follow progress. Provide either `input` or `command` (for
    human-in-the-loop resumption) but not both.
    """
    existing_thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id))
    if existing_thread and existing_thread.user_id != user.identity:
        raise HTTPException(404, f"Thread '{thread_id}' not found")

    # Authorization check (create_run action on threads resource)
    ctx = build_auth_context(user, "threads", "create_run")
    value = {**request.model_dump(), "thread_id": thread_id}
    filters = await handle_event(ctx, value)

    # If handler modified config/context, update request
    if filters:
        if "config" in filters and isinstance(filters["config"], dict):
            request.config = {**(request.config or {}), **filters["config"]}
        if "context" in filters and isinstance(filters["context"], dict):
            request.context = {**(request.context or {}), **filters["context"]}
    else:
        value_config = value.get("config")
        if isinstance(value_config, dict):
            request.config = {**(request.config or {}), **value_config}

        value_context = value.get("context")
        if isinstance(value_context, dict):
            request.context = {**(request.context or {}), **value_context}

    _run_id, run, _job = await _prepare_run(session, thread_id, request, user, initial_status="pending")

    return run


@router.post("/threads/{thread_id}/runs/stream", responses={**SSE_RESPONSE, **NOT_FOUND, **CONFLICT})
async def create_and_stream_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
) -> EventSourceResponse:
    """Create a new run and stream its execution via SSE.

    Returns a `text/event-stream` response with Server-Sent Events. Each
    event has a `type` field (e.g. `values`, `updates`, `messages`,
    `metadata`, `end`) and a JSON `data` payload.

    Set `on_disconnect` to `"continue"` if the run should keep executing
    after the client disconnects (default is `"cancel"`). Use `stream_mode`
    to control which event types are emitted.

    A periodic SSE keepalive comment is sent every
    ``KEEPALIVE_INTERVAL_SECS`` so idle proxies don't drop long-running
    silent nodes (e.g. agents holding an upstream WebSocket).
    """
    maker = _get_session_maker()
    async with maker() as session:
        existing_thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id))
        if existing_thread and existing_thread.user_id != user.identity:
            raise HTTPException(404, f"Thread '{thread_id}' not found")

        run_id, run, _job = await _prepare_run(session, thread_id, request, user, initial_status="pending")

    # Default to cancel on disconnect - this matches user expectation that clicking
    # "Cancel" in the frontend will stop the backend task. Users can explicitly
    # set on_disconnect="continue" if they want the task to continue.
    cancel_on_disconnect = (request.on_disconnect or "cancel").lower() == "cancel"

    async def _cancel_on_client_close(_msg: MutableMapping[str, Any]) -> None:
        try:
            await broker_manager.request_cancel(run_id, "cancel")
        except (RedisError, OSError):
            # Swallow infra/transport failures so sse-starlette's task group
            # tears down cleanly. Programmer errors (TypeError, AttributeError,
            # ...) propagate. The lease reaper picks up unreachable runs.
            # OSError covers ConnectionError/TimeoutError (3.11+ subclasses).
            logger.exception("Failed to cancel run on client disconnect", run_id=run_id)

    close_handler = _cancel_on_client_close if cancel_on_disconnect else None

    return make_sse_response(
        sse_to_bytes(streaming_service.stream_run_execution(run, None)),
        close_handler=close_handler,
        headers={
            **get_sse_headers(),
            "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.get("/threads/{thread_id}/runs/{run_id}", response_model=Run, responses={**NOT_FOUND})
async def get_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Get a run by its ID.

    Returns the current state of the run including its status, input, output,
    and error information.
    """
    # Authorization check (read action on runs resource)
    ctx = build_auth_context(user, "runs", "read")
    value = {"run_id": run_id, "thread_id": thread_id}
    await handle_event(ctx, value)

    stmt = select(RunORM).where(
        RunORM.run_id == str(run_id),
        RunORM.thread_id == thread_id,
        RunORM.user_id == user.identity,
    )
    logger.info(f"[get_run] querying DB run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(stmt)
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # No refresh needed: fresh per-request session + expire_on_commit=False means
    # the scalar() row is already current; a refresh() would just re-SELECT it.
    logger.info(
        f"[get_run] found run status={run_orm.status} user={user.identity} thread_id={thread_id} run_id={run_id}"
    )
    return Run.model_validate(run_orm)


# SDK RunSelectField values; fields Aegra does not store are omitted from rows.
_RUN_SELECT_FIELDS = frozenset(
    {
        "run_id",
        "thread_id",
        "assistant_id",
        "created_at",
        "updated_at",
        "status",
        "metadata",
        "kwargs",
        "multitask_strategy",
    }
)


@router.get("/threads/{thread_id}/runs")
async def list_runs(
    thread_id: str,
    limit: int = Query(10, ge=1, le=1000, description="Maximum number of runs to return"),
    offset: int = Query(0, ge=0, description="Number of runs to skip for pagination"),
    status: str | None = Query(
        None, description="Filter by run status (e.g. pending, running, success, error, interrupted)"
    ),
    select_fields: list[str] | None = Query(
        None, alias="select", description="Return only these run fields (SDK RunSelectField values)."
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[Run] | list[dict[str, Any]]:
    """List runs for a thread.

    Returns runs ordered by creation time (newest first). Use `status` to
    filter, `limit`/`offset` to paginate, and `select` to project fields.
    """
    if select_fields:
        invalid = [f for f in select_fields if f not in _RUN_SELECT_FIELDS]
        if invalid:
            raise HTTPException(422, f"Invalid select columns: {invalid}. Expected: {sorted(_RUN_SELECT_FIELDS)}")
    stmt = (
        select(RunORM)
        .where(
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
            *([RunORM.status == status] if status else []),
        )
        .limit(limit)
        .offset(offset)
        .order_by(RunORM.created_at.desc())
    )
    logger.info(f"[list_runs] querying DB thread_id={thread_id} user={user.identity}")
    result = await session.scalars(stmt)
    rows = result.all()
    runs = [Run.model_validate(r) for r in rows]
    logger.info(f"[list_runs] total={len(runs)} user={user.identity} thread_id={thread_id}")
    if select_fields:
        wanted = set(select_fields)
        return [{k: v for k, v in r.model_dump(mode="json").items() if k in wanted} for r in runs]
    return runs


@router.patch("/threads/{thread_id}/runs/{run_id}", response_model=Run, responses={**NOT_FOUND})
async def update_run(
    thread_id: str,
    run_id: str,
    request: RunStatus,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Update a run's status.

    Primarily used to interrupt a running execution. Set `status` to
    `"interrupted"` to cooperatively stop the run.
    """
    logger.info(f"[update_run] fetch for update run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # Handle interruption/cancellation
    # Validate status conforms to API specification
    validated_status = validate_run_status(request.status)

    if validated_status == "interrupted":
        logger.info(f"[update_run] cancelling/interrupting run_id={run_id} user={user.identity} thread_id={thread_id}")
        # Handle interruption - use interrupt_run for cooperative interruption
        await streaming_service.interrupt_run(run_id)
        logger.info(f"[update_run] set DB status=interrupted run_id={run_id}")
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        await session.commit()
        logger.info(f"[update_run] commit done (interrupted) run_id={run_id}")

    # Return final run state
    run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")
    # Refresh to ensure we have the latest data after our own update
    await session.refresh(run_orm)
    return Run.model_validate(run_orm)


@router.get("/threads/{thread_id}/runs/{run_id}/join", responses={**NOT_FOUND})
async def join_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Wait for a run to complete and return its output.

    Returns a chunked ``application/json`` response. While the run is still
    executing, the server sends periodic ``\\n`` heartbeat bytes to keep the
    connection alive through proxies and load balancers (AWS ALB, Cloudflare,
    etc.). The final chunk is the JSON result. Leading whitespace is ignored
    by JSON parsers, so clients can parse the concatenated body normally.

    If the run is already in a terminal state, the output is returned
    immediately with no heartbeat overhead.

    Sessions are managed manually (not via ``Depends``) to avoid holding a
    pool connection during the long wait.
    """
    maker = _get_session_maker()

    # Short-lived session: validate run exists and check terminal state
    async with maker() as session:
        run_orm = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == str(run_id),
                RunORM.thread_id == thread_id,
                RunORM.user_id == user.identity,
            )
        )
        if not run_orm:
            raise HTTPException(404, f"Run '{run_id}' not found")

        if run_orm.status in TERMINAL_STATES:
            result = wrap_run_result(run_orm.status, run_orm.output, run_orm.error_message)
            return StreamingResponse(
                iter([encode_output(result)]),
                media_type="application/json",
            )

    return StreamingResponse(
        heartbeat_wait_body(
            run_id,
            thread_id,
            user.identity,
            timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
        ),
        media_type="application/json",
        headers={
            "Location": f"/threads/{thread_id}/runs/{run_id}/join",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.post("/threads/{thread_id}/runs/wait", responses={**NOT_FOUND, **CONFLICT})
async def wait_for_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Create a run, execute it, and wait for completion.

    Returns a chunked ``application/json`` response with periodic ``\\n``
    heartbeat bytes to keep the connection alive. The final chunk is the
    JSON result. Uses ``BG_JOB_TIMEOUT_SECS`` (default 1 hour) as the
    safety-net timeout.

    Sessions are managed manually (not via ``Depends``) to avoid holding a
    pool connection during the long wait.
    """
    maker = _get_session_maker()

    # Session block: all pre-execution DB work (validate, create run, submit)
    async with maker() as session:
        existing_thread = await session.scalar(select(ThreadORM).where(ThreadORM.thread_id == thread_id))
        if existing_thread and existing_thread.user_id != user.identity:
            raise HTTPException(404, f"Thread '{thread_id}' not found")

        run_id, _run, _job = await _prepare_run(session, thread_id, request, user, initial_status="pending")

    # No pool connection held from here — safe for long waits
    return StreamingResponse(
        heartbeat_wait_body(
            run_id,
            thread_id,
            user.identity,
            timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
        ),
        media_type="application/json",
        headers={
            "Location": f"/threads/{thread_id}/runs/{run_id}/join",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.get("/threads/{thread_id}/runs/{run_id}/stream", responses={**SSE_RESPONSE, **NOT_FOUND})
async def stream_run(
    thread_id: str,
    run_id: str,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    _stream_mode: str | None = Query(None, description="Override the stream mode for this connection."),
    user: User = Depends(get_current_user),
) -> EventSourceResponse:
    """Stream an existing run's execution via SSE.

    Attach to a run that was created without streaming (e.g. via the create
    endpoint) to receive its events in real time. If the run has already
    finished, a single `end` event is emitted. Use the `Last-Event-ID`
    header to resume from a specific event after a disconnect.

    A periodic SSE keepalive comment is sent every
    ``KEEPALIVE_INTERVAL_SECS`` so idle proxies don't drop attached streams.
    """
    maker = _get_session_maker()
    async with maker() as session:
        logger.info(f"[stream_run] fetch for stream run_id={run_id} thread_id={thread_id} user={user.identity}")
        run_orm = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == str(run_id),
                RunORM.thread_id == thread_id,
                RunORM.user_id == user.identity,
            )
        )
        if not run_orm:
            raise HTTPException(404, f"Run '{run_id}' not found")

        logger.info(f"[stream_run] status={run_orm.status} user={user.identity} thread_id={thread_id} run_id={run_id}")
        run_status = run_orm.status
        run_model = Run.model_validate(run_orm)
    # No client_close_handler_callable: this is a reconnect-style endpoint, so
    # a single client disconnecting must not cancel the shared run — other
    # consumers may still be attached via /join or another /stream.
    # If already terminal and no Last-Event-ID, just emit end.
    # If Last-Event-ID is present, fall through to stream_run_execution
    # which will replay missed events from the buffer before ending.
    if run_status in TERMINAL_STATES and not last_event_id:
        final_status = "error" if run_status == "error" else run_status

        async def generate_final() -> AsyncGenerator[str, None]:
            yield create_end_event(status=final_status)

        logger.info(f"[stream_run] starting terminal stream run_id={run_id} status={run_status}")
        return make_sse_response(
            sse_to_bytes(generate_final()),
            headers={
                **get_sse_headers(),
                "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
                "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
            },
        )

    # Stream active or pending runs via broker

    return make_sse_response(
        sse_to_bytes(streaming_service.stream_run_execution(run_model, last_event_id)),
        headers={
            **get_sse_headers(),
            "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


_THREAD_STREAM_MODES = frozenset({"run_modes", "lifecycle", "state_update"})


@router.get("/threads/{thread_id}/stream", responses={**SSE_RESPONSE, **NOT_FOUND})
async def join_thread_stream(
    thread_id: str,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    stream_mode: str | None = Query(
        None, description="Comma-separated ThreadStreamMode values (SDK sends 'stream_mode')."
    ),
    user: User = Depends(get_current_user),
) -> EventSourceResponse:
    """Join the stream of the thread's active (or most recent) run.

    Resolves the thread's in-flight run — falling back to the newest run —
    and attaches to its event stream with `Last-Event-ID` replay, exactly
    like the run-scoped stream endpoint. Only `run_modes` is served.
    """
    requested = [m.strip() for m in stream_mode.split(",")] if stream_mode else ["run_modes"]
    invalid = [m for m in requested if m not in _THREAD_STREAM_MODES]
    if invalid:
        raise HTTPException(422, f"Invalid stream mode: {invalid[0]}")
    if "run_modes" not in requested:
        raise HTTPException(422, "Only the 'run_modes' thread stream mode is supported")

    maker = _get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(
            select(RunORM)
            .where(
                RunORM.thread_id == thread_id,
                RunORM.user_id == user.identity,
                RunORM.status.in_(("pending", "running")),
            )
            .order_by(RunORM.created_at.desc())
            .limit(1)
        )
        if run_orm is None:
            run_orm = await session.scalar(
                select(RunORM)
                .where(RunORM.thread_id == thread_id, RunORM.user_id == user.identity)
                .order_by(RunORM.created_at.desc())
                .limit(1)
            )
        if run_orm is None:
            raise HTTPException(404, f"Thread '{thread_id}' has no runs to stream")
        run_status = run_orm.status
        run_model = Run.model_validate(run_orm)

    if run_status in TERMINAL_STATES and not last_event_id:

        async def generate_final() -> AsyncGenerator[str, None]:
            yield create_end_event(status="error" if run_status == "error" else run_status)

        return make_sse_response(
            sse_to_bytes(generate_final()),
            headers={
                **get_sse_headers(),
                "Location": f"/threads/{thread_id}/stream",
                "Content-Location": f"/threads/{thread_id}/runs/{run_model.run_id}",
            },
        )

    return make_sse_response(
        sse_to_bytes(streaming_service.stream_run_execution(run_model, last_event_id)),
        headers={
            **get_sse_headers(),
            "Location": f"/threads/{thread_id}/stream",
            "Content-Location": f"/threads/{thread_id}/runs/{run_model.run_id}",
        },
    )


@router.post(
    "/threads/{thread_id}/runs/{run_id}/cancel",
    response_model=Run,
    responses={**NOT_FOUND},
)
async def cancel_run_endpoint(
    thread_id: str,
    run_id: str,
    wait: int = Query(0, ge=0, le=1, description="Set to 1 to wait for the run task to settle before returning."),
    action: CancelAction = Query(
        "interrupt",
        description=(
            "Cancellation strategy: 'interrupt' (default) for a cooperative "
            "interrupt that lets the graph save partial state, or 'rollback' to "
            "cancel then delete the run and the checkpoints it produced."
        ),
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Cancel or interrupt a running execution.

    Use `action=interrupt` (default) to cooperatively interrupt so the graph can
    handle the interrupt and save partial state, or `action=rollback` to cancel
    and then discard the run record plus its checkpoints. Set `wait=1` to block
    until the background task has fully settled before returning the updated run.
    """
    logger.info(f"[cancel_run] fetch run run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    if action == "interrupt":
        logger.info(f"[cancel_run] interrupt run_id={run_id} user={user.identity} thread_id={thread_id}")
        await streaming_service.interrupt_run(run_id)
    else:
        logger.info(f"[cancel_run] {action} run_id={run_id} user={user.identity} thread_id={thread_id}")
        await streaming_service.cancel_run(run_id)
    await session.execute(
        update(RunORM).where(RunORM.run_id == str(run_id)).values(status="interrupted", updated_at=datetime.now(UTC))
    )
    await session.commit()

    # Rollback must always settle first: deleting checkpoints while the
    # executor is still finalizing would let it write rows back afterwards.
    if wait or action == "rollback":
        # Poll DB until the run reaches a terminal state (or 10s timeout).
        # This is simpler and more reliable than pub/sub for cancel-with-wait
        # since the cancel has already been issued and the status update committed.
        for _ in range(20):
            await asyncio.sleep(0.5)
            session.expire_all()  # sync method, clears cache
            fresh = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
            if fresh and fresh.status in TERMINAL_STATES:
                break

    # Reload the settled snapshot (also what a rollback returns post-delete).
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found after cancellation")
    run = Run.model_validate(run_orm)

    if action == "rollback":
        await db_manager.get_checkpointer().adelete_for_runs([run_id])
        await session.delete(run_orm)
        await session.commit()
    return run


@router.post("/runs/cancel", status_code=204)
async def cancel_runs_bulk(
    request: RunsCancelRequest,
    action: str = Query(
        "interrupt",
        pattern="^(interrupt|rollback)$",
        description="'interrupt' marks runs interrupted; 'rollback' also deletes them and their checkpoints.",
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Bulk-cancel runs by status, or by thread_id + run_ids."""
    where = [RunORM.user_id == user.identity]
    if request.status is not None:
        statuses = ("pending", "running") if request.status == "all" else (request.status,)
        where.append(RunORM.status.in_(statuses))
    else:
        where.append(RunORM.thread_id == request.thread_id)
        where.append(RunORM.run_id.in_(request.run_ids or []))
    rows = list((await session.scalars(select(RunORM).where(*where))).all())
    if not rows:
        raise HTTPException(404, "No runs found to cancel")

    run_ids = [row.run_id for row in rows]
    for rid in run_ids:
        if action == "interrupt":
            await streaming_service.interrupt_run(rid)
        else:
            await streaming_service.cancel_run(rid)
    await session.execute(
        update(RunORM).where(RunORM.run_id.in_(run_ids)).values(status="interrupted", updated_at=datetime.now(UTC))
    )
    await session.commit()

    if action == "rollback":
        # Settle before deleting checkpoints so a finalizing executor cannot
        # write rows back afterwards (bounded to the single-cancel wait).
        for _ in range(20):
            await asyncio.sleep(0.5)
            session.expire_all()
            statuses = (await session.scalars(select(RunORM.status).where(RunORM.run_id.in_(run_ids)))).all()
            if all(s in TERMINAL_STATES for s in statuses):
                break
        await db_manager.get_checkpointer().adelete_for_runs(run_ids)
        await session.execute(delete(RunORM).where(RunORM.run_id.in_(run_ids)))
        await session.commit()
    return Response(status_code=204)


@router.delete(
    "/threads/{thread_id}/runs/{run_id}",
    status_code=204,
    responses={**NOT_FOUND, **CONFLICT},
)
async def delete_run(
    thread_id: str,
    run_id: str,
    force: int = Query(0, ge=0, le=1, description="Set to 1 to cancel an active run before deleting it."),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a run record.

    If the run is active (pending or running) and `force=0`, returns 409
    Conflict. Set `force=1` to cancel the run first (best-effort) and then
    delete it. Returns 204 No Content on success.
    """
    # Authorization check (delete action on runs resource)
    ctx = build_auth_context(user, "runs", "delete")
    value = {"run_id": run_id, "thread_id": thread_id}
    await handle_event(ctx, value)
    logger.info(f"[delete_run] fetch run run_id={run_id} thread_id={thread_id} user={user.identity}")
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # If active and not forcing, reject deletion
    if run_orm.status in ["pending", "running"] and not force:
        raise HTTPException(
            status_code=409,
            detail="Run is active. Retry with force=1 to cancel and delete.",
        )

    # If forcing and active, cancel first
    if force and run_orm.status in ["pending", "running"]:
        logger.info(f"[delete_run] force-cancelling active run run_id={run_id}")
        await streaming_service.cancel_run(run_id)
        # Best-effort: wait for bg task to settle
        task = active_runs.get(run_id)
        if task:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # Delete the record
    await session.execute(
        delete(RunORM).where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.identity,
        )
    )
    await session.commit()

    # Clean up active task if exists
    task = active_runs.pop(run_id, None)
    if task and not task.done():
        task.cancel()

    # 204 No Content
    return
