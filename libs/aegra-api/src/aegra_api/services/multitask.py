"""Multitask strategy enforcement for concurrent runs on a thread.

Implements LangGraph Platform double-texting semantics — reject / interrupt /
rollback / enqueue. reject/interrupt/rollback act at run-creation time against
any in-flight run of the thread; enqueue defers to the executor's per-thread
serialization (worker claim gate in prod, local per-thread chain in dev).
"""

import asyncio
import contextlib

import structlog
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.active_runs import active_runs
from aegra_api.core.database import db_manager
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models.auth import User
from aegra_api.services.run_status import update_run_status
from aegra_api.services.streaming_service import streaming_service

logger = structlog.getLogger(__name__)

DEFAULT_MULTITASK_STRATEGY = "reject"
VALID_STRATEGIES = frozenset({"reject", "interrupt", "rollback", "enqueue"})
_ACTIVE_STATUSES = ("pending", "running")
_TERMINAL_STATUSES = frozenset({"success", "error", "interrupted", "timeout"})

# Bound the request latency of an interrupt/rollback double-text while giving
# the executing worker (same-process or cross-instance) time to finalize on the
# cancel signal before we force the run terminal.
_CANCEL_SETTLE_ATTEMPTS = 20
_CANCEL_SETTLE_INTERVAL_SECONDS = 0.1


async def resolve_multitask(
    session: AsyncSession,
    thread_id: str,
    strategy: str | None,
    user: User,
) -> None:
    """Enforce ``multitask_strategy`` against in-flight runs on the thread.

    Called at the start of run preparation. ``reject`` raises 409 when the
    thread is busy; ``interrupt``/``rollback`` cancel the in-flight run(s)
    (rollback also discards their state); ``enqueue`` is a no-op here — the
    executor serializes the new run behind the running one.
    """
    effective = (strategy or DEFAULT_MULTITASK_STRATEGY).lower()
    if effective not in VALID_STRATEGIES:
        raise HTTPException(status_code=422, detail=f"Unknown multitask_strategy {strategy!r}")
    if effective == "enqueue":
        return

    active = await _active_runs_on_thread(session, thread_id, user.identity)
    if not active:
        return

    if effective == "reject":
        raise HTTPException(
            status_code=409,
            detail="Thread is already running a task. Wait for it to finish or choose a different multitask strategy.",
        )

    for run in active:
        await _cancel_active_run(run.run_id)
        if effective == "rollback":
            await _rollback_run_state(thread_id, run.run_id)


async def _active_runs_on_thread(session: AsyncSession, thread_id: str, user_id: str) -> list[RunORM]:
    stmt = select(RunORM).where(
        RunORM.thread_id == thread_id,
        RunORM.user_id == user_id,
        RunORM.status.in_(_ACTIVE_STATUSES),
    )
    return list((await session.scalars(stmt)).all())


async def _cancel_active_run(run_id: str) -> None:
    """Signal cancellation and wait for the run to reach a terminal state.

    The local task is cancelled directly; cross-instance runs receive the broker
    cancel signal and finalize on their own worker. We then poll until terminal
    (force-interrupting a never-started pending run) so the successor's
    thread-free claim gate cannot observe a still-active run.
    """
    task = active_runs.pop(run_id, None)
    try:
        await streaming_service.cancel_run(run_id)
    except Exception as exc:
        # Non-fatal: local task cancel + terminal poll below still stop the run.
        logger.debug("Cancel signal failed", run_id=run_id, error=str(exc))
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    await _await_terminal(run_id)


async def _await_terminal(run_id: str) -> None:
    session_maker = _get_session_maker()
    for _ in range(_CANCEL_SETTLE_ATTEMPTS):
        async with session_maker() as session:
            status = await session.scalar(select(RunORM.status).where(RunORM.run_id == run_id))
        if status is None or status in _TERMINAL_STATUSES:
            return
        await asyncio.sleep(_CANCEL_SETTLE_INTERVAL_SECONDS)
    await update_run_status(run_id, "interrupted")


async def _rollback_run_state(thread_id: str, run_id: str) -> None:
    """Discard the state a rolled-back run produced.

    Deletes exactly the checkpoints that run created — plus their writes,
    GC'ing orphaned blobs — via the checkpointer's per-run deletion, returning
    the thread to its pre-run state while keeping earlier history.
    """
    await db_manager.get_checkpointer().adelete_for_runs([run_id])
    logger.info("Rolled back run state", thread_id=thread_id, run_id=run_id)
