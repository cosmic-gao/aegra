"""Background sweeper that deletes stale threads and their checkpoints (TTL).

Opt-in via ``CHECKPOINTER_TTL_ENABLED``. Deletes threads with no active run
whose ``updated_at`` is older than ``CHECKPOINTER_TTL_MINUTES``, along with
their checkpoints. langgraph's saver has no native TTL, so this covers the
thread/checkpoint retention feature. Off by default — it permanently deletes.
"""

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import delete, select

from aegra_api.core.database import db_manager
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

_ACTIVE_STATUSES = ("pending", "running")


class ThreadTTLSweeper:
    """Periodically deletes stale threads and their checkpoints."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if not settings.checkpointer.CHECKPOINTER_TTL_ENABLED:
            logger.info("Thread TTL sweeper disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Thread TTL sweeper started",
            ttl_minutes=settings.checkpointer.CHECKPOINTER_TTL_MINUTES,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Thread TTL sweeper stopped")

    async def _loop(self) -> None:
        interval = settings.checkpointer.CHECKPOINTER_SWEEP_INTERVAL_MINUTES * 60
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in thread TTL sweeper tick")
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        cutoff = datetime.now(UTC) - timedelta(minutes=settings.checkpointer.CHECKPOINTER_TTL_MINUTES)
        active = (
            select(RunORM.run_id)
            .where(RunORM.thread_id == ThreadORM.thread_id, RunORM.status.in_(_ACTIVE_STATUSES))
            .exists()
        )
        checkpointer = db_manager.get_checkpointer()
        maker = _get_session_maker()
        # Claim + delete in one transaction: FOR UPDATE SKIP LOCKED holds the row
        # locks to commit, so concurrent replicas sweep disjoint threads.
        async with maker() as session:
            rows = await session.scalars(
                select(ThreadORM.thread_id)
                .where(ThreadORM.updated_at < cutoff, ~active)
                .limit(settings.checkpointer.CHECKPOINTER_SWEEP_BATCH_SIZE)
                .with_for_update(skip_locked=True, of=ThreadORM)
            )
            stale = list(rows.all())
            if not stale:
                return
            logger.info("Sweeping stale threads", count=len(stale))
            for thread_id in stale:
                await self._delete_checkpoints(thread_id, checkpointer)
            # Deleting the thread rows cascades to their runs (FK ON DELETE CASCADE).
            await session.execute(delete(ThreadORM).where(ThreadORM.thread_id.in_(stale)))
            await session.commit()

    @staticmethod
    async def _delete_checkpoints(thread_id: str, checkpointer: Any) -> None:
        try:
            await checkpointer.adelete_thread(thread_id)
        except Exception as exc:
            logger.warning("Failed to delete checkpoints for stale thread", thread_id=thread_id, error=str(exc))


thread_ttl_sweeper = ThreadTTLSweeper()
