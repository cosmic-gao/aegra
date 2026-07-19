"""Background scheduler that submits due delayed runs (``after_seconds``).

Wakes every ``DELAYED_RUN_POLL_INTERVAL_SECONDS``, atomically claims pending
runs whose ``scheduled_at`` has passed (clearing ``scheduled_at`` so only one
instance submits each), reconstructs the RunJob, and hands it to the executor.
Follows the same start()/stop() lifecycle as CronScheduler / LeaseReaper.
"""

import asyncio
import contextlib

import structlog
from sqlalchemy import select, text

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models.run_job import RunJob
from aegra_api.services.executor import executor
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Atomically claim due runs across instances: clear scheduled_at (marking them
# ready) and return their ids. SKIP LOCKED keeps concurrent pollers disjoint.
_CLAIM_DUE_SQL = text(
    """
    UPDATE runs SET scheduled_at = NULL, updated_at = now()
    WHERE run_id IN (
        SELECT run_id FROM runs
        WHERE status = 'pending' AND claimed_by IS NULL
          AND scheduled_at IS NOT NULL AND scheduled_at <= now()
        ORDER BY scheduled_at ASC
        LIMIT :batch
        FOR UPDATE SKIP LOCKED
    )
    RETURNING run_id
    """
)


def _next_delay(current: float, had_work: bool, base: float, cap: float) -> float:
    """Next poll delay: reset to ``base`` after work, else exponential backoff to ``cap``."""
    if had_work:
        return base
    return min(current * 2, cap)


class DelayedRunScheduler:
    """Periodically submits runs whose ``after_seconds`` delay has elapsed."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Delayed-run scheduler started",
            interval_seconds=settings.worker.DELAYED_RUN_POLL_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Delayed-run scheduler stopped")

    async def _loop(self) -> None:
        base = settings.worker.DELAYED_RUN_POLL_INTERVAL_SECONDS
        cap = min(base * 6, 30.0)  # cap empty-tick backoff; delayed runs are approximate
        delay = base
        while self._running:
            had_work = False
            try:
                had_work = await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in delayed-run scheduler tick")
            # A busy tick resets to base for prompt submission; idle ticks back off
            # so an unused feature doesn't issue a write transaction every `base`s.
            delay = _next_delay(delay, had_work, base, cap)
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> bool:
        """Claim and submit due runs; return whether any were found."""
        maker = _get_session_maker()
        async with maker() as session:
            result = await session.execute(_CLAIM_DUE_SQL, {"batch": settings.worker.DELAYED_RUN_BATCH_SIZE})
            due_ids = [row[0] for row in result.fetchall()]
            await session.commit()

        if not due_ids:
            return False
        logger.info("Submitting due delayed runs", count=len(due_ids))
        for run_id in due_ids:
            await self._submit(run_id)
        return True

    @staticmethod
    async def _submit(run_id: str) -> None:
        maker = _get_session_maker()
        async with maker() as session:
            run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
            if run_orm is None or run_orm.execution_params is None:
                logger.warning("Due delayed run missing or has no execution_params", run_id=run_id)
                return
            job = RunJob.from_run_orm(run_orm)
        await executor.submit(job)
        logger.info("Submitted delayed run", run_id=run_id)


delayed_run_scheduler = DelayedRunScheduler()
