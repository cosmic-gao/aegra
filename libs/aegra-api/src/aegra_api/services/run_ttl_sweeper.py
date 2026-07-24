"""Background sweeper that prunes old terminal run rows (per-run TTL).

Opt-in via ``RUN_TTL_ENABLED``. Deletes terminal runs whose ``updated_at`` is
older than ``RUN_TTL_MINUTES`` so the ``runs`` table doesn't grow without bound
on busy threads. Thread state and checkpoints are untouched — only historical
run rows are removed (webhook_deliveries cascade with them). Off by default
since it permanently deletes.
"""

import asyncio
import contextlib

import structlog
from sqlalchemy import text

from aegra_api.core.orm import _get_session_maker
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Internal tuning (not env-configurable — RUN_TTL_ENABLED / RUN_TTL_MINUTES are
# the user-facing knobs). Sweep cadence and per-tick delete cap.
_SWEEP_INTERVAL_MINUTES = 60
_BATCH_SIZE = 1000

# Status literals are inlined (not bound params) so the predicate matches the
# idx_runs_ttl_sweep partial index; SKIP LOCKED keeps replicas' deletes disjoint.
_SWEEP_SQL = text(
    """
    DELETE FROM runs WHERE run_id IN (
        SELECT run_id FROM runs
        WHERE status IN ('success', 'error', 'interrupted', 'timeout')
          AND updated_at < now() - make_interval(mins => :ttl)
        ORDER BY updated_at ASC
        LIMIT :batch
        FOR UPDATE SKIP LOCKED
    )
    RETURNING run_id
    """
)


class RunTTLSweeper:
    """Periodically deletes terminal runs older than the configured TTL."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if not settings.run_ttl.RUN_TTL_ENABLED:
            logger.info("Run TTL sweeper disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Run TTL sweeper started", ttl_minutes=settings.run_ttl.RUN_TTL_MINUTES)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Run TTL sweeper stopped")

    async def _loop(self) -> None:
        interval = _SWEEP_INTERVAL_MINUTES * 60
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in run TTL sweeper tick")
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        maker = _get_session_maker()
        async with maker() as session:
            result = await session.execute(
                _SWEEP_SQL,
                {"ttl": settings.run_ttl.RUN_TTL_MINUTES, "batch": _BATCH_SIZE},
            )
            deleted = result.fetchall()
            await session.commit()
        if deleted:
            logger.info("Pruned stale runs", count=len(deleted))


run_ttl_sweeper = RunTTLSweeper()
