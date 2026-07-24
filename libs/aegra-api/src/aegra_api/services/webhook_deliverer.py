"""Background deliverer for the webhook_deliveries outbox.

Claims due rows (SKIP LOCKED, so replicas take disjoint sets), builds the
Run-shaped payload from the persisted run, and POSTs via ``deliver_webhook``.
A failed row is rescheduled with exponential backoff and moved to ``dead`` once
attempts are exhausted; a row stuck in ``sending`` (deliverer crashed mid-flight)
is reclaimed after a stale window.
"""

import asyncio
import contextlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select, text

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.services.webhooks import deliver_webhook
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Internal tuning (retry/dead-letter policy is the env-configurable part, in
# WebhookSettings). _SENDING_STALE_SECONDS reclaims a row a crashed deliverer
# left mid-flight.
_POLL_INTERVAL_SECONDS = 5
_BATCH_SIZE = 100
_SENDING_STALE_SECONDS = 120

# Claim due pending rows plus any 'sending' row abandoned by a crashed deliverer.
_CLAIM_SQL = text(
    """
    UPDATE webhook_deliveries SET status = 'sending', updated_at = now()
    WHERE id IN (
        SELECT id FROM webhook_deliveries
        WHERE (status = 'pending' AND next_attempt_at <= now())
           OR (status = 'sending' AND updated_at < now() - make_interval(secs => :stale))
        ORDER BY next_attempt_at ASC
        LIMIT :batch
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id, run_id, url, attempts
    """
)
_MARK_DELIVERED_SQL = text(
    "UPDATE webhook_deliveries SET status = 'delivered', last_error = NULL, updated_at = now() WHERE id = :id"
)
_MARK_DEAD_SQL = text(
    "UPDATE webhook_deliveries SET status = 'dead', attempts = :attempts, last_error = :err, updated_at = now() "
    "WHERE id = :id"
)
_RESCHEDULE_SQL = text(
    "UPDATE webhook_deliveries SET status = 'pending', attempts = :attempts, last_error = :err, "
    "next_attempt_at = now() + make_interval(secs => :delay), updated_at = now() WHERE id = :id"
)


def _backoff_seconds(attempts: int) -> float:
    """Exponential backoff for the next retry, capped at 5 minutes."""
    return min(settings.webhook.WEBHOOK_BACKOFF_BASE_SECONDS * (2**attempts), 300.0)


class WebhookDeliverer:
    """Polls the outbox and delivers pending run-completion webhooks."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        if not settings.webhook.WEBHOOK_ENABLED:
            logger.info("Webhook deliverer disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Webhook deliverer started", interval=_POLL_INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Webhook deliverer stopped")

    async def _loop(self) -> None:
        interval = _POLL_INTERVAL_SECONDS
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in webhook deliverer tick")
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    async def _tick(self) -> None:
        claimed = await self._claim()
        if not claimed:
            return
        logger.info("Delivering webhooks", count=len(claimed))
        await asyncio.gather(*(self._deliver(row) for row in claimed), return_exceptions=True)

    @staticmethod
    async def _claim() -> Sequence[Any]:
        maker = _get_session_maker()
        async with maker() as session:
            result = await session.execute(
                _CLAIM_SQL,
                {
                    "stale": _SENDING_STALE_SECONDS,
                    "batch": _BATCH_SIZE,
                },
            )
            rows = result.fetchall()
            await session.commit()
        return rows

    async def _deliver(self, row: Any) -> None:
        payload = await self._build_payload(row.run_id)
        if payload is None:
            await self._finish(row.id, ok=False, attempts=row.attempts, error="run row missing")
            return
        ok = await deliver_webhook(row.url, payload)
        await self._finish(row.id, ok=ok, attempts=row.attempts, error=None if ok else "delivery attempt failed")

    @staticmethod
    async def _build_payload(run_id: str) -> dict[str, object] | None:
        maker = _get_session_maker()
        async with maker() as session:
            run = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
        if run is None:
            return None
        now = datetime.now(UTC).isoformat()
        return {
            "run_id": run.run_id,
            "thread_id": run.thread_id,
            "assistant_id": run.assistant_id,
            "status": run.status,
            "run_started_at": run.created_at.isoformat() if run.created_at else None,
            "run_ended_at": run.updated_at.isoformat() if run.updated_at else None,
            "webhook_sent_at": now,
            "values": run.output or {},
            "error": run.error_message,
            "metadata": run.metadata_dict or {},
        }

    async def _finish(self, delivery_id: str, *, ok: bool, attempts: int, error: str | None) -> None:
        maker = _get_session_maker()
        async with maker() as session:
            if ok:
                await session.execute(_MARK_DELIVERED_SQL, {"id": delivery_id})
            elif attempts + 1 >= settings.webhook.WEBHOOK_MAX_ATTEMPTS:
                await session.execute(_MARK_DEAD_SQL, {"id": delivery_id, "attempts": attempts + 1, "err": error})
                logger.warning("Webhook delivery dead-lettered", delivery_id=delivery_id, attempts=attempts + 1)
            else:
                await session.execute(
                    _RESCHEDULE_SQL,
                    {
                        "id": delivery_id,
                        "attempts": attempts + 1,
                        "err": error,
                        "delay": _backoff_seconds(attempts + 1),
                    },
                )
            await session.commit()


webhook_deliverer = WebhookDeliverer()
