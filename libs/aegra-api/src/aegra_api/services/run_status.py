"""Run and thread status management.

Provides the database-level status update operations used by both the
API layer (cancel, interrupt) and the execution layer (run_executor,
worker_executor). Extracted from api/runs.py to eliminate the circular
dependency where service code imported from the API module.
"""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from sqlalchemy import CursorResult, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import ThreadState as ThreadStateORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.settings import settings
from aegra_api.utils.status_compat import validate_run_status, validate_thread_status

logger = structlog.getLogger(__name__)
_serializer = GeneralSerializer()


def _values_hash(values: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def _thread_state_update_columns(
    *, unchanged: bool, values: dict[str, Any], interrupts: dict[str, Any], values_hash: str, now: datetime
) -> dict[str, Any]:
    """Columns to write on conflict; skip the big ``values`` when it's unchanged."""
    if unchanged:
        return {"interrupts": interrupts, "updated_at": now}
    return {"values": values, "interrupts": interrupts, "values_hash": values_hash, "updated_at": now}


async def materialize_thread_state(
    session: AsyncSession,
    thread_id: str,
    values: dict[str, Any],
    interrupts: dict[str, Any],
) -> None:
    """Upsert the thread's latest state into ``thread_state``, skipping no-op writes.

    Gated by ``THREAD_STATE_MATERIALIZE``. A values hash lets us skip rewriting the
    (potentially large) ``values`` column when the state is unchanged, only
    refreshing the cheap ``interrupts``/``updated_at`` columns.
    """
    if not settings.checkpointer.THREAD_STATE_MATERIALIZE:
        return
    new_hash = _values_hash(values)
    current = await session.scalar(
        select(ThreadStateORM.values_hash).where(ThreadStateORM.thread_id == thread_id)
    )
    now = datetime.now(UTC)
    insert = pg_insert(ThreadStateORM).values(
        thread_id=thread_id, values=values, interrupts=interrupts, values_hash=new_hash, updated_at=now
    )
    update_set = _thread_state_update_columns(
        unchanged=current == new_hash, values=values, interrupts=interrupts, values_hash=new_hash, now=now
    )
    await session.execute(insert.on_conflict_do_update(index_elements=["thread_id"], set_=update_set))


async def update_run_status(
    run_id: str,
    status: str,
    *,
    output: Any = None,
    error: str | None = None,
) -> None:
    """Persist a run's status to the database.

    Opens a short-lived session to avoid holding a connection during
    long-running graph execution.
    """
    validated = validate_run_status(status)
    maker = _get_session_maker()
    async with maker() as session:
        values: dict[str, Any] = {
            "status": validated,
            "updated_at": datetime.now(UTC),
        }
        if output is not None:
            values["output"], _ = _safe_serialize(output, run_id)
        if error is not None:
            values["error_message"] = error

        logger.info("Updating run status", run_id=run_id, status=validated)
        await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(**values))
        await session.commit()


async def set_thread_status(session: AsyncSession, thread_id: str, status: str) -> None:
    """Update a thread's status column.

    Does NOT commit — the caller controls the transaction boundary.
    This allows thread status and run updates to share a single commit.
    """
    validated = validate_thread_status(status)
    result = cast(
        CursorResult,
        await session.execute(
            update(ThreadORM)
            .where(ThreadORM.thread_id == thread_id)
            .values(status=validated, updated_at=datetime.now(UTC))
        ),
    )
    if result.rowcount == 0:
        raise ValueError(f"Thread '{thread_id}' not found")


async def finalize_run(
    run_id: str,
    thread_id: str,
    *,
    status: str,
    thread_status: str,
    output: Any = None,
    error: str | None = None,
    interrupts: dict[str, list[Any]] | None = None,
) -> None:
    """Update run status + thread status in a single transaction.

    Batches two UPDATE statements into one DB round-trip instead of
    opening separate sessions for update_run_status and set_thread_status.
    ``interrupts`` is the SDK task-keyed map (``{task_id: [...]}``); it defaults
    to ``{}`` so completing a run clears any pending interrupts on the thread.
    """
    validated_run = validate_run_status(status)
    validated_thread = validate_thread_status(thread_status)
    maker = _get_session_maker()

    run_values: dict[str, Any] = {
        "status": validated_run,
        "updated_at": datetime.now(UTC),
    }
    thread_values: dict[str, Any] = {
        "status": validated_thread,
        "updated_at": datetime.now(UTC),
    }
    state: tuple[dict[str, Any], dict[str, Any]] | None = None
    if output is not None:
        serialized, ok = _safe_serialize(output, run_id)
        run_values["output"] = serialized
        # Materialize into thread_state (not the thread row) only for genuinely
        # serialized dict state — never the _safe_serialize failure fallback.
        if ok and isinstance(serialized, dict):
            state = (serialized, interrupts or {})
    if error is not None:
        run_values["error_message"] = error

    async with maker() as session:
        await session.execute(update(RunORM).where(RunORM.run_id == run_id).values(**run_values))
        await session.execute(update(ThreadORM).where(ThreadORM.thread_id == thread_id).values(**thread_values))
        if state is not None:
            await materialize_thread_state(session, thread_id, state[0], state[1])
        await session.commit()

    logger.info("Finalized run", run_id=run_id, status=validated_run, thread_status=validated_thread)


def _safe_serialize(output: Any, run_id: str) -> tuple[Any, bool]:
    """Serialize output; return ``(value, ok)`` where ``ok`` is False on a fallback.

    Callers persist ``value`` to ``run.output`` regardless, but must not
    materialize the failure fallback as thread state.
    """
    try:
        return _serializer.serialize(output), True
    except Exception as exc:
        logger.warning("Output serialization failed", run_id=run_id, error=str(exc))
        return {"error": "Output serialization failed", "original_type": str(type(output))}, False
