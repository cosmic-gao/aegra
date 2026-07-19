"""Heartbeat keep-alive utilities for join/wait endpoints.

Provides an async generator that streams periodic ``\\n`` heartbeat bytes
to keep HTTP connections alive through proxies and load balancers, then
yields the final JSON result when the run completes.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from sqlalchemy import select

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.services.executor import executor
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Terminal run states — used by join/wait to skip waiting
TERMINAL_STATES = {"success", "error", "interrupted"}


def wrap_run_result(status: str, output: dict[str, Any] | None, error_message: str | None) -> dict[str, Any]:
    """Shape a terminal run's payload for the non-streaming wait/join response.

    On error, emit the SDK's in-band ``{"__error__": {...}}`` envelope so
    ``client.runs.wait(raise_error=True)`` raises instead of returning silently.
    """
    if status == "error":
        return {"__error__": {"error": "error", "message": error_message or "Run failed"}}
    return output or {}


async def read_run_output(
    run_id: str,
    thread_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Open a short-lived DB session and read the run's terminal wait/join payload."""
    maker = _get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(
            select(RunORM).where(
                RunORM.run_id == run_id,
                RunORM.thread_id == thread_id,
                RunORM.user_id == user_id,
            )
        )
        if not run_orm:
            return {}
        return wrap_run_result(run_orm.status, run_orm.output, run_orm.error_message)


def encode_output(output: dict[str, Any]) -> bytes:
    """Serialize a run output dict to JSON bytes."""
    return json.dumps(output, default=str).encode()


async def heartbeat_wait_body(
    run_id: str,
    thread_id: str,
    user_id: str,
    *,
    timeout: float,
) -> AsyncIterator[bytes]:
    """Async generator that keeps the HTTP connection alive while waiting.

    Yields ``b"\\n"`` heartbeat bytes every ``KEEPALIVE_INTERVAL_SECS``
    until the run finishes, then yields the JSON result. Leading whitespace
    is ignored by JSON parsers so clients parse the concatenated body normally.
    """
    done = asyncio.Event()

    async def _wait_for_run() -> None:
        try:
            await executor.wait_for_completion(run_id, timeout=timeout)
        except TimeoutError:
            logger.warning("heartbeat_wait timeout", run_id=run_id, timeout=timeout)
        except Exception:
            logger.exception("heartbeat_wait error", run_id=run_id)
        finally:
            done.set()

    task = asyncio.create_task(_wait_for_run())
    interval = settings.app.KEEPALIVE_INTERVAL_SECS
    try:
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=interval)
            except TimeoutError:
                yield b"\n"
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        else:
            # Surface any exception so asyncio doesn't warn about
            # "Task exception was never retrieved"
            if task.exception() is not None:
                logger.error(
                    "heartbeat_wait task failed",
                    run_id=run_id,
                    exc_info=task.exception(),
                )

    output = await read_run_output(run_id, thread_id, user_id)
    yield encode_output(output)
