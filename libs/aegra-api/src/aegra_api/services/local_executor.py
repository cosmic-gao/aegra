"""In-process executor using asyncio tasks.

Used in development mode (REDIS_BROKER_ENABLED=false). Runs execute as
background coroutines in the same event loop as the API server. Per-thread
serialization mirrors the worker claim gate so multitask ``enqueue`` holds a
second run until the first on that thread finishes (and so concurrent runs
never collide on the one-running-per-thread invariant).
"""

import asyncio
import contextlib
from collections import defaultdict, deque

import structlog

from aegra_api.core.active_runs import active_runs
from aegra_api.models.run_job import RunJob
from aegra_api.observability.span_enrichment import make_run_trace_context
from aegra_api.services.base_executor import BaseExecutor

logger = structlog.getLogger(__name__)


class LocalExecutor(BaseExecutor):
    """Runs graphs as local asyncio tasks (single-instance dev mode)."""

    def __init__(self) -> None:
        self._running_threads: set[str] = set()
        self._pending: dict[str, deque[RunJob]] = defaultdict(deque)

    async def submit(self, job: RunJob) -> None:
        thread_id = job.identity.thread_id
        if thread_id in self._running_threads:
            self._pending[thread_id].append(job)
            logger.info("Queued run behind running thread", run_id=job.identity.run_id, thread_id=thread_id)
            return
        self._running_threads.add(thread_id)
        self._start(job)

    def _start(self, job: RunJob) -> None:
        trace_ctx = make_run_trace_context(
            job.identity.run_id,
            job.identity.thread_id,
            job.identity.graph_id,
            job.user.identity,
            extra_metadata=job.run_metadata,
        )
        task = asyncio.create_task(self._run_then_next(job), context=trace_ctx)
        active_runs[job.identity.run_id] = task
        logger.info("Submitted run to local executor", run_id=job.identity.run_id, task_id=id(task))

    async def _run_then_next(self, job: RunJob) -> None:
        # Deferred import: run_executor imports services that reference the
        # executor singleton, creating a circular chain at module level.
        from aegra_api.services.run_executor import execute_run

        try:
            await execute_run(job)
        finally:
            thread_id = job.identity.thread_id
            active_runs.pop(job.identity.run_id, None)
            queue = self._pending.get(thread_id)
            if queue:
                self._start(queue.popleft())
                if not queue:
                    self._pending.pop(thread_id, None)
            else:
                self._running_threads.discard(thread_id)
                self._pending.pop(thread_id, None)

    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        task = active_runs.get(run_id)
        if task is None:
            return
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def start(self) -> None:
        logger.info("Local executor started (in-process asyncio tasks)")

    async def stop(self) -> None:
        tasks_to_cancel = [task for task in active_runs.values() if not task.done()]
        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            logger.info("Draining cancelled tasks", count=len(tasks_to_cancel))
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        logger.info("Local executor stopped")
