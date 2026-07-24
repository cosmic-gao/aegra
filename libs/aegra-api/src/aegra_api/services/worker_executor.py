"""Redis-backed executor with concurrent async execution and lease-based crash recovery.

Production mode (REDIS_BROKER_ENABLED=true). Each worker loop dequeues
run_ids from Redis via BLPOP and spawns up to N_JOBS_PER_WORKER
concurrent asyncio tasks. Each task acquires a lease, executes the
graph with periodic heartbeats, and releases the lease on completion.
If a worker crashes, the lease expires and a background reaper
re-enqueues the run.
"""

import asyncio
import contextlib
import contextvars
import os
import re
import socket
from datetime import UTC, datetime, timedelta

import structlog
from asgi_correlation_id import correlation_id
from redis import RedisError
from redis import TimeoutError as RedisTimeoutError
from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from aegra_api.core.active_runs import active_runs
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.redis_manager import redis_manager
from aegra_api.models.enums import TERMINAL_RUN_STATUSES
from aegra_api.models.run_job import RunJob
from aegra_api.observability.span_enrichment import bind_run_trace_id, merge_run_metadata, set_trace_context
from aegra_api.services.base_executor import BaseExecutor
from aegra_api.services.run_executor import (
    _lease_loss_cancellations,
    _shutdown_cancellations,
    _timeout_cancellations,
    execute_run,
)
from aegra_api.services.run_status import finalize_run, update_run_status
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Bounded retries for the enqueue RPUSH; enqueue is best-effort since the row is
# already durable and the reaper's stuck-pending sweep backstops a lost push.
_ENQUEUE_MAX_ATTEMPTS = 3


def _is_valid_run_id(value: str) -> bool:
    """Check if a string is a valid UUID v4 hex format."""
    return bool(_RUN_ID_PATTERN.match(value))


def _db_lease_expiry(duration_seconds: int) -> ColumnElement[datetime]:
    """Lease expiry computed by the DB clock (``now()``), not the pod's wall clock.

    Multi-pod deployments have clock skew; deriving both the expiry write and the
    reaper's comparison from one authoritative clock — the database — closes the
    false-reclaim / double-run window that per-pod ``datetime.now()`` opens.
    Parameterized via a bound Interval, never string-concatenated SQL.
    """
    return func.now() + timedelta(seconds=duration_seconds)


class WorkerExecutor(BaseExecutor):
    """Dispatches runs via Redis List; workers consume with BLPOP + semaphore."""

    def __init__(self) -> None:
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._running = False
        self._instance_id = f"{socket.gethostname()}-{os.getpid()}"

    # ------------------------------------------------------------------
    # Submit (API side)
    # ------------------------------------------------------------------

    async def submit(self, job: RunJob) -> None:
        # Best-effort: the row is already durable, so the reaper backstops a
        # lost enqueue rather than the API failing the accepted run.
        run_id = job.identity.run_id
        try:
            await self._enqueue(run_id)
        except RedisError as exc:
            logger.warning("Enqueue failed; reaper will dispatch", run_id=run_id, error=str(exc))
            return
        logger.info("Enqueued run", run_id=run_id, queue=settings.worker.WORKER_QUEUE_KEY)

    @staticmethod
    async def _enqueue(run_id: str) -> None:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_ENQUEUE_MAX_ATTEMPTS),
            wait=wait_random_exponential(multiplier=0.1, max=2.0),
            retry=retry_if_exception_type(RedisError),
            reraise=True,
        ):
            with attempt:
                client = redis_manager.get_client()
                await client.rpush(settings.worker.WORKER_QUEUE_KEY, run_id)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Wait for completion (API side)
    # ------------------------------------------------------------------

    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        """Wait for a run to finish by polling a Redis done-key with DB fallback."""
        done_key = f"{settings.redis.REDIS_CHANNEL_PREFIX}done:{run_id}"
        client = redis_manager.get_client()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        poll_count = 0

        while loop.time() < deadline:
            try:
                if await client.exists(done_key):
                    return
            except RedisError:
                pass

            poll_count += 1
            if poll_count % 2 == 0 and await _is_run_terminal(run_id):
                return

            await asyncio.sleep(2.0)

        raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        count = settings.worker.WORKER_COUNT
        if count == 0:
            logger.warning(
                "WORKER_COUNT=0: no workers on this instance, runs will queue until another instance picks them up"
            )
        for idx in range(count):
            name = f"{self._instance_id}-worker-{idx}"
            task = asyncio.create_task(self._worker_loop(name))
            self._worker_tasks.append(task)

        max_concurrent = count * settings.worker.N_JOBS_PER_WORKER
        logger.info(
            "Worker executor started",
            worker_count=count,
            jobs_per_worker=settings.worker.N_JOBS_PER_WORKER,
            max_concurrent=max_concurrent,
            instance=self._instance_id,
        )

    async def stop(self) -> None:
        self._running = False
        drain_timeout = settings.worker.WORKER_DRAIN_TIMEOUT
        stranded: list[str] = []

        # Drain in-flight jobs. Whatever is still running after the window is
        # handed off, not killed: mark it shutdown-cancelled so execute_run
        # reverts it to pending (no terminal ``interrupted``) instead of writing
        # a dead state a rolling upgrade could never resume.
        if self._job_tasks:
            logger.info("Draining in-flight jobs", count=len(self._job_tasks))
            _, pending = await asyncio.wait(self._job_tasks, timeout=drain_timeout)
            if pending:
                stranded = _mark_shutdown_cancellations(pending)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        # Cancel worker loops
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._worker_tasks.clear()
        self._job_tasks.clear()
        # Re-enqueue the handed-off runs so another instance resumes them at once.
        # Best-effort: the reaper's stuck-pending sweep recovers any that miss.
        await _reenqueue_stranded(stranded)
        logger.info("Worker executor stopped", instance=self._instance_id, stranded=len(stranded))

    # ------------------------------------------------------------------
    # Worker loop (dequeue + spawn concurrent tasks)
    # ------------------------------------------------------------------

    async def _worker_loop(self, worker_name: str) -> None:
        """Dequeue run_ids and spawn concurrent execution tasks.

        Each worker loop manages a semaphore that limits concurrent runs
        to N_JOBS_PER_WORKER. When all slots are busy, the loop blocks
        on semaphore.acquire until a slot frees up.
        """
        n_jobs = settings.worker.N_JOBS_PER_WORKER
        if n_jobs <= 0:
            raise ValueError(f"N_JOBS_PER_WORKER must be >= 1, got {n_jobs}")
        semaphore = asyncio.Semaphore(n_jobs)
        logger.info(
            "Worker started",
            worker=worker_name,
            max_concurrent=settings.worker.N_JOBS_PER_WORKER,
        )

        while self._running:
            try:
                await semaphore.acquire()

                if not self._running:
                    semaphore.release()
                    break

                run_id = await self._dequeue()
                if run_id is None:
                    semaphore.release()
                    continue

                if not _is_valid_run_id(run_id):
                    logger.warning("Invalid run_id dequeued, discarding", value=run_id[:64])
                    semaphore.release()
                    continue

                task = asyncio.create_task(self._execute_and_release(run_id, worker_name, semaphore))
                self._job_tasks.add(task)
                task.add_done_callback(self._job_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception:
                semaphore.release()
                logger.exception("Unexpected error in worker loop", worker=worker_name)
                await asyncio.sleep(1.0)

        logger.info("Worker stopped", worker=worker_name)

    async def _execute_and_release(
        self,
        run_id: str,
        worker_name: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a run under its lease, then release the semaphore slot."""
        # active_runs lets cancel-on-disconnect / explicit cancel find this task.
        current_task = asyncio.current_task()
        if current_task is not None:
            active_runs[run_id] = current_task
        claimed = False
        try:
            claimed = await self._execute_with_lease(run_id, worker_name)
        except asyncio.CancelledError:
            logger.info("Job task cancelled", worker=worker_name, run_id=run_id)
            raise
        except Exception:
            logger.exception("Unexpected error in job execution", run_id=run_id)
            await _finalize_orphan(run_id, worker_name)
        finally:
            active_runs.pop(run_id, None)
            semaphore.release()
            # A declined claim frees nothing; waking then would re-enqueue this
            # very run into a BLPOP→fail→re-enqueue spin.
            if claimed:
                await _wake_thread_successor(run_id)

    # ------------------------------------------------------------------
    # Job execution (lease + heartbeat)
    # ------------------------------------------------------------------

    async def _dequeue(self) -> str | None:
        """BLPOP with 5s timeout. Falls back to Postgres polling if Redis is down."""
        try:
            client = redis_manager.get_client()
            result = await client.blpop(settings.worker.WORKER_QUEUE_KEY, timeout=5)  # type: ignore[arg-type]
            if result is None:
                return None
            return result[1]
        except RedisTimeoutError:
            # Idle expiry: a blocking BLPOP hit the socket timeout with no jobs.
            # Normal when the queue is empty, not a connectivity failure — re-loop.
            return None
        except RedisError as exc:
            logger.warning("Redis BLPOP failed, falling back to Postgres poll", error=str(exc))
            await asyncio.sleep(settings.worker.POSTGRES_POLL_INTERVAL_SECONDS)
            return await self._poll_postgres()

    async def _execute_with_lease(self, run_id: str, worker_name: str) -> bool:
        """Acquire lease, load job, execute with heartbeat + timeout.

        Returns True if this worker claimed and ran the run, False if the claim
        was declined (already claimed / thread busy) so the caller skips the
        successor wake.
        """
        lease_acquired_at = datetime.now(UTC)
        loaded = await _acquire_and_load(run_id, worker_name)
        if loaded is None:
            logger.debug("Lease not acquired or job missing, skipping", run_id=run_id, worker=worker_name)
            return False

        _restore_trace_context(run_id, loaded.job, loaded.trace)
        logger.info(
            "Worker picked up run",
            worker=worker_name,
            run_id=run_id,
            graph_id=loaded.job.identity.graph_id,
        )
        # Wrap execute_run in a task so the heartbeat can cancel it on
        # lease loss, preventing double execution by a second worker.
        job_task = asyncio.create_task(execute_run(loaded.job))
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(run_id, worker_name, job_task=job_task),
            context=contextvars.copy_context(),
        )
        timed_out = False

        try:
            # shield so timeout cancels the wait, not job_task; we flag
            # _timeout_cancellations before cancelling so execute_run defers finalize.
            await asyncio.wait_for(asyncio.shield(job_task), timeout=settings.worker.BG_JOB_TIMEOUT_SECS)
        except TimeoutError:
            timed_out = True
            _timeout_cancellations.add(run_id)
            logger.error(
                "Job exceeded timeout, cancelling",
                worker=worker_name,
                run_id=run_id,
                timeout_secs=settings.worker.BG_JOB_TIMEOUT_SECS,
            )
            job_task.cancel()
        except asyncio.CancelledError:
            logger.info("Worker job cancelled", worker=worker_name, run_id=run_id)
            raise
        except Exception:
            logger.exception("Worker job failed", worker=worker_name, run_id=run_id)
        finally:
            if not job_task.done():
                job_task.cancel()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(job_task, heartbeat_task, return_exceptions=True)
            # Single authoritative timeout write, after job_task has settled.
            if timed_out:
                await _finalize_timeout(run_id)
            await _release_lease(run_id, worker_name)

            elapsed = (datetime.now(UTC) - lease_acquired_at).total_seconds()
            logger.info(
                "Worker finished run",
                worker=worker_name,
                run_id=run_id,
                execution_seconds=round(elapsed, 2),
            )
        return True

    @staticmethod
    async def _poll_postgres() -> str | None:
        """Pick the oldest pending, unclaimed run from Postgres."""
        maker = _get_session_maker()
        async with maker() as session:
            run_id = await session.scalar(
                select(RunORM.run_id)
                .where(RunORM.status == "pending", RunORM.claimed_by.is_(None))
                .order_by(RunORM.created_at.asc())
                .limit(1)
            )
            return run_id


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _get_thread_id_for_run(run_id: str) -> str | None:
    """Look up the thread_id for a run. Returns None if the row is missing."""
    maker = _get_session_maker()
    async with maker() as session:
        return await session.scalar(select(RunORM.thread_id).where(RunORM.run_id == run_id))


async def _finalize_timeout(run_id: str) -> None:
    # Single authoritative timeout write: execute_run's timeout settlement defers here.
    _TIMEOUT_ERROR = "Job exceeded maximum execution time"
    thread_id = await _get_thread_id_for_run(run_id)
    if thread_id is not None:
        await finalize_run(run_id, thread_id, status="timeout", thread_status="error", error=_TIMEOUT_ERROR)
    else:
        await update_run_status(run_id, "timeout", error=_TIMEOUT_ERROR)


async def _finalize_orphan(run_id: str, worker_name: str) -> None:
    # Guard on claimed_by so a run the reaper already reassigned isn't clobbered.
    maker = _get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name))
    if run_orm is None or run_orm.status in TERMINAL_RUN_STATUSES:
        return
    await finalize_run(run_id, run_orm.thread_id, status="error", thread_status="error", error="Worker execution error")
    await _release_lease(run_id, worker_name)


def _mark_shutdown_cancellations(pending: set[asyncio.Task[None]]) -> list[str]:
    """Flag drained-but-still-running jobs as shutdown-cancelled; return their run_ids.

    Maps each pending job task back to its run_id via the ``active_runs`` registry
    (populated at the start of ``_execute_and_release``, before the lease is
    acquired, so any task holding a lease is present here). Recording the run_ids
    in ``_shutdown_cancellations`` before cancelling makes execute_run's handler
    revert them to pending rather than write a terminal status.
    """
    by_task = {task: run_id for run_id, task in active_runs.items()}
    run_ids = [by_task[task] for task in pending if task in by_task]
    _shutdown_cancellations.update(run_ids)
    return run_ids


async def _reenqueue_stranded(run_ids: list[str]) -> None:
    """Re-enqueue shutdown-stranded runs so another instance resumes them promptly.

    Best-effort: on Redis failure (including a not-yet/already-closed client) the
    reaper's stuck-pending sweep still recovers them, just slower. A stray
    re-enqueue of a run that actually finalized during drain is harmless — the
    claim requires ``status='pending'`` and simply discards it.
    """
    if not run_ids:
        return
    queue_key = settings.worker.WORKER_QUEUE_KEY
    try:
        client = redis_manager.get_client()
        for run_id in run_ids:
            await client.rpush(queue_key, run_id)  # type: ignore[arg-type]
    except (RedisError, RuntimeError) as exc:
        logger.warning("Shutdown re-enqueue failed; reaper will recover", error=str(exc), run_ids=run_ids)


async def _wake_thread_successor(done_run_id: str) -> None:
    """Re-enqueue the oldest pending run on the just-finished run's thread.

    A run deferred by the per-thread claim gate left no sentinel in the queue;
    this nudges a worker to claim the successor as soon as the predecessor
    frees the thread (multitask enqueue), instead of waiting for the idle tick.
    Best-effort: on any failure the idle tick / Postgres poll still recovers it.
    """
    try:
        maker = _get_session_maker()
        thread_subq = select(RunORM.thread_id).where(RunORM.run_id == done_run_id).scalar_subquery()
        async with maker() as session:
            successor = await session.scalar(
                select(RunORM.run_id)
                .where(
                    RunORM.thread_id == thread_subq,
                    RunORM.status == "pending",
                    RunORM.claimed_by.is_(None),
                )
                .order_by(RunORM.created_at.asc())
                .limit(1)
            )
        if successor is None:
            return
        client = redis_manager.get_client()
        await client.rpush(settings.worker.WORKER_QUEUE_KEY, successor)  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("Successor wakeup failed; idle tick will recover", run_id=done_run_id, error=str(exc))


# ------------------------------------------------------------------
# Lease operations (module-level for reuse by LeaseReaper)
# ------------------------------------------------------------------


class _LoadedRun:
    """RunJob plus raw trace metadata from execution_params."""

    __slots__ = ("job", "trace")

    def __init__(self, job: RunJob, trace: dict[str, str]) -> None:
        self.job = job
        self.trace = trace


# Correlated "is another run already running on this thread?" check.
_RunBusy = aliased(RunORM)
_thread_has_running_run = (
    select(_RunBusy.run_id).where(_RunBusy.thread_id == RunORM.thread_id, _RunBusy.status == "running").exists()
)


async def _acquire_and_load(run_id: str, worker_name: str) -> _LoadedRun | None:
    """Claim a pending run and load its job, enforcing per-thread serialization.

    The claim only succeeds when no other run on the same thread is already
    running (multitask enqueue). A concurrent claim that slips past that check
    is caught by the ``uq_runs_one_running_per_thread`` unique index →
    IntegrityError → treated as "not claimed". Missing execution_params
    (corruption / pre-migration row) releases the claim and errors the run.
    """
    maker = _get_session_maker()
    try:
        async with maker() as session:
            result = await session.execute(
                update(RunORM)
                .where(
                    RunORM.run_id == run_id,
                    RunORM.status == "pending",
                    RunORM.claimed_by.is_(None),
                    ~_thread_has_running_run,
                )
                .values(
                    claimed_by=worker_name,
                    lease_expires_at=_db_lease_expiry(settings.worker.LEASE_DURATION_SECONDS),
                    status="running",
                )
            )
            if result.rowcount == 0:  # type: ignore[union-attr]
                await session.rollback()
                return None

            run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
            await session.commit()

            if run_orm is None or run_orm.execution_params is None:
                logger.warning(
                    "Run not found or missing execution_params after lease, releasing claim",
                    run_id=run_id,
                    worker=worker_name,
                )
                await session.execute(
                    update(RunORM)
                    .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
                    .values(
                        claimed_by=None,
                        lease_expires_at=None,
                        status="error",
                        error_message="Run missing execution_params (data corruption or pre-migration row)",
                    )
                )
                await session.commit()
                return None

            job = RunJob.from_run_orm(run_orm)
            trace = run_orm.execution_params.get("trace", {})
            return _LoadedRun(job=job, trace=trace)
    except IntegrityError:
        # Lost the one-running-per-thread race; run stays pending and is
        # re-enqueued when the winning run finishes (see _wake_thread_successor).
        return None


async def _release_lease(run_id: str, worker_name: str) -> None:
    """Clear lease fields after job completion, only if this worker still owns the lease."""
    maker = _get_session_maker()
    async with maker() as session:
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
            .values(claimed_by=None, lease_expires_at=None)
        )
        await session.commit()


async def _heartbeat_loop(
    run_id: str,
    worker_name: str,
    *,
    job_task: asyncio.Task[None] | None = None,
) -> None:
    """Extend the lease periodically and honor a persistent cancel request.

    One DB round-trip per tick does both jobs via ``UPDATE ... RETURNING``:
    - Extends the lease using the DB clock (``now()``), so multi-pod clock skew
      cannot open a false-reclaim / double-run window.
    - Reads ``cancel_requested`` in the same statement. A cancel persisted by the
      API is honored here even when the pub/sub signal was lost — cross-instance
      and independent of Redis (which stays only an accelerator).

    Lease loss (no row updated → another worker owns it) cancels ``job_task`` as
    a lease-loss handoff; a cancel request cancels it as a user cancel. The two
    differ: only lease loss marks ``_lease_loss_cancellations`` so execute_run
    skips finalize; a cancel request must finalize ``interrupted``.

    Self-fencing: when the DB is unreachable for >= the lease duration the lease
    has expired on the DB clock and the reaper may reassign the run, so we cancel
    ``job_task`` as a lease loss to close the double-execution window.
    """
    interval = settings.worker.HEARTBEAT_INTERVAL_SECONDS
    duration = settings.worker.LEASE_DURATION_SECONDS
    maker = _get_session_maker()
    loop = asyncio.get_running_loop()
    last_renew = loop.time()

    while True:
        await asyncio.sleep(interval)
        try:
            async with maker() as session:
                result = await session.execute(
                    update(RunORM)
                    .where(RunORM.run_id == run_id, RunORM.claimed_by == worker_name)
                    .values(lease_expires_at=_db_lease_expiry(duration))
                    .returning(RunORM.cancel_requested)
                )
                row = result.first()
                await session.commit()
        except Exception:
            unrenewed = loop.time() - last_renew
            if unrenewed >= duration and job_task is not None and not job_task.done():
                logger.error(
                    "Lease unrenewable beyond its duration, self-fencing to prevent double execution",
                    run_id=run_id,
                    worker=worker_name,
                    unrenewed_seconds=round(unrenewed, 1),
                )
                _lease_loss_cancellations.add(run_id)
                job_task.cancel()
                return
            logger.warning(
                "Heartbeat lease extension failed",
                run_id=run_id,
                worker=worker_name,
                unrenewed_seconds=round(unrenewed, 1),
            )
            continue

        last_renew = loop.time()
        if row is None:
            logger.warning(
                "Lease lost, cancelling job to prevent double execution",
                run_id=run_id,
                worker=worker_name,
            )
            if job_task is not None and not job_task.done():
                _lease_loss_cancellations.add(run_id)
                job_task.cancel()
            return
        if row.cancel_requested:
            logger.info("Cancel requested, stopping job", run_id=run_id, worker=worker_name)
            if job_task is not None and not job_task.done():
                job_task.cancel()
            return
        logger.debug("Lease extended", run_id=run_id, worker=worker_name)


async def _is_run_terminal(run_id: str) -> bool:
    """Check if a run has reached a terminal state in the DB."""
    maker = _get_session_maker()
    async with maker() as session:
        run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
        if run_orm is None:
            return True
        return run_orm.status in TERMINAL_RUN_STATUSES


def _restore_trace_context(run_id: str, job: RunJob, trace: dict[str, str]) -> None:
    """Restore OTEL and structlog trace context for a worker-executed run.

    Clears previous context first to prevent bleed between concurrent
    jobs processed by the same worker.  User-supplied ``run_metadata`` is
    merged with the system runtime keys; system keys win on collision —
    see :func:`merge_run_metadata`.
    """
    structlog.contextvars.clear_contextvars()

    original_request_id = trace.get("correlation_id", "")
    if original_request_id:
        correlation_id.set(original_request_id)

    system_metadata: dict[str, str | int | float | bool] = {
        "run_id": run_id,
        "thread_id": job.identity.thread_id,
        "graph_id": job.identity.graph_id,
    }
    # Gate on non-empty: requests without an upstream correlation-id header
    # leave ``original_request_id`` as ``""`` — including the empty string
    # would emit a noisy ``langfuse.trace.metadata.original_request_id=""``
    # attribute on every such trace.
    if original_request_id:
        system_metadata["original_request_id"] = original_request_id
    set_trace_context(
        user_id=job.user.identity,
        session_id=job.identity.thread_id,
        trace_name=job.identity.graph_id,
        metadata=merge_run_metadata(job.run_metadata, system_metadata),
    )
    # Worker context has no ambient span (no HTTP request), so the run's root span
    # is already a true root — bind the derived trace id, no detach needed.
    bind_run_trace_id(run_id)

    structlog.contextvars.bind_contextvars(
        run_id=run_id,
        thread_id=job.identity.thread_id,
        graph_id=job.identity.graph_id,
        user_id=job.user.identity,
        original_request_id=original_request_id,
    )
