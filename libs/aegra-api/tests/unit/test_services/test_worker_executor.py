"""Unit tests for worker_executor service."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import ConnectionError as RedisConnectionError
from redis import TimeoutError as RedisTimeoutError

from aegra_api.core.active_runs import active_runs
from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.observability.span_enrichment import _run_trace_id
from aegra_api.services.worker_executor import (
    WorkerExecutor,
    _acquire_and_load,
    _heartbeat_loop,
    _is_run_terminal,
    _is_valid_run_id,
    _LoadedRun,
    _release_lease,
    _restore_trace_context,
)

MODULE = "aegra_api.services.worker_executor"


def _make_session_maker(session: AsyncMock) -> MagicMock:
    """Wrap a mock session in a context-manager-returning maker."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    maker = MagicMock(return_value=ctx)
    return maker


def _make_run_job(
    *,
    run_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    thread_id: str = "11111111-2222-3333-4444-555555555555",
    graph_id: str = "test-graph",
) -> RunJob:
    """Create a minimal RunJob for testing."""
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id=thread_id, graph_id=graph_id),
        user=User(identity="test-user"),
        execution=RunExecution(),
        behavior=RunBehavior(),
    )


def _make_run_orm(
    *,
    run_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    thread_id: str = "11111111-2222-3333-4444-555555555555",
    status: str = "pending",
    execution_params: dict | None = None,
) -> MagicMock:
    """Create a mock RunORM row."""
    orm = MagicMock()
    orm.run_id = run_id
    orm.thread_id = thread_id
    orm.status = status
    orm.execution_params = execution_params or {
        "graph_id": "test-graph",
        "user": {"identity": "test-user", "is_authenticated": True, "permissions": []},
        "execution": {
            "input_data": {},
            "config": {},
            "context": {},
            "stream_mode": None,
            "checkpoint": None,
            "command": None,
        },
        "behavior": {
            "interrupt_before": None,
            "interrupt_after": None,
            "multitask_strategy": None,
            "subgraphs": False,
        },
        "trace": {"correlation_id": "req-123"},
    }
    return orm


# ------------------------------------------------------------------
# _is_valid_run_id
# ------------------------------------------------------------------


class TestIsValidRunId:
    def test_returns_true_for_valid_uuid(self) -> None:
        assert _is_valid_run_id("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee") is True

    def test_returns_false_for_empty_string(self) -> None:
        assert _is_valid_run_id("") is False

    def test_returns_false_for_non_uuid_string(self) -> None:
        assert _is_valid_run_id("not-a-uuid") is False

    def test_returns_false_for_uuid_with_wrong_format(self) -> None:
        # Too short in last segment
        assert _is_valid_run_id("aaaaaaaa-bbbb-cccc-dddd-eeeeeeee") is False
        # Uppercase (pattern is lowercase hex only)
        assert _is_valid_run_id("AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE") is False


# ------------------------------------------------------------------
# _acquire_and_load
# ------------------------------------------------------------------


class TestAcquireAndLoad:
    @pytest.mark.asyncio
    async def test_returns_loaded_run_when_lease_acquired(self) -> None:
        run_orm = _make_run_orm()
        session = AsyncMock()

        # First execute: UPDATE (lease acquisition)
        update_result = MagicMock()
        update_result.rowcount = 1
        # Second call: scalar (SELECT run)
        session.execute = AsyncMock(return_value=update_result)
        session.scalar = AsyncMock(return_value=run_orm)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _acquire_and_load("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        assert result is not None
        assert isinstance(result, _LoadedRun)
        assert result.job.identity.run_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert result.trace == {"correlation_id": "req-123"}
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_lease_already_taken(self) -> None:
        session = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 0
        session.execute = AsyncMock(return_value=update_result)
        session.rollback = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _acquire_and_load("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        assert result is None
        session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_execution_params_is_none(self) -> None:
        run_orm = _make_run_orm()
        run_orm.execution_params = None

        session = AsyncMock()
        update_result = MagicMock()
        update_result.rowcount = 1
        session.execute = AsyncMock(return_value=update_result)
        session.scalar = AsyncMock(return_value=run_orm)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _acquire_and_load("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        assert result is None


# ------------------------------------------------------------------
# _release_lease
# ------------------------------------------------------------------


class TestReleaseLease:
    @pytest.mark.asyncio
    async def test_clears_claimed_by_and_lease_expires_at(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            await _release_lease("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "test-worker")

        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()


# ------------------------------------------------------------------
# _heartbeat_loop
# ------------------------------------------------------------------


class TestHeartbeatLoop:
    @pytest.mark.asyncio
    async def test_extends_lease_on_each_iteration(self) -> None:
        session = AsyncMock()
        # Lease extension UPDATE...RETURNING(cancel_requested): a matched row with
        # cancel_requested=False means "lease still ours, no cancel" → keep looping.
        result = MagicMock()
        result.first.return_value = MagicMock(cancel_requested=False)
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        call_count = 0

        async def counting_sleep(delay: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError
            # Don't actually sleep

        with (
            patch(f"{MODULE}._get_session_maker", return_value=maker),
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.asyncio.sleep", side_effect=counting_sleep),
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
            mock_settings.worker.LEASE_DURATION_SECONDS = 30

            with pytest.raises(asyncio.CancelledError):
                await _heartbeat_loop("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        # One iteration completed before cancellation on second sleep
        assert session.execute.await_count == 1
        assert session.commit.await_count == 1

    @pytest.mark.asyncio
    async def test_continues_loop_on_db_error(self) -> None:
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB connection lost"))
        maker = _make_session_maker(session)

        call_count = 0

        async def counting_sleep(delay: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise asyncio.CancelledError

        with (
            patch(f"{MODULE}._get_session_maker", return_value=maker),
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.asyncio.sleep", side_effect=counting_sleep),
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
            mock_settings.worker.LEASE_DURATION_SECONDS = 30

            with pytest.raises(asyncio.CancelledError):
                await _heartbeat_loop("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "worker-0")

        # Loop continued despite DB errors (2 iterations before cancel on 3rd sleep)
        assert session.execute.await_count == 2


# ------------------------------------------------------------------
# _is_run_terminal
# ------------------------------------------------------------------


class TestHeartbeatCancelMarker:
    """P0-5: a persisted cancel is honored by the heartbeat even without pub/sub."""

    @pytest.mark.asyncio
    async def test_cancels_job_on_marker_without_lease_loss_flag(self) -> None:
        from aegra_api.services.run_executor import _lease_loss_cancellations

        session = AsyncMock()
        result = MagicMock()
        result.first.return_value = MagicMock(cancel_requested=True)
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        job_task = MagicMock()
        job_task.done.return_value = False
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        async def one_tick(_delay: float) -> None:
            return None

        with (
            patch(f"{MODULE}._get_session_maker", return_value=maker),
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.asyncio.sleep", side_effect=one_tick),
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
            mock_settings.worker.LEASE_DURATION_SECONDS = 30
            await _heartbeat_loop(run_id, "worker-0", job_task=job_task)

        # Job cancelled as a USER cancel (execute_run must finalize interrupted),
        # so the run_id must NOT be flagged as a lease-loss handoff.
        job_task.cancel.assert_called_once()
        assert run_id not in _lease_loss_cancellations

    @pytest.mark.asyncio
    async def test_lease_loss_flags_set_and_cancels(self) -> None:
        from aegra_api.services.run_executor import _lease_loss_cancellations

        session = AsyncMock()
        result = MagicMock()
        result.first.return_value = None  # no row updated → lease lost
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        maker = _make_session_maker(session)

        job_task = MagicMock()
        job_task.done.return_value = False
        run_id = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"

        async def one_tick(_delay: float) -> None:
            return None

        with (
            patch(f"{MODULE}._get_session_maker", return_value=maker),
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.asyncio.sleep", side_effect=one_tick),
        ):
            mock_settings.worker.HEARTBEAT_INTERVAL_SECONDS = 1
            mock_settings.worker.LEASE_DURATION_SECONDS = 30
            try:
                await _heartbeat_loop(run_id, "worker-0", job_task=job_task)
                job_task.cancel.assert_called_once()
                assert run_id in _lease_loss_cancellations
            finally:
                _lease_loss_cancellations.discard(run_id)


class TestMarkShutdownCancellations:
    """P0-4: stop() flags drained-but-running jobs so their handlers hand off."""

    def test_maps_pending_tasks_to_run_ids_via_active_runs(self) -> None:
        from aegra_api.services.run_executor import _shutdown_cancellations
        from aegra_api.services.worker_executor import _mark_shutdown_cancellations

        t1 = MagicMock()
        t2 = MagicMock()
        untracked = MagicMock()
        active_runs["run-a"] = t1
        active_runs["run-b"] = t2
        try:
            result = _mark_shutdown_cancellations({t1, untracked})
            assert result == ["run-a"]
            assert "run-a" in _shutdown_cancellations
            assert "run-b" not in _shutdown_cancellations
        finally:
            active_runs.pop("run-a", None)
            active_runs.pop("run-b", None)
            _shutdown_cancellations.discard("run-a")


class TestReenqueueStranded:
    @pytest.mark.asyncio
    async def test_rpushes_each_run_id(self) -> None:
        from aegra_api.services.worker_executor import _reenqueue_stranded

        client = AsyncMock()
        with (
            patch(f"{MODULE}.redis_manager") as mock_rm,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.return_value = client
            await _reenqueue_stranded(["run-1", "run-2"])

        assert client.rpush.await_count == 2

    @pytest.mark.asyncio
    async def test_noop_on_empty_list(self) -> None:
        from aegra_api.services.worker_executor import _reenqueue_stranded

        with patch(f"{MODULE}.redis_manager") as mock_rm:
            await _reenqueue_stranded([])

        mock_rm.get_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_best_effort_when_redis_client_unavailable(self) -> None:
        from aegra_api.services.worker_executor import _reenqueue_stranded

        with (
            patch(f"{MODULE}.redis_manager") as mock_rm,
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.logger.warning") as mock_warning,
        ):
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"
            mock_rm.get_client.side_effect = RuntimeError("Redis not initialized")
            # Must not raise — the reaper's stuck-pending sweep recovers these.
            await _reenqueue_stranded(["run-1"])

        mock_warning.assert_called_once()


class TestDbLeaseExpiry:
    """P1-4: lease expiry is computed by the DB clock, parameterized."""

    def test_expiry_is_based_on_db_now(self) -> None:
        from aegra_api.services.worker_executor import _db_lease_expiry

        sql = str(_db_lease_expiry(30)).lower()
        assert "now()" in sql


class TestIsRunTerminal:
    @pytest.mark.asyncio
    async def test_returns_true_for_success(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "success"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_for_error(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "error"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_for_interrupted(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "interrupted"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_run_not_found(self) -> None:
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=None)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_pending(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "pending"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_running(self) -> None:
        run_orm = MagicMock()
        run_orm.status = "running"
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=run_orm)
        maker = _make_session_maker(session)

        with patch(f"{MODULE}._get_session_maker", return_value=maker):
            result = await _is_run_terminal("run-1")

        assert result is False


# ------------------------------------------------------------------
# _restore_trace_context
# ------------------------------------------------------------------


class TestRestoreTraceContext:
    def test_sets_structlog_context_vars(self) -> None:
        job = _make_run_job()
        trace = {"correlation_id": "req-abc"}

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        mock_set_trace.assert_called_once()
        call_kwargs = mock_set_trace.call_args.kwargs
        assert call_kwargs["user_id"] == "test-user"
        assert call_kwargs["session_id"] == "11111111-2222-3333-4444-555555555555"
        assert call_kwargs["trace_name"] == "test-graph"

    def test_sets_run_trace_id_from_run_id(self) -> None:
        """The run's OTEL trace id is derived from run_id so it equals the trace
        the downstream attaches scores/feedback to (LangSmith parity)."""
        job = _make_run_job()
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        _run_trace_id.set(None)
        _restore_trace_context(run_id, job, {"correlation_id": "req-abc"})
        assert _run_trace_id.get() == uuid.UUID(run_id).int

    def test_clears_previous_context_before_setting_new(self) -> None:
        job = _make_run_job()
        trace = {"correlation_id": "req-abc"}
        call_order: list[str] = []

        with (
            patch(f"{MODULE}.structlog.contextvars.clear_contextvars", side_effect=lambda: call_order.append("clear")),
            patch(f"{MODULE}.set_trace_context", side_effect=lambda **kw: call_order.append("set_trace")),
            patch(
                f"{MODULE}.structlog.contextvars.bind_contextvars", side_effect=lambda **kw: call_order.append("bind")
            ),
            patch(f"{MODULE}.correlation_id"),
        ):
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        assert call_order == ["clear", "set_trace", "bind"]

    def test_user_metadata_merged_with_system_keys(self) -> None:
        """job.run_metadata is merged into the trace context metadata."""
        job = RunJob(
            identity=RunIdentity(
                run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                thread_id="11111111-2222-3333-4444-555555555555",
                graph_id="test-graph",
            ),
            user=User(identity="test-user"),
            run_metadata={"tenant": "acme", "feature_flag": True},
        )
        trace = {"correlation_id": "req-abc"}

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        metadata = mock_set_trace.call_args.kwargs["metadata"]
        assert metadata["tenant"] == "acme"
        assert metadata["feature_flag"] is True
        # System keys still present
        assert metadata["run_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert metadata["thread_id"] == "11111111-2222-3333-4444-555555555555"
        assert metadata["graph_id"] == "test-graph"
        assert metadata["original_request_id"] == "req-abc"

    def test_user_metadata_cannot_override_system_keys(self) -> None:
        """Reserved system keys win on collision; user spoof is dropped."""
        job = RunJob(
            identity=RunIdentity(
                run_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                thread_id="11111111-2222-3333-4444-555555555555",
                graph_id="test-graph",
            ),
            user=User(identity="test-user"),
            run_metadata={"run_id": "spoofed", "tenant": "acme"},
        )
        trace = {"correlation_id": "req-abc"}

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        metadata = mock_set_trace.call_args.kwargs["metadata"]
        assert metadata["run_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert metadata["tenant"] == "acme"

    def test_empty_run_metadata_with_correlation_id_keeps_four_system_keys(self) -> None:
        """When a correlation-id is present, ``original_request_id`` is
        included in the metadata alongside the three runtime keys."""
        job = _make_run_job()  # run_metadata defaults to {}
        trace = {"correlation_id": "req-abc"}

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        metadata = mock_set_trace.call_args.kwargs["metadata"]
        assert set(metadata.keys()) == {"run_id", "thread_id", "graph_id", "original_request_id"}

    def test_missing_correlation_id_omits_original_request_id(self) -> None:
        """Requests without an upstream correlation-id header should not produce
        a ``langfuse.trace.metadata.original_request_id=""`` empty attribute."""
        job = _make_run_job()
        trace: dict[str, str] = {}  # no correlation_id

        with patch(f"{MODULE}.set_trace_context") as mock_set_trace:
            _restore_trace_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", job, trace)

        metadata = mock_set_trace.call_args.kwargs["metadata"]
        assert "original_request_id" not in metadata
        assert set(metadata.keys()) == {"run_id", "thread_id", "graph_id"}


# ------------------------------------------------------------------
# WorkerExecutor.submit
# ------------------------------------------------------------------


class TestWorkerExecutorSubmit:
    @pytest.mark.asyncio
    async def test_pushes_run_id_to_redis(self) -> None:
        mock_client = AsyncMock()
        mock_client.rpush = AsyncMock()

        job = _make_run_job()

        with (
            patch(f"{MODULE}.redis_manager") as mock_redis,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_redis.get_client.return_value = mock_client
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"

            executor = WorkerExecutor()
            await executor.submit(job)

        mock_client.rpush.assert_awaited_once_with("aegra:jobs", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    @pytest.mark.asyncio
    async def test_enqueue_failure_is_best_effort(self) -> None:
        """P0-3: a Redis outage on enqueue must not raise — the row is durably
        pending and the reaper's stuck-pending sweep dispatches it."""
        mock_client = AsyncMock()
        mock_client.rpush = AsyncMock(side_effect=RedisConnectionError("down"))

        job = _make_run_job()

        with (
            patch(f"{MODULE}.redis_manager") as mock_redis,
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}._ENQUEUE_MAX_ATTEMPTS", 2),
        ):
            mock_redis.get_client.return_value = mock_client
            mock_settings.worker.WORKER_QUEUE_KEY = "aegra:jobs"

            executor = WorkerExecutor()
            await executor.submit(job)  # must not raise

        # Retried up to the cap before giving up.
        assert mock_client.rpush.await_count == 2


# ------------------------------------------------------------------
# WorkerExecutor.wait_for_completion
# ------------------------------------------------------------------


class TestWorkerExecutorWaitForCompletion:
    @pytest.mark.asyncio
    async def test_done_key_uses_configured_channel_prefix(self) -> None:
        """Regression: done-key must derive from REDIS_CHANNEL_PREFIX, not a hardcoded string."""
        mock_client = AsyncMock()
        mock_client.exists = AsyncMock(return_value=True)

        with (
            patch(f"{MODULE}.redis_manager") as mock_redis,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_redis.get_client.return_value = mock_client
            mock_settings.redis.REDIS_CHANNEL_PREFIX = "aegra:agent-foo:run:"

            executor = WorkerExecutor()
            await executor.wait_for_completion("run-1")

        mock_client.exists.assert_awaited_once_with("aegra:agent-foo:run:done:run-1")


# ------------------------------------------------------------------
# WorkerExecutor.start / stop
# ------------------------------------------------------------------


class TestWorkerExecutorStart:
    @pytest.mark.asyncio
    async def test_creates_worker_tasks(self) -> None:
        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.worker.WORKER_COUNT = 2
            mock_settings.worker.N_JOBS_PER_WORKER = 5

            executor = WorkerExecutor()
            # Patch _worker_loop to be a no-op coroutine
            executor._worker_loop = AsyncMock()  # type: ignore[method-assign]
            await executor.start()

        assert len(executor._worker_tasks) == 2
        # Clean up tasks
        for t in executor._worker_tasks:
            t.cancel()
        await asyncio.gather(*executor._worker_tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_warns_when_worker_count_zero(self) -> None:
        with (
            patch(f"{MODULE}.settings") as mock_settings,
            patch(f"{MODULE}.logger") as mock_logger,
        ):
            mock_settings.worker.WORKER_COUNT = 0
            mock_settings.worker.N_JOBS_PER_WORKER = 5

            executor = WorkerExecutor()
            await executor.start()

        mock_logger.warning.assert_called_once()
        assert "WORKER_COUNT=0" in mock_logger.warning.call_args[0][0]
        assert len(executor._worker_tasks) == 0


class TestWorkerExecutorStop:
    @pytest.mark.asyncio
    async def test_cancels_worker_tasks(self) -> None:
        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.worker.WORKER_DRAIN_TIMEOUT = 1.0

            executor = WorkerExecutor()

            # Create some fake tasks
            async def hang_forever() -> None:
                await asyncio.sleep(9999)

            task1 = asyncio.create_task(hang_forever())
            task2 = asyncio.create_task(hang_forever())
            executor._worker_tasks = [task1, task2]

            await executor.stop()

        assert task1.cancelled()
        assert task2.cancelled()
        assert len(executor._worker_tasks) == 0


# ------------------------------------------------------------------
# _execute_and_release
# ------------------------------------------------------------------


class TestExecuteAndRelease:
    @pytest.mark.asyncio
    async def test_registers_in_active_runs_and_cleans_up(self) -> None:
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()  # Pre-acquire so we can verify release

        executor = WorkerExecutor()

        registered_in_active: bool = False

        async def mock_execute_with_lease(rid: str, wn: str) -> None:
            nonlocal registered_in_active
            registered_in_active = run_id in active_runs

        executor._execute_with_lease = AsyncMock(side_effect=mock_execute_with_lease)  # type: ignore[method-assign]

        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 60

            await executor._execute_and_release(run_id, "worker-0", semaphore)

        # Task was registered during execution
        assert registered_in_active is True
        # Cleaned up after execution
        assert run_id not in active_runs
        # Semaphore was released
        assert not semaphore.locked()

    @pytest.mark.asyncio
    async def test_wakes_successor_only_when_claimed(self) -> None:
        """P1-5: a declined claim (thread busy) must NOT re-enqueue itself — that
        would spin BLPOP→claim-fail→re-enqueue. Successor wake fires only after a
        run that actually held (and thus freed) the thread."""
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        executor = WorkerExecutor()

        # Claim declined → _execute_with_lease returns False → no wake.
        executor._execute_with_lease = AsyncMock(return_value=False)  # type: ignore[method-assign]
        with patch(f"{MODULE}._wake_thread_successor", new_callable=AsyncMock) as mock_wake:
            await executor._execute_and_release(run_id, "worker-0", asyncio.Semaphore(1))
        mock_wake.assert_not_awaited()

        # Claim succeeded → _execute_with_lease returns True → wake fires once.
        executor._execute_with_lease = AsyncMock(return_value=True)  # type: ignore[method-assign]
        with patch(f"{MODULE}._wake_thread_successor", new_callable=AsyncMock) as mock_wake:
            await executor._execute_and_release(run_id, "worker-0", asyncio.Semaphore(1))
        mock_wake.assert_awaited_once_with(run_id)

    @pytest.mark.asyncio
    async def test_orphaned_error_finalized_on_unexpected_exception(self) -> None:
        """P0-4: an error escaping _execute_with_lease finalizes the run as error
        instead of stranding it 'running' until the reaper."""
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        executor = WorkerExecutor()
        executor._execute_with_lease = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        with patch(f"{MODULE}._finalize_orphan", new_callable=AsyncMock) as mock_orphan:
            await executor._execute_and_release(run_id, "worker-0", asyncio.Semaphore(1))

        mock_orphan.assert_awaited_once_with(run_id, "worker-0")


class TestExecuteWithLease:
    @pytest.mark.asyncio
    async def test_cancels_job_task_in_finally(self) -> None:
        """Regression: when _execute_with_lease is cancelled (worker shutdown),
        the inner job_task must also be cancelled to prevent orphaned execution,
        and the CancelledError must propagate (not be swallowed)."""
        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        executor = WorkerExecutor()

        job_task_was_cancelled = False

        async def long_running_job(job: object) -> None:
            nonlocal job_task_was_cancelled
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                job_task_was_cancelled = True
                raise

        mock_loaded = MagicMock(spec=_LoadedRun)
        mock_loaded.job = _make_run_job()
        mock_loaded.trace = {}

        with (
            patch(f"{MODULE}._acquire_and_load", new_callable=AsyncMock, return_value=mock_loaded),
            patch(f"{MODULE}._restore_trace_context"),
            patch(f"{MODULE}.execute_run", side_effect=long_running_job),
            patch(f"{MODULE}._heartbeat_loop", new_callable=AsyncMock),
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock),
        ):
            task = asyncio.create_task(executor._execute_with_lease(run_id, "worker-0"))
            await asyncio.sleep(0.05)  # Let it start
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert job_task_was_cancelled, "job_task must be cancelled when _execute_with_lease is cancelled"

    @pytest.mark.asyncio
    async def test_timeout_finalizes_once_as_timeout(self) -> None:
        """P0-2: a job exceeding BG_JOB_TIMEOUT_SECS is cancelled and finalized as
        a single 'timeout' write (not 'error', not a double write), with the run
        flagged in _timeout_cancellations so execute_run defers the finalize."""
        from aegra_api.services.run_executor import _timeout_cancellations

        run_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        thread_id = "tttttttt-tttt-tttt-tttt-tttttttttttt"
        executor = WorkerExecutor()
        flagged_during_run = False

        async def slow_job(job: object) -> None:
            nonlocal flagged_during_run
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                flagged_during_run = run_id in _timeout_cancellations
                raise

        mock_loaded = MagicMock(spec=_LoadedRun)
        mock_loaded.job = _make_run_job()
        mock_loaded.trace = {}

        with (
            patch(f"{MODULE}._acquire_and_load", new_callable=AsyncMock, return_value=mock_loaded),
            patch(f"{MODULE}._restore_trace_context"),
            patch(f"{MODULE}.execute_run", side_effect=slow_job),
            patch(f"{MODULE}._heartbeat_loop", new_callable=AsyncMock),
            patch(f"{MODULE}._get_thread_id_for_run", new_callable=AsyncMock, return_value=thread_id),
            patch(f"{MODULE}.finalize_run", new_callable=AsyncMock) as mock_finalize,
            patch(f"{MODULE}._release_lease", new_callable=AsyncMock) as mock_release,
            patch(f"{MODULE}.settings") as mock_settings,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 0.01
            try:
                claimed = await executor._execute_with_lease(run_id, "worker-0")
            finally:
                _timeout_cancellations.discard(run_id)

        assert claimed is True
        assert flagged_during_run, "run must be flagged in _timeout_cancellations before job cancel"
        mock_finalize.assert_awaited_once_with(
            run_id,
            thread_id,
            status="timeout",
            thread_status="error",
            error="Job exceeded maximum execution time",
        )
        mock_release.assert_awaited_once_with(run_id, "worker-0")


class TestDequeue:
    """Tests for WorkerExecutor._dequeue BLPOP handling."""

    def _make_executor_with_blpop(self, blpop: AsyncMock) -> WorkerExecutor:
        executor = WorkerExecutor()
        executor._poll_postgres = AsyncMock(return_value="from-postgres")  # type: ignore[method-assign]
        self._client = MagicMock()
        self._client.blpop = blpop
        return executor

    @pytest.mark.asyncio
    async def test_returns_run_id_on_queue_hit(self) -> None:
        blpop = AsyncMock(return_value=("aegra:worker:queue", "run-123"))
        executor = self._make_executor_with_blpop(blpop)

        with patch(f"{MODULE}.redis_manager.get_client", return_value=self._client):
            result = await executor._dequeue()

        assert result == "run-123"
        executor._poll_postgres.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_none_when_blpop_returns_none(self) -> None:
        blpop = AsyncMock(return_value=None)
        executor = self._make_executor_with_blpop(blpop)

        with patch(f"{MODULE}.redis_manager.get_client", return_value=self._client):
            result = await executor._dequeue()

        assert result is None
        executor._poll_postgres.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_idle_socket_timeout_returns_none_without_fallback(self) -> None:
        """A blocking BLPOP that hits the socket timeout raises redis TimeoutError
        (a RedisError subclass). That is a normal idle expiry, not a connectivity
        failure: it must return None silently, never poll Postgres (GH #bug)."""
        blpop = AsyncMock(side_effect=RedisTimeoutError("Timeout reading from redis:6379"))
        executor = self._make_executor_with_blpop(blpop)

        with (
            patch(f"{MODULE}.redis_manager.get_client", return_value=self._client),
            patch(f"{MODULE}.logger.warning") as mock_warning,
        ):
            result = await executor._dequeue()

        assert result is None
        executor._poll_postgres.assert_not_awaited()
        mock_warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_connection_error_falls_back_to_postgres(self) -> None:
        """A genuine Redis failure (connection lost) must still warn and fall
        back to the Postgres poll so jobs are not stranded."""
        blpop = AsyncMock(side_effect=RedisConnectionError("Connection refused"))
        executor = self._make_executor_with_blpop(blpop)

        with (
            patch(f"{MODULE}.redis_manager.get_client", return_value=self._client),
            patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock),
            patch(f"{MODULE}.logger.warning") as mock_warning,
        ):
            result = await executor._dequeue()

        assert result == "from-postgres"
        executor._poll_postgres.assert_awaited_once()
        mock_warning.assert_called_once()
