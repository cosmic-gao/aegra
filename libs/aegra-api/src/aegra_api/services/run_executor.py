"""Graph run execution logic.

Single source of truth for executing a graph run. Both LocalExecutor
(asyncio.create_task) and WorkerExecutor (Redis BLPOP) call
`execute_run`. All database and broker interactions are delegated to
`run_status` and `streaming_service` respectively.
"""

import asyncio
from typing import Any

import structlog
from sqlalchemy import update

from aegra_api.core.active_runs import active_runs
from aegra_api.core.auth_ctx import with_auth_ctx
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.core.redis_manager import redis_manager
from aegra_api.core.serializers.langgraph import LangGraphSerializer
from aegra_api.models.run_job import RunJob
from aegra_api.services.broker import broker_manager
from aegra_api.services.event_streaming.native_stream import stream_native_v3_events
from aegra_api.services.graph_streaming import stream_graph_events
from aegra_api.services.langgraph_service import create_run_config, get_langgraph_service
from aegra_api.services.run_status import finalize_run, update_run_status
from aegra_api.services.streaming_service import streaming_service
from aegra_api.settings import settings
from aegra_api.utils.run_utils import map_command_to_langgraph

logger = structlog.getLogger(__name__)

_DEFAULT_STREAM_MODES = ["values"]

# Run IDs whose cancellation was triggered by lease loss (not user action).
# When a heartbeat detects lease loss, it adds the run_id here before
# cancelling the job task. execute_run's CancelledError handler checks
# this set to skip finalize_run and SSE signaling — the reaper has already
# re-enqueued the run and another worker will execute it. Without this,
# the old worker would write status="interrupted" and send an SSE end event,
# prematurely closing client streams and potentially overwriting the new
# worker's status.
_lease_loss_cancellations: set[str] = set()

# Run IDs whose cancellation was triggered by graceful worker shutdown (rolling
# upgrade / scale-down), not user action. WorkerExecutor.stop() adds the run_id
# here before cancelling the drained job task. execute_run's CancelledError
# handler reverts the run to ``pending`` (clearing the lease) instead of writing
# a terminal ``interrupted`` — the run is still viable and resumes from its last
# checkpoint on another instance. Mirrors the ``_lease_loss_cancellations`` idiom.
_shutdown_cancellations: set[str] = set()

# Run IDs cancelled by the job-timeout guard. The worker marks the run before
# cancelling, so execute_run skips its own finalize (the worker writes the single
# ``timeout`` status) and the webhook, but still tears down the stream.
_timeout_cancellations: set[str] = set()


async def execute_run(job: RunJob) -> None:
    """Execute a graph run, stream events to the broker, and update DB.

    Handles the full lifecycle: status transitions, event streaming,
    interrupt detection, cancellation, and error signaling.
    """
    run_id = job.identity.run_id
    thread_id = job.identity.thread_id
    # finalize_run enqueues the outbox row transactionally when it wins, so a
    # double-execution loser can't double-post and no in-process task is needed.
    webhook = job.execution.webhook
    handed_off = False

    try:
        await update_run_status(run_id, "running")

        final_output = await _stream_graph(job)

        if final_output.has_interrupt:
            await finalize_run(
                run_id,
                thread_id,
                status="interrupted",
                thread_status="interrupted",
                output=final_output.data,
                interrupts=final_output.interrupts,
                state_values=final_output.state_values,
                webhook=webhook,
            )
        else:
            await finalize_run(
                run_id,
                thread_id,
                status="success",
                thread_status="idle",
                output=final_output.data,
                state_values=final_output.state_values,
                webhook=webhook,
            )

    except asyncio.CancelledError:
        settlement = await _handle_cancel(run_id, thread_id, webhook)
        handed_off = settlement == "handed_off"
        raise
    except Exception as exc:
        logger.exception("Run failed", run_id=run_id)
        safe_message = f"{type(exc).__name__}: execution failed"
        # No output: a failed run has no new state to materialize into thread_state.
        await finalize_run(run_id, thread_id, status="error", thread_status="error", error=str(exc), webhook=webhook)
        await _best_effort_signal(streaming_service.signal_run_error, run_id, safe_message, type(exc).__name__)
    else:
        status = "interrupted" if final_output.has_interrupt else "success"
        await _best_effort_signal(_signal_end_event, run_id, status)
    finally:
        _lease_loss_cancellations.discard(run_id)
        _shutdown_cancellations.discard(run_id)
        _timeout_cancellations.discard(run_id)
        active_runs.pop(run_id, None)
        # A handed-off run (lease loss or graceful shutdown) is owned by another
        # worker now — skip finalize signaling, done-key, and cleanup.
        if not handed_off:
            await streaming_service.cleanup_run(run_id)
            await _signal_run_done(run_id)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


async def _best_effort_signal(fn: Any, *args: Any) -> None:
    """Call a signaling function, logging but not raising on failure.

    Signaling (end events, cancel/error notifications) must not override
    an already-committed DB status if it fails.
    """
    try:
        await fn(*args)
    except Exception:
        logger.warning("Signal failed (best-effort, DB status already committed)", fn=fn.__name__)


async def _handle_cancel(run_id: str, thread_id: str, webhook: str | None) -> str:
    """Settle a CancelledError by its source. Returns the settlement kind.

    Four sources, three settlement kinds:
    - lease loss / graceful shutdown → ``"handed_off"``: another worker owns the
      run now (or it was reverted to pending), so skip finalize AND teardown.
    - job timeout → ``"timeout"``: the worker writes the authoritative ``timeout``
      status, so skip finalize here and suppress the webhook, but still tear down.
    - user cancel → ``"cancelled"``: finalize ``interrupted`` and signal SSE.
    """
    if run_id in _lease_loss_cancellations:
        logger.info("Lease-loss cancel, skipping finalize", run_id=run_id)
        return "handed_off"
    if run_id in _shutdown_cancellations:
        await _release_for_recovery(run_id)
        logger.info("Shutdown cancel, reverted run to pending for recovery", run_id=run_id)
        return "handed_off"
    if run_id in _timeout_cancellations:
        logger.info("Timeout cancel, worker finalizes as timeout", run_id=run_id)
        return "timeout"
    # No output: a cancel has no new state, so don't materialize thread_state
    # (an empty output would wipe the thread's real materialized values).
    await finalize_run(run_id, thread_id, status="interrupted", thread_status="idle", webhook=webhook)
    await _best_effort_signal(streaming_service.signal_run_cancelled, run_id)
    return "cancelled"


async def _release_for_recovery(run_id: str) -> None:
    """Revert a shutdown-cancelled run to ``pending`` and drop its lease.

    Guarded on ``status='running'`` so a run that finalized during the drain
    window is left untouched. A re-enqueue from WorkerExecutor.stop (or, failing
    that, the reaper's stuck-pending sweep) then hands it to another instance,
    which resumes from the last checkpoint. No retry budget is charged — a
    rolling deploy is not a failure.
    """
    maker = _get_session_maker()
    async with maker() as session:
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == run_id, RunORM.status == "running")
            .values(status="pending", claimed_by=None, lease_expires_at=None)
        )
        await session.commit()


class _GraphResult:
    """Accumulates output and interrupt state during graph streaming.

    ``data`` is the run's stream output (persisted to ``run.output``).
    ``state_values`` is the checkpointer snapshot used to materialize thread
    state — the single source of truth, distinct from the stream output which
    is empty for non-``values`` stream modes. ``None`` means the state read
    failed, so finalize skips materialization rather than wiping real state.
    """

    __slots__ = ("data", "has_interrupt", "interrupts", "state_values")

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.has_interrupt: bool = False
        self.interrupts: dict[str, list[Any]] = {}
        self.state_values: dict[str, Any] | None = None


async def _read_state(
    graph: Any, run_config: dict[str, Any], run_id: str
) -> tuple[dict[str, Any] | None, dict[str, list[Any]]]:
    """Read the checkpointer snapshot: serialized values + task-keyed interrupts.

    The checkpointer is the single source of truth for materialized thread
    state. Returns ``(values, interrupts_map)`` where ``interrupts_map`` matches
    the SDK ``Thread.interrupts`` shape. On read failure returns ``(None, {})``
    so finalize skips materialization instead of overwriting real state with an
    empty dict.
    """
    try:
        state = await graph.aget_state(run_config)
    except Exception as exc:
        logger.warning("Could not read checkpointer state", run_id=run_id, error=str(exc))
        return None, {}
    serializer = LangGraphSerializer()
    values = serializer.serialize(getattr(state, "values", None) or {})
    return values, serializer.build_interrupts_map(state)


async def _stream_graph(job: RunJob) -> _GraphResult:
    """Load the graph, stream events to the broker, return final output."""
    run_config = _build_run_config(job)
    execution_input = _resolve_input(job)
    stream_modes = _resolve_stream_modes(job.execution.stream_mode)

    langgraph_service = get_langgraph_service()
    result = _GraphResult()

    async with (
        langgraph_service.get_graph(
            job.identity.graph_id,
            config=run_config,
            access_context="threads.create_run",
            user=job.user,
            context=job.execution.context,
        ) as graph,
        with_auth_ctx(job.user, job.user.permissions),  # type: ignore[arg-type]
    ):
        if job.execution.event_streaming_v2:
            await _stream_native_v2(job, graph, execution_input, run_config, result)
        else:
            await _stream_legacy(job, graph, execution_input, run_config, stream_modes, result)

        # Read the checkpointer snapshot once — for every run, not just interrupts.
        # It is the single source of truth for materialized thread state (the
        # stream output is empty under non-``values`` modes) and carries the
        # authoritative task-keyed interrupts (the streamed __interrupt__ channel
        # is flat). Done inside the graph context.
        result.state_values, result.interrupts = await _read_state(graph, run_config, job.identity.run_id)

    return result


async def _stream_legacy(
    job: RunJob,
    graph: Any,
    execution_input: Any,
    run_config: dict[str, Any],
    stream_modes: list[str],
    result: _GraphResult,
) -> None:
    """Stream via the v1 producer (legacy SSE endpoints)."""
    run_id = job.identity.run_id
    async for event_type, event_data in stream_graph_events(
        graph=graph,
        input_data=execution_input,
        config=run_config,
        stream_mode=stream_modes,
        context=job.execution.context,
        subgraphs=job.behavior.subgraphs,
        durability=job.execution.durability,
        on_checkpoint=lambda _: None,
        on_task_result=lambda _: None,
    ):
        event_id = await broker_manager.allocate_event_id(run_id)
        await streaming_service.put_to_broker(run_id, event_id, (event_type, event_data))

        if isinstance(event_data, dict) and "__interrupt__" in event_data:
            result.has_interrupt = True
        if event_type.startswith("values"):
            result.data = event_data


async def _stream_native_v2(
    job: RunJob,
    graph: Any,
    execution_input: Any,
    run_config: dict[str, Any],
    result: _GraphResult,
) -> None:
    """Stream via the native v3 protocol producer (Agent Protocol v2).

    Each native event goes into the broker as ``(method, protocol_event)``.
    Interrupts ride on a ``values`` event's ``params.interrupts``; the final
    state is the last ``values`` event's ``params.data``.
    """
    run_id = job.identity.run_id
    async for method, event in stream_native_v3_events(
        graph=graph,
        input_data=execution_input,
        config=run_config,
        context=job.execution.context,
    ):
        event_id = await broker_manager.allocate_event_id(run_id)
        await streaming_service.put_to_broker(run_id, event_id, (method, event))

        params = event.get("params", {})
        data = params.get("data")
        # Interrupts ride on values.params.interrupts, or as an __interrupt__ key
        # in a values/updates payload (the path session.py routes to input.requested).
        if params.get("interrupts") or (isinstance(data, dict) and "__interrupt__" in data):
            result.has_interrupt = True
        if method == "values" and isinstance(data, dict):
            result.data = data


def _build_run_config(job: RunJob) -> dict[str, Any]:
    """Assemble the LangGraph run config from a RunJob."""
    config = create_run_config(
        job.identity.run_id,
        job.identity.thread_id,
        job.user,
        additional_config=job.execution.config,
        checkpoint=job.execution.checkpoint,
    )
    if job.behavior.interrupt_before is not None:
        items = job.behavior.interrupt_before
        config["interrupt_before"] = items if isinstance(items, list) else [items]
    if job.behavior.interrupt_after is not None:
        items = job.behavior.interrupt_after
        config["interrupt_after"] = items if isinstance(items, list) else [items]
    return config


def _resolve_input(job: RunJob) -> Any:
    """Return graph input — either raw data or a LangGraph Command."""
    if job.execution.command is not None:
        return map_command_to_langgraph(job.execution.command)
    return job.execution.input_data


def _resolve_stream_modes(stream_mode: str | list[str] | None) -> list[str]:
    """Normalize stream_mode to a list."""
    if stream_mode is None:
        return _DEFAULT_STREAM_MODES.copy()
    if isinstance(stream_mode, str):
        return [stream_mode]
    return list(stream_mode)


# ------------------------------------------------------------------
# End event (tells SSE consumers the stream is finished)
# ------------------------------------------------------------------


async def _signal_end_event(run_id: str, status: str) -> None:
    """Publish an 'end' event to the broker so SSE consumers close cleanly.

    Without this, the SSE connection hangs after the last data event
    because the broker never signals completion on success/interrupt.
    Error and cancel paths already send end events via signal_run_error
    and signal_run_cancelled respectively.
    """
    broker = broker_manager.get_broker(run_id)
    if broker is None or broker.is_finished():
        return

    event_id = await broker_manager.allocate_event_id(run_id)
    await broker.put(event_id, ("end", {"status": status}))


# ------------------------------------------------------------------
# Done signal (Redis key for fast completion polling)
# ------------------------------------------------------------------

_DONE_KEY_TTL_SECONDS = 3600


async def _signal_run_done(run_id: str) -> None:
    """Set a Redis key indicating the run has finished.

    WorkerExecutor.wait_for_completion polls this key instead of
    subscribing to the broker's event channel. Simpler, no subscription
    race, no message parsing. Falls back silently if Redis unavailable.
    """
    try:
        client = redis_manager.get_client()
        done_key = f"{settings.redis.REDIS_CHANNEL_PREFIX}done:{run_id}"
        await client.set(done_key, "1", ex=_DONE_KEY_TTL_SECONDS)
    except Exception:
        logger.debug("Redis done-key set failed (non-critical)", run_id=run_id)
