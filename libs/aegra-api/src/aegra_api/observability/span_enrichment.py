"""Per-request OTEL span enrichment via context variables.

Enriches *every* span (not just the root) with Langfuse (``langfuse.*``) and
Phoenix/OpenInference (``user.id``, ``session.id``) attributes: the Langfuse v4
immutable-observation model has no server-side trace->child join, so each
observation must carry its own user/session/name.
"""

import contextvars
import hashlib
import logging
import uuid
from typing import Any

import structlog
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator

logger = logging.getLogger(__name__)

# OTEL accepts only these; the SDK silently drops other types, so we filter
# here and warn instead of letting drops happen invisibly.
_PRIMITIVE_ATTR_TYPES: tuple[type, ...] = (str, int, float, bool)

# Per-request context variable holding span attributes to inject.
# None means no trace context is set; on_start() is a no-op in that case.
_trace_attrs: contextvars.ContextVar[dict[str, str | int | float | bool] | None] = contextvars.ContextVar(
    "aegra_otel_trace_attrs", default=None
)

# Per-request OTEL trace id derived from run_id — LangSmith parity: a trace is
# identified by its run, so downstream attaches scores/feedback by run_id.
_run_trace_id: contextvars.ContextVar[int | None] = contextvars.ContextVar("aegra_otel_run_trace_id", default=None)


def trace_id_from_run(run_id: str) -> int | None:
    """Map run_id to a 128-bit OTEL trace id.

    UUID run_ids map 1:1 (trace hex == run_id without dashes); other ids hash to
    128 bits. Returns None for a 0 result — 0 is an invalid OTEL trace id.
    """
    try:
        derived = uuid.UUID(run_id).int
    except ValueError:
        derived = int.from_bytes(hashlib.blake2b(run_id.encode(), digest_size=16).digest(), "big")
    return derived or None


class RunIdGenerator(IdGenerator):
    """Forces the run root span's trace id to :func:`trace_id_from_run` when a run
    context is active, random otherwise. Span ids stay random."""

    def __init__(self) -> None:
        self._fallback = RandomIdGenerator()

    def generate_span_id(self) -> int:
        return self._fallback.generate_span_id()

    def generate_trace_id(self) -> int:
        return _run_trace_id.get() or self._fallback.generate_trace_id()


class SpanEnrichmentProcessor(SpanProcessor):
    """Sets the per-request ``_trace_attrs`` on every span in ``on_start``."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        attrs = _trace_attrs.get()
        if not attrs:
            return
        for key, value in attrs.items():
            span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def set_trace_context(
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    trace_name: str | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> None:
    """Populate the per-request span-attribute context variable.

    Must run inside the task/context that executes the graph. Sets Langfuse
    (``langfuse.*``) and Phoenix (``user.id``/``session.id``) aliases so one
    call works for either backend; ``metadata`` lands under
    ``langfuse.trace.metadata.<key>`` to stay queryable as a trace field.
    """
    attrs: dict[str, str | int | float | bool] = {}
    if user_id:
        attrs["langfuse.user.id"] = user_id
        attrs["user.id"] = user_id
    if session_id:
        attrs["langfuse.session.id"] = session_id
        attrs["session.id"] = session_id
    if trace_name:
        attrs["langfuse.trace.name"] = trace_name
    if metadata:
        for key, value in metadata.items():
            attrs[f"langfuse.trace.metadata.{key}"] = value
    _trace_attrs.set(attrs or None)


def merge_run_metadata(
    extra_metadata: dict[str, Any] | None,
    system_metadata: dict[str, str | int | float | bool],
) -> dict[str, str | int | float | bool]:
    """Merge user metadata with system runtime keys; system wins on collision.

    Collisions and non-primitive values (OTEL accepts str/int/float/bool only)
    are skipped with a warning so the rejected key is visible upstream.
    """
    if not extra_metadata:
        return dict(system_metadata)
    merged: dict[str, str | int | float | bool] = {}
    for key, value in extra_metadata.items():
        if key in system_metadata:
            logger.warning(
                "User metadata key '%s' overridden by system value",
                key,
            )
            continue
        if not isinstance(value, _PRIMITIVE_ATTR_TYPES):
            logger.warning(
                "User metadata key '%s' has non-primitive type %s; dropping "
                "(OTEL attributes accept str/int/float/bool only)",
                key,
                type(value).__name__,
            )
            continue
        merged[key] = value
    merged.update(system_metadata)
    return merged


def bind_run_trace_id(run_id: str) -> None:
    """Set the run's derived trace id on the current context so its root span
    adopts it (see :class:`RunIdGenerator`). Worker + inline paths both call this."""
    _run_trace_id.set(trace_id_from_run(run_id))


def _detach_ambient_span() -> None:
    """Drop any ambient OTEL span so the run's first span is a true root and the
    trace id comes from :class:`RunIdGenerator`, not parent inheritance. Needed
    where a run executes inside a request span — LocalExecutor's copied context
    and the A2A/MCP inline paths."""
    otel_context.attach(otel_trace.set_span_in_context(otel_trace.INVALID_SPAN))


def bind_run_trace_context(
    *,
    run_id: str,
    thread_id: str,
    graph_id: str,
    user_identity: str | None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Bind span-enrichment + structlog + run trace id into the CURRENT context,
    rooting the trace so its id is :func:`trace_id_from_run`.

    For inline ``ainvoke``/``astream`` paths (A2A, MCP) that skip the executor's
    task-based setup (cf. :func:`make_run_trace_context`). System keys win over
    ``extra_metadata``; ``user_id`` is bound only when present.
    """
    system_metadata: dict[str, str | int | float | bool] = {
        "run_id": run_id,
        "thread_id": thread_id,
        "graph_id": graph_id,
    }
    set_trace_context(
        user_id=user_identity,
        session_id=thread_id,
        trace_name=graph_id,
        metadata=merge_run_metadata(extra_metadata, system_metadata),
    )
    bind_run_trace_id(run_id)
    _detach_ambient_span()
    structlog_bindings: dict[str, str] = {
        "run_id": run_id,
        "thread_id": thread_id,
        "graph_id": graph_id,
    }
    if user_identity is not None:
        structlog_bindings["user_id"] = user_identity
    structlog.contextvars.bind_contextvars(**structlog_bindings)


def make_run_trace_context(
    run_id: str,
    thread_id: str,
    graph_id: str,
    user_identity: str | None,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> contextvars.Context:
    """Return an isolated context copy with trace context pre-set.

    Pass to ``asyncio.create_task(..., context=ctx)`` so the background task's
    spans and logs carry the run identifiers — mirrors the worker path's
    ``_restore_trace_context``. Delegates the binding to
    :func:`bind_run_trace_context`.
    """
    # ``ctx.run`` so the binding lands in the returned context, not the caller's.
    ctx = contextvars.copy_context()
    ctx.run(
        bind_run_trace_context,
        run_id=run_id,
        thread_id=thread_id,
        graph_id=graph_id,
        user_identity=user_identity,
        extra_metadata=extra_metadata,
    )
    return ctx
