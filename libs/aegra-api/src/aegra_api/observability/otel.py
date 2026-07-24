"""
Unified OpenTelemetry Provider.
Orchestrates trace generation and fan-out export to multiple targets.
"""

import logging
from typing import Any

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    ParentBasedTraceIdRatio,
    Sampler,
    TraceIdRatioBased,
)

from aegra_api.observability.span_enrichment import RunIdGenerator, SpanEnrichmentProcessor
from aegra_api.observability.targets import (
    BaseOtelTarget,
    GenericOtelTarget,
    LangfuseTarget,
    PhoenixTarget,
)
from aegra_api.settings import settings

logger = logging.getLogger(__name__)


class OpenTelemetryProvider:
    """Configures the global OpenTelemetry tracer and fans out to targets."""

    def __init__(self) -> None:
        self._enabled = False
        self._tracer_provider: TracerProvider | None = None

        self._active_targets: list[BaseOtelTarget] = self._resolve_targets()
        self._has_langfuse = any(isinstance(target, LangfuseTarget) for target in self._active_targets)

        if self._active_targets or settings.observability.OTEL_CONSOLE_EXPORT:
            self._enabled = True

    def is_enabled(self) -> bool:
        return self._enabled

    def _resolve_targets(self) -> list[BaseOtelTarget]:
        targets: list[BaseOtelTarget] = []
        raw_targets = settings.observability.OTEL_TARGETS

        if not raw_targets:
            return targets

        for name in raw_targets.split(","):
            name_clean = name.strip().upper()
            if not name_clean:
                continue

            if name_clean == "LANGFUSE":
                targets.append(LangfuseTarget())
            elif name_clean == "PHOENIX":
                targets.append(PhoenixTarget())
            elif name_clean in ("GENERIC", "DEFAULT", "OTLP"):
                targets.append(GenericOtelTarget())
            else:
                logger.warning(f"Unknown OTEL target in settings: {name_clean}")

        return targets

    def _resolve_sampler(self) -> Sampler | None:
        """Map ``OTEL_TRACES_SAMPLER`` to a Sampler; None keeps the SDK default.

        Ratio sampling is consistent per run: the trace id is derived from
        run_id (``span_enrichment.trace_id_from_run``), so every span in a run
        shares one trace id and one decision — a run's trace is never split.
        """
        name = settings.observability.OTEL_TRACES_SAMPLER.strip().lower()
        if not name:
            return None

        if name == "always_on":
            return ALWAYS_ON
        if name == "always_off":
            return ALWAYS_OFF
        if name == "parentbased_always_on":
            return ParentBased(ALWAYS_ON)
        if name == "parentbased_always_off":
            return ParentBased(ALWAYS_OFF)

        if name in ("traceidratio", "parentbased_traceidratio"):
            ratio = settings.observability.OTEL_TRACES_SAMPLER_ARG
            if not 0.0 <= ratio <= 1.0:
                logger.warning(f"OTEL_TRACES_SAMPLER_ARG={ratio} outside [0.0, 1.0]; keeping default sampler")
                return None
            if name == "traceidratio":
                return TraceIdRatioBased(ratio)
            return ParentBasedTraceIdRatio(ratio)

        logger.warning(f"Unknown OTEL_TRACES_SAMPLER in settings: {name}; keeping default sampler")
        return None

    def _build_trace_config(self) -> TraceConfig:
        """Build the OpenInference TraceConfig from the LLM redaction toggles.

        A disabled toggle stays None (not False) so OpenInference still honors
        its native OPENINFERENCE_HIDE_* env vars — off must not silently undo a
        pre-existing redaction.
        """
        obs = settings.observability
        return TraceConfig(
            hide_inputs=True if obs.OTEL_HIDE_LLM_INPUTS else None,
            hide_outputs=True if obs.OTEL_HIDE_LLM_OUTPUTS else None,
        )

    def add_custom_target(self, target: BaseOtelTarget) -> None:
        """Allow registering custom targets dynamically."""
        self._active_targets.append(target)
        if isinstance(target, LangfuseTarget):
            self._has_langfuse = True
        self._enabled = True

        if self._tracer_provider is not None:
            try:
                exporter = target.get_exporter()
                if exporter:
                    self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                    logger.info(f"Observability: Attached target '{target.name}'")
            except Exception as e:
                logger.error(f"Observability: Failed to attach target '{target.name}': {e}")

    def setup(self) -> None:
        """Initializes the Global Tracer Provider. Runs once."""
        if self._tracer_provider:
            return

        resource = Resource.create(
            attributes={
                "service.name": settings.observability.OTEL_SERVICE_NAME,
                "service.version": settings.app.VERSION,
                "deployment.environment": settings.app.ENV_MODE.lower(),
            }
        )

        self._tracer_provider = TracerProvider(
            resource=resource, id_generator=RunIdGenerator(), sampler=self._resolve_sampler()
        )
        self._tracer_provider.add_span_processor(SpanEnrichmentProcessor())
        processors_count = 0

        for target in self._active_targets:
            try:
                exporter = target.get_exporter()
                if exporter:
                    processor = BatchSpanProcessor(exporter)
                    self._tracer_provider.add_span_processor(processor)
                    processors_count += 1
                    logger.info(f"Observability: Attached target '{target.name}'")
            except Exception as e:
                logger.error(f"Observability: Failed to attach target '{target.name}': {e}")

        if settings.observability.OTEL_CONSOLE_EXPORT:
            self._tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            processors_count += 1
            logger.info("Observability: Console export enabled")

        if processors_count > 0:
            trace.set_tracer_provider(self._tracer_provider)
            LangChainInstrumentor().instrument(
                tracer_provider=self._tracer_provider, config=self._build_trace_config()
            )
            HTTPXClientInstrumentor().instrument(tracer_provider=self._tracer_provider)
            logger.info("Observability: Auto-instrumentation enabled (LangChain + HTTPX)")

    def get_metadata(self, run_id: str, thread_id: str, user_identity: str | None = None) -> dict[str, Any]:
        if not self.is_enabled():
            return {}

        meta: dict[str, Any] = {
            "run_id": run_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        }
        if user_identity:
            meta["user_id"] = user_identity

        if self._has_langfuse:
            # Langfuse CallbackHandler only promotes langfuse_* prefixed keys
            # to trace-level fields; plain session_id stays in generic metadata.
            meta["langfuse_session_id"] = thread_id
            if user_identity:
                meta["langfuse_user_id"] = user_identity

        return meta


otel_provider = OpenTelemetryProvider()
