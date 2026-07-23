import logging

from aegra_api.observability.otel import otel_provider

logger = logging.getLogger(__name__)


def setup_observability() -> None:
    """Initialize the OpenTelemetry subsystem when any target is configured."""
    if not otel_provider.is_enabled():
        logger.info("Observability is disabled (no targets configured).")
        return
    try:
        otel_provider.setup()
    except Exception as e:
        logger.error(f"Failed to initialize observability: {e}")
        return
    logger.info("Observability subsystem initialized successfully.")
