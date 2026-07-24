"""Unit tests for the OpenTelemetry provider."""

from unittest.mock import MagicMock, patch

import pytest
from openinference.instrumentation import TraceConfig
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    ParentBasedTraceIdRatio,
    TraceIdRatioBased,
)

from aegra_api.observability.otel import OpenTelemetryProvider
from aegra_api.observability.span_enrichment import RunIdGenerator
from aegra_api.observability.targets import (
    BaseOtelTarget,
    GenericOtelTarget,
    LangfuseTarget,
    PhoenixTarget,
)


class TestOpenTelemetryProviderInit:
    """Tests for initialization and target resolution logic."""

    def test_init_disabled_by_default(self):
        """Test that provider is disabled when no targets are configured."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            assert provider.is_enabled() is False
            assert len(provider._active_targets) == 0

    def test_init_parses_targets_correctly(self):
        """Test that known targets are parsed and instantiated."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "LANGFUSE, PHOENIX, GENERIC"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            assert provider.is_enabled() is True
            assert len(provider._active_targets) == 3

            target_types = {type(t) for t in provider._active_targets}
            assert LangfuseTarget in target_types
            assert PhoenixTarget in target_types
            assert GenericOtelTarget in target_types

    def test_init_handles_whitespace_and_casing(self):
        """Test that parsing is robust to whitespace and case sensitivity."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "  langfuse ,  phoenix "
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            assert len(provider._active_targets) == 2
            target_types = {type(t) for t in provider._active_targets}
            assert LangfuseTarget in target_types
            assert PhoenixTarget in target_types

    def test_init_ignores_unknown_targets(self):
        """Test that unknown targets are logged and ignored."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "LANGFUSE, UNKNOWN_VENDOR"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            with patch("aegra_api.observability.otel.logger") as mock_logger:
                provider = OpenTelemetryProvider()

                assert len(provider._active_targets) == 1
                assert isinstance(provider._active_targets[0], LangfuseTarget)

                # Verify warning was logged
                mock_logger.warning.assert_called_with("Unknown OTEL target in settings: UNKNOWN_VENDOR")

    def test_init_enables_console_export(self):
        """Test that console export enables the provider even without targets."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()

            assert provider.is_enabled() is True

    def test_add_custom_target(self):
        """Test dynamically adding a custom target."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            # FIX: Explicitly set False so truthy Mock doesn't trigger it
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()
            assert provider.is_enabled() is False

            mock_target = MagicMock(spec=BaseOtelTarget)
            provider.add_custom_target(mock_target)

            assert provider.is_enabled() is True
            assert mock_target in provider._active_targets

    def test_add_custom_target_attaches_exporter_after_setup(self) -> None:
        """Test that adding a target after setup() wires its exporter immediately."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()
            provider.setup()

            mock_target = MagicMock(spec=BaseOtelTarget)
            mock_target.name = "Late"
            mock_exporter = MagicMock()
            mock_target.get_exporter.return_value = mock_exporter

            provider.add_custom_target(mock_target)

            mock_target.get_exporter.assert_called_once()
            assert provider._tracer_provider is not None


class TestOpenTelemetryProviderSetup:
    """Tests for the setup() method and tracer configuration."""

    @pytest.fixture
    def mock_deps(self):
        """Patch all external OTEL dependencies."""
        with (
            patch("aegra_api.observability.otel.TracerProvider") as mock_tp,
            patch("aegra_api.observability.otel.BatchSpanProcessor") as mock_bsp,
            patch("aegra_api.observability.otel.ConsoleSpanExporter") as mock_cse,
            patch("aegra_api.observability.otel.LangChainInstrumentor") as mock_lci,
            patch("aegra_api.observability.otel.HTTPXClientInstrumentor") as mock_hci,
            patch("aegra_api.observability.otel.trace") as mock_trace,
            patch("aegra_api.observability.otel.Resource") as mock_resource,
            patch("aegra_api.observability.otel.settings") as mock_settings,
        ):
            # Setup defaults
            mock_settings.observability.OTEL_SERVICE_NAME = "test-service"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False  # Default to False to prevent noise
            mock_settings.observability.OTEL_HIDE_LLM_INPUTS = False
            mock_settings.observability.OTEL_HIDE_LLM_OUTPUTS = False
            mock_settings.observability.OTEL_TRACES_SAMPLER = ""
            mock_settings.observability.OTEL_TRACES_SAMPLER_ARG = 1.0
            mock_settings.app.VERSION = "1.0.0"
            mock_settings.app.ENV_MODE = "TEST"

            yield {
                "tp": mock_tp,
                "bsp": mock_bsp,
                "cse": mock_cse,
                "lci": mock_lci,
                "hci": mock_hci,
                "trace": mock_trace,
                "resource": mock_resource,
                "settings": mock_settings,
            }

    def test_setup_is_idempotent(self, mock_deps):
        """Test that setup runs only once."""
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True

        provider = OpenTelemetryProvider()

        # First call
        provider.setup()
        assert mock_deps["tp"].called

        # Reset mocks
        mock_deps["tp"].reset_mock()

        # Second call
        provider.setup()
        assert not mock_deps["tp"].called  # Should not be called again

    def test_setup_creates_correct_resource(self, mock_deps):
        """Test that Resource is created with correct attributes."""
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True

        provider = OpenTelemetryProvider()
        provider.setup()

        mock_deps["resource"].create.assert_called_with(
            attributes={
                "service.name": "test-service",
                "service.version": "1.0.0",
                "deployment.environment": "test",
            }
        )
        _, tp_kwargs = mock_deps["tp"].call_args
        assert tp_kwargs["resource"] is mock_deps["resource"].create.return_value
        assert isinstance(tp_kwargs["id_generator"], RunIdGenerator)

    def test_setup_attaches_configured_targets(self, mock_deps):
        """Test that exporters from targets are attached to the tracer."""
        # Setup a mock target that returns an exporter
        mock_exporter = MagicMock()
        mock_target = MagicMock(spec=BaseOtelTarget)
        mock_target.get_exporter.return_value = mock_exporter
        mock_target.name = "MockTarget"

        provider = OpenTelemetryProvider()
        # Manually inject target
        provider._active_targets = [mock_target]
        provider._enabled = True

        provider.setup()

        mock_target.get_exporter.assert_called_once()
        mock_deps["bsp"].assert_any_call(mock_exporter)
        tracer_provider_instance = mock_deps["tp"].return_value
        tracer_provider_instance.add_span_processor.assert_called()

    def test_setup_handles_target_errors_gracefully(self, mock_deps):
        """Test that setup continues even if one target fails."""
        # FIX: Ensure console export is disabled for this test to avoid confusion
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = False

        # Target 1 throws exception
        bad_target = MagicMock(spec=BaseOtelTarget)
        bad_target.get_exporter.side_effect = Exception("Config Error")
        bad_target.name = "BadTarget"

        # Target 2 works
        good_exporter = MagicMock()
        good_target = MagicMock(spec=BaseOtelTarget)
        good_target.get_exporter.return_value = good_exporter
        good_target.name = "GoodTarget"

        provider = OpenTelemetryProvider()
        provider._active_targets = [bad_target, good_target]
        provider._enabled = True

        with patch("aegra_api.observability.otel.logger") as mock_logger:
            provider.setup()

            # Should log error for bad target
            mock_logger.error.assert_called()

            # Should still add processor for good target
            tracer_provider_instance = mock_deps["tp"].return_value
            mock_deps["bsp"].assert_called_with(good_exporter)
            # SpanEnrichmentProcessor is added unconditionally + one BatchSpanProcessor
            # for the good target → two calls total
            assert tracer_provider_instance.add_span_processor.call_count == 2

    def test_setup_instruments_globally(self, mock_deps):
        """Test that global tracer and instrumentation are set."""
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True

        provider = OpenTelemetryProvider()
        provider.setup()

        mock_deps["trace"].set_tracer_provider.assert_called_with(mock_deps["tp"].return_value)

        _, lci_kwargs = mock_deps["lci"].return_value.instrument.call_args
        assert lci_kwargs["tracer_provider"] is mock_deps["tp"].return_value
        assert isinstance(lci_kwargs["config"], TraceConfig)
        mock_deps["hci"].return_value.instrument.assert_called_with(tracer_provider=mock_deps["tp"].return_value)

    def test_setup_passes_resolved_sampler_to_tracer_provider(self, mock_deps):
        """A configured ratio sampler is constructed and handed to TracerProvider."""
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True
        mock_deps["settings"].observability.OTEL_TRACES_SAMPLER = "traceidratio"
        mock_deps["settings"].observability.OTEL_TRACES_SAMPLER_ARG = 0.25

        provider = OpenTelemetryProvider()
        provider.setup()

        _, tp_kwargs = mock_deps["tp"].call_args
        sampler = tp_kwargs["sampler"]
        assert isinstance(sampler, TraceIdRatioBased)
        assert sampler.rate == 0.25

    def test_setup_defaults_to_none_sampler(self, mock_deps):
        """No sampler configured → None passed, preserving the SDK default."""
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True
        mock_deps["settings"].observability.OTEL_TRACES_SAMPLER = ""

        provider = OpenTelemetryProvider()
        provider.setup()

        _, tp_kwargs = mock_deps["tp"].call_args
        assert tp_kwargs["sampler"] is None

    def test_setup_passes_redaction_config_to_instrumentor(self, mock_deps, monkeypatch):
        """Redaction toggles reach LangChainInstrumentor as TraceConfig hide flags."""
        monkeypatch.delenv("OPENINFERENCE_HIDE_INPUTS", raising=False)
        monkeypatch.delenv("OPENINFERENCE_HIDE_OUTPUTS", raising=False)
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True
        mock_deps["settings"].observability.OTEL_HIDE_LLM_INPUTS = True
        mock_deps["settings"].observability.OTEL_HIDE_LLM_OUTPUTS = False

        provider = OpenTelemetryProvider()
        provider.setup()

        _, lci_kwargs = mock_deps["lci"].return_value.instrument.call_args
        assert lci_kwargs["config"].hide_inputs is True
        assert lci_kwargs["config"].hide_outputs is False


class TestOpenTelemetryProviderRuntime:
    """Tests for runtime methods (get_metadata)."""

    def test_get_metadata_returns_correct_structure(self):
        """Test metadata generation when enabled."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata(run_id="run-123", thread_id="thread-456", user_identity="user-789")

            assert meta == {
                "run_id": "run-123",
                "thread_id": "thread-456",
                "session_id": "thread-456",
                "user_id": "user-789",
            }

    def test_get_metadata_includes_langfuse_keys_when_langfuse_active(self) -> None:
        """Test that langfuse_* keys are emitted only when Langfuse target is active."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "LANGFUSE"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata(run_id="run-123", thread_id="thread-456", user_identity="user-789")

            assert meta["langfuse_session_id"] == "thread-456"
            assert meta["langfuse_user_id"] == "user-789"

    def test_get_metadata_langfuse_without_user_identity(self) -> None:
        """Test that langfuse_session_id is present but langfuse_user_id absent when user_identity is None."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "LANGFUSE"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata(run_id="run-123", thread_id="thread-456")

            assert meta["langfuse_session_id"] == "thread-456"
            assert "langfuse_user_id" not in meta

    def test_get_metadata_excludes_langfuse_keys_when_langfuse_inactive(self) -> None:
        """Test that langfuse_* keys are NOT emitted when only non-Langfuse targets are active."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata(run_id="run-123", thread_id="thread-456", user_identity="user-789")

            assert "langfuse_session_id" not in meta
            assert "langfuse_user_id" not in meta

    def test_get_metadata_empty_when_disabled(self):
        """Test metadata returns empty dict when disabled."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata("run-1", "thread-1")
            assert meta == {}


class TestResolveSampler:
    """Tests for OTEL_TRACES_SAMPLER → Sampler resolution."""

    @pytest.fixture
    def mock_settings(self):
        """Patch settings with tracing off so the provider constructs cleanly."""
        with patch("aegra_api.observability.otel.settings") as mock:
            mock.observability.OTEL_TARGETS = ""
            mock.observability.OTEL_CONSOLE_EXPORT = False
            mock.observability.OTEL_TRACES_SAMPLER = ""
            mock.observability.OTEL_TRACES_SAMPLER_ARG = 1.0
            yield mock

    def test_empty_keeps_sdk_default(self, mock_settings):
        """Empty sampler name → None so TracerProvider keeps its own default."""
        provider = OpenTelemetryProvider()
        assert provider._resolve_sampler() is None

    def test_always_on(self, mock_settings):
        mock_settings.observability.OTEL_TRACES_SAMPLER = "always_on"
        provider = OpenTelemetryProvider()
        assert provider._resolve_sampler() is ALWAYS_ON

    def test_always_off(self, mock_settings):
        mock_settings.observability.OTEL_TRACES_SAMPLER = "always_off"
        provider = OpenTelemetryProvider()
        assert provider._resolve_sampler() is ALWAYS_OFF

    def test_parentbased_always_on(self, mock_settings):
        mock_settings.observability.OTEL_TRACES_SAMPLER = "parentbased_always_on"
        provider = OpenTelemetryProvider()
        sampler = provider._resolve_sampler()
        assert isinstance(sampler, ParentBased)
        assert not isinstance(sampler, ParentBasedTraceIdRatio)

    def test_traceidratio(self, mock_settings):
        mock_settings.observability.OTEL_TRACES_SAMPLER = "traceidratio"
        mock_settings.observability.OTEL_TRACES_SAMPLER_ARG = 0.1
        provider = OpenTelemetryProvider()
        sampler = provider._resolve_sampler()
        assert isinstance(sampler, TraceIdRatioBased)
        assert sampler.rate == 0.1

    def test_parentbased_traceidratio(self, mock_settings):
        mock_settings.observability.OTEL_TRACES_SAMPLER = "parentbased_traceidratio"
        mock_settings.observability.OTEL_TRACES_SAMPLER_ARG = 0.5
        provider = OpenTelemetryProvider()
        sampler = provider._resolve_sampler()
        assert isinstance(sampler, ParentBasedTraceIdRatio)
        assert "TraceIdRatioBased{0.5}" in sampler.get_description()

    def test_whitespace_and_casing_normalized(self, mock_settings):
        """Parsing is robust to surrounding whitespace and case, like OTEL_TARGETS."""
        mock_settings.observability.OTEL_TRACES_SAMPLER = "  TraceIdRatio  "
        mock_settings.observability.OTEL_TRACES_SAMPLER_ARG = 0.3
        provider = OpenTelemetryProvider()
        sampler = provider._resolve_sampler()
        assert isinstance(sampler, TraceIdRatioBased)
        assert sampler.rate == 0.3

    def test_unknown_name_falls_back_to_default(self, mock_settings):
        mock_settings.observability.OTEL_TRACES_SAMPLER = "bogus"
        provider = OpenTelemetryProvider()
        with patch("aegra_api.observability.otel.logger") as mock_logger:
            assert provider._resolve_sampler() is None
            mock_logger.warning.assert_called_once()

    def test_ratio_out_of_range_falls_back_to_default(self, mock_settings):
        mock_settings.observability.OTEL_TRACES_SAMPLER = "traceidratio"
        mock_settings.observability.OTEL_TRACES_SAMPLER_ARG = 1.5
        provider = OpenTelemetryProvider()
        with patch("aegra_api.observability.otel.logger") as mock_logger:
            assert provider._resolve_sampler() is None
            mock_logger.warning.assert_called_once()


class TestBuildTraceConfig:
    """Tests for OTEL_HIDE_LLM_* → OpenInference TraceConfig mapping."""

    @pytest.fixture
    def mock_settings(self, monkeypatch):
        """Patch settings and clear native env vars for deterministic resolution."""
        monkeypatch.delenv("OPENINFERENCE_HIDE_INPUTS", raising=False)
        monkeypatch.delenv("OPENINFERENCE_HIDE_OUTPUTS", raising=False)
        with patch("aegra_api.observability.otel.settings") as mock:
            mock.observability.OTEL_TARGETS = ""
            mock.observability.OTEL_CONSOLE_EXPORT = False
            mock.observability.OTEL_HIDE_LLM_INPUTS = False
            mock.observability.OTEL_HIDE_LLM_OUTPUTS = False
            yield mock

    def test_both_off_resolves_to_false(self, mock_settings):
        """Toggles off with no env vars → flags resolve to False (status quo)."""
        provider = OpenTelemetryProvider()
        config = provider._build_trace_config()
        assert config.hide_inputs is False
        assert config.hide_outputs is False

    def test_inputs_on(self, mock_settings):
        mock_settings.observability.OTEL_HIDE_LLM_INPUTS = True
        provider = OpenTelemetryProvider()
        config = provider._build_trace_config()
        assert config.hide_inputs is True
        assert config.hide_outputs is False

    def test_outputs_on(self, mock_settings):
        mock_settings.observability.OTEL_HIDE_LLM_OUTPUTS = True
        provider = OpenTelemetryProvider()
        config = provider._build_trace_config()
        assert config.hide_inputs is False
        assert config.hide_outputs is True

    def test_off_defers_to_native_env_var(self, mock_settings, monkeypatch):
        """Toggle off must not force redaction off: a native env var still wins."""
        monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "true")
        provider = OpenTelemetryProvider()
        config = provider._build_trace_config()
        assert config.hide_inputs is True
