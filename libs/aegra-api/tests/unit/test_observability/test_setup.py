"""Unit tests for the observability setup module."""

from unittest.mock import patch

import pytest

from aegra_api.observability.setup import setup_observability


class TestSetupObservability:
    """Tests for the setup_observability function."""

    @pytest.fixture
    def mock_deps(self):
        """Patch the OTEL provider and logger used in setup.py."""
        with (
            patch("aegra_api.observability.setup.otel_provider") as mock_otel_provider,
            patch("aegra_api.observability.setup.logger") as mock_logger,
        ):
            yield {"provider": mock_otel_provider, "logger": mock_logger}

    def test_setup_initializes_when_enabled(self, mock_deps):
        """setup() runs and success is logged when the provider is enabled."""
        mock_deps["provider"].is_enabled.return_value = True

        setup_observability()

        mock_deps["provider"].setup.assert_called_once()
        mock_deps["logger"].info.assert_called_with("Observability subsystem initialized successfully.")

    def test_setup_skips_initialization_when_disabled(self, mock_deps):
        """setup() is not called when the provider is disabled."""
        mock_deps["provider"].is_enabled.return_value = False

        setup_observability()

        mock_deps["provider"].setup.assert_not_called()
        mock_deps["logger"].info.assert_called_with("Observability is disabled (no targets configured).")

    def test_setup_handles_exceptions_gracefully(self, mock_deps):
        """Exceptions during setup are caught and logged, not raised."""
        mock_deps["provider"].is_enabled.return_value = True
        mock_deps["provider"].setup.side_effect = Exception("Connection failed")

        setup_observability()

        mock_deps["provider"].setup.assert_called_once()
        mock_deps["logger"].error.assert_called_once()
        assert "Failed to initialize observability" in mock_deps["logger"].error.call_args[0][0]
