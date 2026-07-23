"""Tests for application lifespan and startup logic"""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_sets_up_observability():
    """Test that the lifespan function initializes observability during startup."""
    import aegra_api.main as main_module

    importlib.reload(main_module)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock),
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability") as mock_setup_observability,
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()

        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service

        mock_app = MagicMock()

        async with main_module.lifespan(mock_app):
            pass

        mock_setup_observability.assert_called_once()
        mock_db_manager.close.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_calls_required_initialization():
    """Test that lifespan calls all required initialization functions."""
    # Reload main to ensure clean state
    import aegra_api.main as main_module

    importlib.reload(main_module)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock) as mock_migrations,
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability") as mock_setup_observability,
    ):
        # Setup mocks
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()

        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service

        mock_app = MagicMock()

        # Run the lifespan function
        async with main_module.lifespan(mock_app):
            pass

        # Verify migrations run first, then initialization
        mock_migrations.assert_called_once()
        mock_db_manager.initialize.assert_called_once()
        mock_langgraph_service.initialize.assert_called_once()

        # Verify observability setup was called
        mock_setup_observability.assert_called_once()

        # Verify cleanup
        mock_db_manager.close.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_skips_migrations_when_disabled(monkeypatch):
    """When RUN_MIGRATIONS_ON_STARTUP=false, lifespan must not call alembic."""
    import aegra_api.main as main_module
    from aegra_api.settings import settings

    importlib.reload(main_module)

    monkeypatch.setattr(settings.app, "RUN_MIGRATIONS_ON_STARTUP", False)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock) as mock_migrations,
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability"),
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()

        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service

        async with main_module.lifespan(MagicMock()):
            pass

        mock_migrations.assert_not_called()
        # The rest of startup must still happen.
        mock_db_manager.initialize.assert_called_once()
        mock_langgraph_service.initialize.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_runs_migrations_when_enabled(monkeypatch):
    """Default (RUN_MIGRATIONS_ON_STARTUP=true) keeps the auto-migrate behavior."""
    import aegra_api.main as main_module
    from aegra_api.settings import settings

    importlib.reload(main_module)

    monkeypatch.setattr(settings.app, "RUN_MIGRATIONS_ON_STARTUP", True)

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock) as mock_migrations,
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.setup_observability"),
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()

        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service

        async with main_module.lifespan(MagicMock()):
            pass

        mock_migrations.assert_called_once()


@pytest.mark.unit
def test_instrument_fastapi_skips_when_observability_disabled():
    """No HTTP server-span instrumentation when tracing is off (avoids overhead)."""
    import aegra_api.main as main_module

    with (
        patch.object(main_module.otel_provider, "is_enabled", return_value=False),
        patch.object(main_module, "FastAPIInstrumentor") as mock_instrumentor,
    ):
        main_module._instrument_fastapi(MagicMock())

    mock_instrumentor.instrument_app.assert_not_called()


@pytest.mark.unit
def test_instrument_fastapi_wraps_app_when_observability_enabled():
    """The app is instrumented for HTTP server spans when tracing is on."""
    import aegra_api.main as main_module

    app = MagicMock()
    with (
        patch.object(main_module.otel_provider, "is_enabled", return_value=True),
        patch.object(main_module, "FastAPIInstrumentor") as mock_instrumentor,
    ):
        main_module._instrument_fastapi(app)

    mock_instrumentor.instrument_app.assert_called_once()
    assert mock_instrumentor.instrument_app.call_args.args[0] is app
