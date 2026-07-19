"""Unit tests for run_status service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.run_status import (
    _safe_serialize,
    finalize_run,
    set_thread_status,
    update_run_status,
)


def _make_mock_session() -> AsyncMock:
    """Create a mock async session with execute and commit."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


def _make_mock_session_maker(session: AsyncMock) -> MagicMock:
    """Wrap a mock session in a context-manager-returning maker."""
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    maker = MagicMock(return_value=ctx)
    return maker


class TestUpdateRunStatus:
    @pytest.mark.asyncio
    async def test_updates_db_with_status(self) -> None:
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)

        with patch("aegra_api.services.run_status._get_session_maker", return_value=maker):
            await update_run_status("run-1", "running")

        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_includes_output_when_provided(self) -> None:
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)

        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status._safe_serialize", return_value=({"key": "val"}, True)) as mock_ser,
        ):
            await update_run_status("run-1", "success", output={"key": "val"})

        mock_ser.assert_called_once_with({"key": "val"}, "run-1")
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_includes_error_when_provided(self) -> None:
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)

        with patch("aegra_api.services.run_status._get_session_maker", return_value=maker):
            await update_run_status("run-1", "error", error="something broke")

        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_omits_output_and_error_when_not_provided(self) -> None:
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)

        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status._safe_serialize") as mock_ser,
        ):
            await update_run_status("run-1", "running")

        mock_ser.assert_not_called()


class TestSetThreadStatus:
    @pytest.mark.asyncio
    async def test_updates_thread_status(self) -> None:
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute = AsyncMock(return_value=mock_result)

        await set_thread_status(session, "thread-1", "idle")

        session.execute.assert_awaited_once()
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_thread_not_found(self) -> None:
        session = _make_mock_session()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="Thread 'thread-missing' not found"):
            await set_thread_status(session, "thread-missing", "idle")


class TestSafeSerialize:
    def test_returns_serialized_output(self) -> None:
        with patch("aegra_api.services.run_status._serializer") as mock_ser:
            mock_ser.serialize.return_value = {"a": 1}
            result = _safe_serialize({"a": 1}, "run-1")

        assert result == ({"a": 1}, True)

    def test_returns_fallback_on_failure(self) -> None:
        with patch("aegra_api.services.run_status._serializer") as mock_ser:
            mock_ser.serialize.side_effect = TypeError("boom")
            value, ok = _safe_serialize(object(), "run-1")

        assert ok is False
        assert value["error"] == "Output serialization failed"
        assert "original_type" in value


class TestFinalizeRunMaterialization:
    """finalize_run materializes thread_state only for genuinely serialized state."""

    @pytest.mark.asyncio
    async def test_no_materialize_when_output_absent(self) -> None:
        # Cancel/error paths pass no output; materializing an empty {} would wipe
        # the thread's real materialized values/interrupts.
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            await finalize_run("run-1", "thread-1", status="error", thread_status="error", error="boom")

        mat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_materialize_on_serialization_failure(self) -> None:
        # The _safe_serialize failure fallback must not be materialized as state.
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status._safe_serialize", return_value=({"error": "x"}, False)),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            await finalize_run("run-1", "thread-1", status="success", thread_status="idle", output={"v": 1})

        mat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_materializes_genuine_state_with_interrupts(self) -> None:
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)
        interrupts = {"t1": [{"value": "x", "id": "i1"}]}
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status._safe_serialize", return_value=({"v": 1}, True)),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            await finalize_run(
                "run-1",
                "thread-1",
                status="interrupted",
                thread_status="interrupted",
                output={"v": 1},
                interrupts=interrupts,
            )

        mat.assert_awaited_once_with(session, "thread-1", {"v": 1}, interrupts)
