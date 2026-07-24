"""Unit tests for run_status service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.run_status import (
    _safe_serialize,
    finalize_run,
    set_thread_status,
    update_run_status,
)


def _make_mock_session(*, rowcount: int = 1) -> AsyncMock:
    """Create a mock async session with execute and commit.

    ``execute`` returns a result whose ``rowcount`` drives finalize_run's
    compare-and-set (default 1 = this call won the terminal transition).
    """
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=rowcount))
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
    """finalize_run materializes thread_state from the checkpointer snapshot
    (``state_values``), never from the run's stream ``output``."""

    @pytest.mark.asyncio
    async def test_no_materialize_when_state_values_absent(self) -> None:
        # Cancel/error/timeout paths pass no state_values; materializing an empty
        # {} would wipe the thread's real materialized values/interrupts.
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            await finalize_run("run-1", "thread-1", status="error", thread_status="error", error="boom")

        mat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_output_alone_does_not_materialize(self) -> None:
        # run.output persists, but output must NOT drive materialization — only
        # the checkpointer snapshot (state_values) does.
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            await finalize_run("run-1", "thread-1", status="success", thread_status="idle", output={"streamed": 1})

        mat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_materializes_from_state_values_not_output(self) -> None:
        # Regression (P0-3): the materialization source is the checkpointer
        # snapshot, distinct from the (possibly empty) stream output.
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            await finalize_run(
                "run-1",
                "thread-1",
                status="success",
                thread_status="idle",
                output={"streamed": "ignored"},
                state_values={"real": 1},
            )

        mat.assert_awaited_once_with(session, "thread-1", {"real": 1}, {})

    @pytest.mark.asyncio
    async def test_empty_stream_output_does_not_wipe_checkpointer_state(self) -> None:
        # Regression (P0-3): a non-'values' stream mode yields output={}, but the
        # checkpointer still holds the real state — materialize THAT, not {}.
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            await finalize_run(
                "run-1",
                "thread-1",
                status="success",
                thread_status="idle",
                output={},
                state_values={"messages": [{"role": "ai", "content": "hi"}]},
            )

        mat.assert_awaited_once_with(session, "thread-1", {"messages": [{"role": "ai", "content": "hi"}]}, {})

    @pytest.mark.asyncio
    async def test_materializes_state_values_with_interrupts(self) -> None:
        session = _make_mock_session()
        maker = _make_mock_session_maker(session)
        interrupts = {"t1": [{"value": "x", "id": "i1"}]}
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            await finalize_run(
                "run-1",
                "thread-1",
                status="interrupted",
                thread_status="interrupted",
                state_values={"v": 1},
                interrupts=interrupts,
            )

        mat.assert_awaited_once_with(session, "thread-1", {"v": 1}, interrupts)


class TestFinalizeRunCompareAndSet:
    """P0-1: finalize is a compare-and-set on non-terminal status, so a second
    concurrent finalizer (lease-loss double-execution) can't double-write."""

    @pytest.mark.asyncio
    async def test_returns_true_and_writes_thread_when_won(self) -> None:
        session = _make_mock_session(rowcount=1)
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            won = await finalize_run("run-1", "thread-1", status="success", thread_status="idle", state_values={"v": 1})

        assert won is True
        # run UPDATE + thread UPDATE both executed.
        assert session.execute.await_count == 2
        mat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_and_skips_thread_when_already_terminal(self) -> None:
        # rowcount=0 → the guarded run UPDATE matched nothing (already terminal).
        session = _make_mock_session(rowcount=0)
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()) as mat,
        ):
            won = await finalize_run("run-1", "thread-1", status="success", thread_status="idle", state_values={"v": 1})

        assert won is False
        # Only the run UPDATE ran; the loser must NOT touch thread status or state.
        assert session.execute.await_count == 1
        mat.assert_not_awaited()


class TestFinalizeRunWebhookOutbox:
    """P1: a winning finalize enqueues the webhook delivery in the SAME transaction."""

    @pytest.mark.asyncio
    async def test_enqueues_delivery_row_when_won(self) -> None:
        session = _make_mock_session(rowcount=1)
        maker = _make_mock_session_maker(session)
        with (
            patch("aegra_api.services.run_status._get_session_maker", return_value=maker),
            patch("aegra_api.services.run_status.materialize_thread_state", new=AsyncMock()),
        ):
            won = await finalize_run(
                "run-1", "thread-1", status="success", thread_status="idle", webhook="https://hook.example/x"
            )

        assert won is True
        # run UPDATE + thread UPDATE + outbox INSERT = 3 statements in one tx.
        assert session.execute.await_count == 3

    @pytest.mark.asyncio
    async def test_no_delivery_row_when_lost(self) -> None:
        session = _make_mock_session(rowcount=0)
        maker = _make_mock_session_maker(session)
        with patch("aegra_api.services.run_status._get_session_maker", return_value=maker):
            won = await finalize_run(
                "run-1", "thread-1", status="success", thread_status="idle", webhook="https://hook.example/x"
            )

        assert won is False
        # Loser writes nothing beyond the guarded run UPDATE — no orphan delivery.
        assert session.execute.await_count == 1
