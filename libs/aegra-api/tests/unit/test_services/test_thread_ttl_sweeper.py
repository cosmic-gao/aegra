"""Unit tests for ThreadTTLSweeper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.thread_ttl_sweeper import ThreadTTLSweeper

MODULE = "aegra_api.services.thread_ttl_sweeper"


def _session_with_stale(stale: list[str]) -> AsyncMock:
    """Mock session whose locked SELECT returns *stale* thread ids."""
    session = AsyncMock()
    scalars_result = MagicMock()
    scalars_result.all.return_value = stale
    session.scalars = AsyncMock(return_value=scalars_result)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


def _maker_for(session: AsyncMock) -> MagicMock:
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


class TestThreadTTLSweeperTick:
    @pytest.mark.asyncio
    async def test_deletes_checkpoints_and_thread_rows_for_stale(self) -> None:
        session = _session_with_stale(["t1", "t2"])
        checkpointer = AsyncMock()
        with (
            patch(f"{MODULE}._get_session_maker", return_value=_maker_for(session)),
            patch(f"{MODULE}.db_manager.get_checkpointer", return_value=checkpointer),
        ):
            await ThreadTTLSweeper()._tick()

        # Checkpoints deleted per stale thread, then a single cascading row delete.
        assert checkpointer.adelete_thread.await_count == 2
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_when_nothing_stale(self) -> None:
        session = _session_with_stale([])
        checkpointer = AsyncMock()
        with (
            patch(f"{MODULE}._get_session_maker", return_value=_maker_for(session)),
            patch(f"{MODULE}.db_manager.get_checkpointer", return_value=checkpointer),
        ):
            await ThreadTTLSweeper()._tick()

        checkpointer.adelete_thread.assert_not_awaited()
        session.execute.assert_not_awaited()
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_checkpoint_delete_failure_does_not_block_row_delete(self) -> None:
        session = _session_with_stale(["t1"])
        checkpointer = AsyncMock()
        checkpointer.adelete_thread = AsyncMock(side_effect=RuntimeError("saver down"))
        with (
            patch(f"{MODULE}._get_session_maker", return_value=_maker_for(session)),
            patch(f"{MODULE}.db_manager.get_checkpointer", return_value=checkpointer),
        ):
            await ThreadTTLSweeper()._tick()

        # A checkpointer error is logged, not raised — the row delete still runs.
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()
