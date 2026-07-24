"""Unit tests for RunTTLSweeper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.run_ttl_sweeper import RunTTLSweeper

MODULE = "aegra_api.services.run_ttl_sweeper"


def _session_deleting(run_ids: list[str]) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = [(rid,) for rid in run_ids]
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    return session


def _maker(session: AsyncMock) -> MagicMock:
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


class TestRunTTLSweeperTick:
    @pytest.mark.asyncio
    async def test_issues_delete_with_ttl_and_batch_params(self) -> None:
        session = _session_deleting(["r1", "r2"])
        with patch(f"{MODULE}._get_session_maker", return_value=_maker(session)):
            await RunTTLSweeper()._tick()

        session.execute.assert_awaited_once()
        params = session.execute.await_args.args[1]
        assert "ttl" in params and "batch" in params
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_commits_even_when_nothing_pruned(self) -> None:
        session = _session_deleting([])
        with patch(f"{MODULE}._get_session_maker", return_value=_maker(session)):
            await RunTTLSweeper()._tick()

        # The atomic DELETE...RETURNING runs regardless; empty just returns no ids.
        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()


class TestRunTTLSweeperLifecycle:
    @pytest.mark.asyncio
    async def test_start_is_noop_when_disabled(self) -> None:
        sweeper = RunTTLSweeper()
        with patch(f"{MODULE}.settings") as mock_settings:
            mock_settings.run_ttl.RUN_TTL_ENABLED = False
            await sweeper.start()
        assert sweeper._task is None
