"""Unit tests for the webhook outbox deliverer."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services import webhook_deliverer as wd
from aegra_api.services.webhook_deliverer import WebhookDeliverer

MODULE = "aegra_api.services.webhook_deliverer"


def _maker(session: AsyncMock) -> MagicMock:
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=ctx)


def _session() -> AsyncMock:
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


class TestFinish:
    @pytest.mark.asyncio
    async def test_marks_delivered_on_ok(self) -> None:
        session = _session()
        with patch(f"{MODULE}._get_session_maker", return_value=_maker(session)):
            await WebhookDeliverer()._finish("d1", ok=True, attempts=0, error=None)
        assert session.execute.await_args.args[0] is wd._MARK_DELIVERED_SQL
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reschedules_when_under_max(self) -> None:
        session = _session()
        with patch(f"{MODULE}._get_session_maker", return_value=_maker(session)):
            await WebhookDeliverer()._finish("d1", ok=False, attempts=0, error="boom")
        assert session.execute.await_args.args[0] is wd._RESCHEDULE_SQL
        assert session.execute.await_args.args[1]["attempts"] == 1

    @pytest.mark.asyncio
    async def test_dead_letters_when_exhausted(self) -> None:
        # WEBHOOK_MAX_ATTEMPTS defaults to 3, so attempts=2 → 3 exhausts.
        session = _session()
        with patch(f"{MODULE}._get_session_maker", return_value=_maker(session)):
            await WebhookDeliverer()._finish("d1", ok=False, attempts=2, error="boom")
        assert session.execute.await_args.args[0] is wd._MARK_DEAD_SQL
        assert session.execute.await_args.args[1]["attempts"] == 3


class TestDeliver:
    @pytest.mark.asyncio
    async def test_delivers_and_marks_ok(self) -> None:
        deliverer = WebhookDeliverer()
        row = SimpleNamespace(id="d1", run_id="r1", url="https://hook/x", attempts=0)
        with (
            patch(f"{MODULE}.deliver_webhook", new=AsyncMock(return_value=True)) as dw,
            patch.object(deliverer, "_build_payload", new=AsyncMock(return_value={"run_id": "r1"})),
            patch.object(deliverer, "_finish", new=AsyncMock()) as fin,
        ):
            await deliverer._deliver(row)
        dw.assert_awaited_once()
        assert fin.await_args.kwargs["ok"] is True

    @pytest.mark.asyncio
    async def test_missing_run_is_failure_without_posting(self) -> None:
        deliverer = WebhookDeliverer()
        row = SimpleNamespace(id="d1", run_id="gone", url="https://hook/x", attempts=0)
        with (
            patch(f"{MODULE}.deliver_webhook", new=AsyncMock()) as dw,
            patch.object(deliverer, "_build_payload", new=AsyncMock(return_value=None)),
            patch.object(deliverer, "_finish", new=AsyncMock()) as fin,
        ):
            await deliverer._deliver(row)
        dw.assert_not_awaited()  # no POST when the run row is gone
        assert fin.await_args.kwargs["ok"] is False
