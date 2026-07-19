"""Unit tests for thread_state materialization (gate + values-hash skip)."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aegra_api.services import run_status

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 7, 19, tzinfo=UTC)


class _Session:
    def __init__(self, current_hash: str | None):
        self._hash = current_hash
        self.executed: list = []

    async def scalar(self, _stmt):
        return self._hash

    async def execute(self, stmt):
        self.executed.append(stmt)


def test_values_hash_is_order_independent_and_distinct() -> None:
    assert run_status._values_hash({"a": 1, "b": 2}) == run_status._values_hash({"b": 2, "a": 1})
    assert run_status._values_hash({"a": 1}) != run_status._values_hash({"a": 2})


def test_update_columns_skip_values_when_unchanged() -> None:
    cols = run_status._thread_state_update_columns(
        unchanged=True, values={"x": 1}, interrupts={"t": []}, values_hash="h", now=_NOW
    )
    assert set(cols) == {"interrupts", "updated_at"}  # big `values` not rewritten


def test_update_columns_include_values_when_changed() -> None:
    cols = run_status._thread_state_update_columns(
        unchanged=False, values={"x": 1}, interrupts={}, values_hash="h", now=_NOW
    )
    assert set(cols) == {"values", "interrupts", "values_hash", "updated_at"}


async def test_gate_off_skips_write(monkeypatch) -> None:
    monkeypatch.setattr(run_status.settings.checkpointer, "THREAD_STATE_MATERIALIZE", False)
    session = _Session(None)
    await run_status.materialize_thread_state(session, "t1", {"x": 1}, {})
    assert session.executed == []


async def test_gate_on_upserts(monkeypatch) -> None:
    monkeypatch.setattr(run_status.settings.checkpointer, "THREAD_STATE_MATERIALIZE", True)
    session = _Session(None)
    await run_status.materialize_thread_state(session, "t1", {"x": 1}, {})
    assert len(session.executed) == 1


def test_migration_backfill_leaves_values_hash_null() -> None:
    # SQL md5(jsonb::text) can't match the app's md5(json.dumps(sort_keys=True)),
    # so the backfill leaves values_hash NULL for the first materialize to recompute.
    path = (
        Path(__file__).resolve().parents[3]
        / "alembic"
        / "versions"
        / "20260718140000_run_metadata_indexes_thread_state.py"
    )
    spec = importlib.util.spec_from_file_location("mig_e5b9d2f7a3c1", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    backfill_sql = str(mod._BACKFILL)
    assert "md5(t." not in backfill_sql  # the real values-hash call is gone
    assert "NULL, t.updated_at" in backfill_sql  # values_hash column now backfills NULL


def test_migration_creates_ttl_and_scheduled_at() -> None:
    # The consolidation dropped the standalone ttl / scheduled_at migrations; the
    # consolidated migration must (re)create both ORM columns or a fresh DB breaks.
    path = (
        Path(__file__).resolve().parents[3]
        / "alembic"
        / "versions"
        / "20260718140000_run_metadata_indexes_thread_state.py"
    )
    src = path.read_text(encoding="utf-8")
    assert "thread ADD COLUMN IF NOT EXISTS ttl" in src
    assert "runs ADD COLUMN IF NOT EXISTS scheduled_at" in src
    assert "idx_runs_scheduled" in src
