"""Unit tests for the delayed-run scheduler's empty-tick backoff."""

import pytest

from aegra_api.services.delayed_run_scheduler import _next_delay

pytestmark = pytest.mark.unit


def test_work_resets_to_base() -> None:
    assert _next_delay(20.0, had_work=True, base=5.0, cap=30.0) == 5.0


def test_idle_doubles_up_to_cap() -> None:
    assert _next_delay(5.0, had_work=False, base=5.0, cap=30.0) == 10.0
    assert _next_delay(10.0, had_work=False, base=5.0, cap=30.0) == 20.0
    assert _next_delay(20.0, had_work=False, base=5.0, cap=30.0) == 30.0
    assert _next_delay(30.0, had_work=False, base=5.0, cap=30.0) == 30.0  # capped


def test_work_after_backoff_returns_to_base() -> None:
    backed_off = _next_delay(30.0, had_work=False, base=5.0, cap=30.0)
    assert _next_delay(backed_off, had_work=True, base=5.0, cap=30.0) == 5.0
