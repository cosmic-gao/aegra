"""Unit tests for the wait/join result shaping helper."""

import pytest

from aegra_api.services.run_waiters import wrap_run_result

pytestmark = pytest.mark.unit


def test_success_returns_output() -> None:
    assert wrap_run_result("success", {"answer": 42}, None) == {"answer": 42}


def test_interrupted_returns_output() -> None:
    assert wrap_run_result("interrupted", {"partial": True}, None) == {"partial": True}


def test_none_output_becomes_empty_dict() -> None:
    assert wrap_run_result("success", None, None) == {}


def test_error_emits_sdk_error_envelope() -> None:
    """SDK runs.wait(raise_error=True) raises only when the body carries __error__."""
    result = wrap_run_result("error", {"ignored": 1}, "boom")
    assert result == {"__error__": {"error": "error", "message": "boom"}}


def test_error_without_message_falls_back() -> None:
    assert wrap_run_result("error", None, None) == {"__error__": {"error": "error", "message": "Run failed"}}
