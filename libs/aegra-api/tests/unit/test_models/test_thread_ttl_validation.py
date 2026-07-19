"""Unit tests for thread ttl normalization (SDK minutes-or-config input)."""

import json

import pytest
from pydantic import ValidationError

from aegra_api.models.threads import ThreadCreate, ThreadUpdate, _normalize_ttl

pytestmark = pytest.mark.unit


class TestNormalizeTtl:
    def test_none_returns_none(self) -> None:
        assert _normalize_ttl(None) is None

    def test_minutes_number_becomes_config(self) -> None:
        assert _normalize_ttl(30) == {"ttl": 30.0, "strategy": "delete"}

    def test_config_dict_normalized(self) -> None:
        assert _normalize_ttl({"ttl": 15, "strategy": "delete"}) == {"ttl": 15.0, "strategy": "delete"}

    def test_bool_rejected(self) -> None:
        with pytest.raises(ValueError, match="number of minutes or a config dict"):
            _normalize_ttl(True)

    def test_nested_bool_ttl_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive number"):
            _normalize_ttl({"ttl": True})

    def test_non_positive_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _normalize_ttl(0)

    def test_non_numeric_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="number of minutes or a config dict"):
            _normalize_ttl("30")

    def test_bad_strategy_rejected(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            _normalize_ttl({"ttl": 5, "strategy": "archive"})


class TestThreadCreateTtl:
    def test_bool_ttl_rejected_via_model(self) -> None:
        # bool is an int subclass; a JSON `true` must not silently become a 1-minute ttl
        with pytest.raises(ValidationError):
            ThreadCreate(ttl=True)

    def test_json_bool_ttl_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ThreadCreate.model_validate_json(json.dumps({"ttl": True}))

    def test_int_ttl_normalized_via_model(self) -> None:
        assert ThreadCreate(ttl=30).ttl == {"ttl": 30.0, "strategy": "delete"}

    def test_none_ttl_via_model(self) -> None:
        assert ThreadCreate(ttl=None).ttl is None


class TestThreadUpdateTtl:
    def test_bool_ttl_rejected_via_model(self) -> None:
        with pytest.raises(ValidationError):
            ThreadUpdate(ttl=True)
