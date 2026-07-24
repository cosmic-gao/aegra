"""Tests for the v2 wire envelope builders."""

import typing

import langchain_protocol as lp

from aegra_api.services.event_streaming.channels import SUPPORTED_CHANNELS, is_supported_channel
from aegra_api.services.event_streaming.protocol import (
    ErrorCode,
    build_error,
    build_event,
    build_success,
)


def _literal_members(tp: object) -> set[str]:
    """Flatten a Union-of-Literal (with an optional Annotated[str] arm) to its strings."""
    members: set[str] = set()
    for arm in typing.get_args(tp):
        if isinstance(arm, str):
            members.add(arm)
            continue
        members.update(a for a in typing.get_args(arm) if isinstance(a, str))
    return members


class TestBuildEvent:
    def test_event_nests_payload_under_params_data(self) -> None:
        evt = build_event(
            "messages", {"event": "message-start", "role": "ai", "id": "m1"}, seq=7, event_id="run_event_7"
        )
        assert evt["type"] == "event"
        assert evt["seq"] == 7
        assert evt["method"] == "messages"
        assert evt["event_id"] == "run_event_7"
        assert evt["params"]["data"] == {"event": "message-start", "role": "ai", "id": "m1"}
        assert evt["params"]["namespace"] == []

    def test_namespace_is_carried(self) -> None:
        evt = build_event("values", {"x": 1}, namespace=["sub", "graph"], seq=2)
        assert evt["params"]["namespace"] == ["sub", "graph"]

    def test_event_id_omitted_when_none(self) -> None:
        evt = build_event("values", {"x": 1}, seq=1)
        assert "event_id" not in evt
        assert evt["seq"] == 1
        assert evt["method"] == "values"
        assert evt["params"]["data"] == {"x": 1}

    def test_timestamp_is_ms_epoch(self) -> None:
        evt = build_event("values", {"x": 1}, seq=1)
        ts = evt["params"]["timestamp"]
        assert isinstance(ts, int)
        assert ts > 1_600_000_000_000  # sanity: after 2020, in milliseconds not seconds


class TestBuildSuccess:
    def test_success_without_meta(self) -> None:
        assert build_success(3, {"run_id": "r1"}) == {"type": "success", "id": 3, "result": {"run_id": "r1"}}

    def test_success_with_applied_through_seq(self) -> None:
        resp = build_success(3, {}, applied_through_seq=42)
        assert resp["meta"] == {"applied_through_seq": 42}


class TestBuildError:
    def test_error_shape(self) -> None:
        assert build_error(5, "not_supported", "nope") == {
            "type": "error",
            "id": 5,
            "error": "not_supported",
            "message": "nope",
        }

    def test_error_allows_null_id(self) -> None:
        assert build_error(None, "invalid_argument", "bad")["id"] is None


class TestChannels:
    def test_known_channels_supported(self) -> None:
        for ch in ("values", "updates", "messages", "tools", "lifecycle", "custom"):
            assert is_supported_channel(ch)

    def test_custom_namespaced_channel_supported(self) -> None:
        assert is_supported_channel("custom:my_event")

    def test_empty_custom_channel_rejected(self) -> None:
        assert not is_supported_channel("custom:")

    def test_unknown_channel_rejected(self) -> None:
        assert not is_supported_channel("bogus")


class TestProtocolConformance:
    """Pin aegra's channel/error-code sets to the official langchain_protocol types."""

    def test_supported_channels_match_official_channel_literals(self) -> None:
        # ``custom:.+`` is the free-form arm, handled separately by is_supported_channel.
        official = _literal_members(lp.Channel) - {"custom:.+"}
        assert official == SUPPORTED_CHANNELS

    def test_error_code_reuses_official_type(self) -> None:
        assert ErrorCode is lp.ErrorCode

    def test_emitted_error_codes_stay_valid_in_official_set(self) -> None:
        official = _literal_members(lp.ErrorCode)
        emitted = {
            "invalid_argument",
            "unknown_command",
            "unknown_error",
            "no_such_run",
            "no_such_subscription",
            "no_such_interrupt",
            "permission_denied",
        }
        assert emitted <= official
