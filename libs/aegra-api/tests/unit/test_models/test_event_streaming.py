"""Conformance + validation tests for the v2 event-streaming request models.

The Pydantic models are kept over ``langchain_protocol``'s plain TypedDicts (they
back FastAPI bodies, expose attribute access, and add the ``ge=0`` guards below),
but their field sets are pinned to the official TypedDict so drift is caught.
"""

import langchain_protocol as lp
import pytest
from pydantic import ValidationError

from aegra_api.models.event_streaming import EventStreamRequest, ThreadCommand

pytestmark = pytest.mark.unit


class TestEventStreamRequestConformance:
    def test_fields_match_official_typeddict(self) -> None:
        official = set(lp.EventStreamRequest.__annotations__)
        assert set(EventStreamRequest.model_fields) == official

    def test_channels_only_uses_defaults(self) -> None:
        req = EventStreamRequest.model_validate({"channels": ["messages"]})
        assert req.channels == ["messages"]
        assert req.namespaces is None
        assert req.depth is None
        assert req.since is None


class TestEventStreamRequestValidation:
    """Runtime guards the official TypedDict cannot express — the reason to keep Pydantic."""

    def test_negative_since_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EventStreamRequest.model_validate({"channels": ["messages"], "since": -1})

    def test_negative_depth_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EventStreamRequest.model_validate({"channels": ["messages"], "depth": -1})

    def test_missing_channels_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EventStreamRequest.model_validate({})


class TestThreadCommand:
    def test_params_default_empty(self) -> None:
        cmd = ThreadCommand.model_validate({"id": 1, "method": "run.start"})
        assert cmd.params == {}

    def test_roundtrip_model_dump(self) -> None:
        payload = {"id": 2, "method": "input.respond", "params": {"response": "hi"}}
        assert ThreadCommand.model_validate(payload).model_dump() == payload
