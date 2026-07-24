"""Unit tests for v2 event-stream resume-cursor parsing (Last-Event-ID)."""

from aegra_api.api.event_streaming import _parse_since


class TestParseSince:
    def test_parses_integer_header(self) -> None:
        assert _parse_since("42") == 42

    def test_none_header_returns_none(self) -> None:
        assert _parse_since(None) is None

    def test_empty_header_returns_none(self) -> None:
        assert _parse_since("") is None

    def test_non_integer_header_ignored(self) -> None:
        # A malformed header must not crash the stream; resume from the start.
        assert _parse_since("not-a-seq") is None
