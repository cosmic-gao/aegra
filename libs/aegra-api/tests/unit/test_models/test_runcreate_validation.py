"""Tests for RunCreate model validation."""

import pytest
from pydantic import ValidationError

from aegra_api.models.runs import RunCreate


class TestRunCreateValidation:
    """Tests for RunCreate input/command validation."""

    def test_checkpoint_only_payload_preserves_none_input(self):
        """Checkpoint-only payloads must keep input as None so LangGraph resumes
        from the checkpoint instead of restarting the graph from __start__.

        Regression test: previously the validator coerced input to ``{}`` which
        LangGraph Pregel treats as "new input" and re-enters __start__, ignoring
        the checkpoint's ``next=[...]``.
        """
        run_create = RunCreate(
            assistant_id="agent",
            checkpoint={"checkpoint_id": "chk-1", "checkpoint_ns": ""},
        )

        assert run_create.input is None
        assert run_create.command is None
        assert run_create.checkpoint == {"checkpoint_id": "chk-1", "checkpoint_ns": ""}

    def test_rejects_payload_without_input_command_or_checkpoint(self):
        """Ensure payloads with no input, command, or checkpoint are rejected."""
        with pytest.raises(ValueError, match="Must specify at least one of 'input', 'command', or 'checkpoint'"):
            RunCreate(assistant_id="agent")


class TestRunCreateMetadataValidation:
    """``RunCreate.metadata`` accepts arbitrary JSON (SDK ``Json`` type).

    The only limit is a serialized-size cap to close the DoS surface; shape is
    unconstrained (nested objects, lists, and arbitrary keys are all allowed),
    matching what the LangGraph SDK sends.
    """

    def _payload(self, **overrides):
        base = {"assistant_id": "agent", "input": {"x": 1}}
        base.update(overrides)
        return base

    def test_none_metadata_accepted(self):
        run_create = RunCreate(**self._payload(metadata=None))
        assert run_create.metadata is None

    def test_empty_dict_metadata_accepted(self):
        run_create = RunCreate(**self._payload(metadata={}))
        assert run_create.metadata == {}

    def test_primitive_values_accepted(self):
        run_create = RunCreate(**self._payload(metadata={"tenant": "acme", "retries": 3, "ratio": 0.5, "flag": True}))
        assert run_create.metadata["tenant"] == "acme"

    def test_nested_dict_value_accepted(self):
        run_create = RunCreate(**self._payload(metadata={"k": {"nested": {"deep": 1}}}))
        assert run_create.metadata == {"k": {"nested": {"deep": 1}}}

    def test_list_value_accepted(self):
        run_create = RunCreate(**self._payload(metadata={"k": [1, 2, 3]}))
        assert run_create.metadata == {"k": [1, 2, 3]}

    def test_many_keys_accepted(self):
        run_create = RunCreate(**self._payload(metadata={f"k{i}": i for i in range(200)}))
        assert len(run_create.metadata) == 200

    def test_dotted_and_arbitrary_keys_accepted(self):
        run_create = RunCreate(**self._payload(metadata={"a.b.c": 1, "any key!": 2}))
        assert run_create.metadata["a.b.c"] == 1

    def test_oversized_metadata_rejected(self):
        with pytest.raises(ValidationError, match="exceeds"):
            RunCreate(**self._payload(metadata={"big": "v" * 70_000}))

    def test_non_json_serializable_metadata_rejected(self):
        # A non-JSON value (set) must raise a clean validation error, not a 500
        # from json.dumps in the size-cap validator.
        with pytest.raises(ValidationError, match="JSON-serializable"):
            RunCreate(**self._payload(metadata={"bad": {1, 2}}))
