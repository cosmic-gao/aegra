"""Run-related Pydantic models for Agent Protocol"""

import json
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aegra_api.models.enums import DisconnectMode, MultitaskStrategy
from aegra_api.utils.status_compat import validate_run_status

# The SDK types run metadata as arbitrary JSON, so the only limit we impose is a
# serialized-size cap to close the DoS surface; shape is unconstrained.
_METADATA_MAX_BYTES = 64 * 1024


class RunCreate(BaseModel):
    """Request model for creating runs"""

    assistant_id: str = Field(..., description="Assistant to execute")
    input: dict[str, Any] | None = Field(
        default=None,
        description="Input data for the run. Optional when resuming from a checkpoint.",
    )
    config: dict[str, Any] | None = Field(default_factory=dict, description="Execution config")
    context: dict[str, Any] | None = Field(default_factory=dict, description="Execution context")
    checkpoint: dict[str, Any] | None = Field(
        default=None,
        description="Checkpoint configuration (e.g., {'checkpoint_id': '...', 'checkpoint_ns': ''})",
    )
    stream: bool = Field(default=False, description="Enable streaming response")
    stream_mode: str | list[str] | None = Field(default=None, description="Requested stream mode(s)")
    on_disconnect: DisconnectMode | None = Field(
        default=None,
        description="Behavior on client disconnect: 'cancel' (default) or 'continue'.",
    )
    on_completion: Literal["delete", "keep"] | None = Field(
        default=None,
        description="Behavior after stateless run completes: 'delete' (default) removes the ephemeral thread, 'keep' preserves it.",
    )

    multitask_strategy: MultitaskStrategy | None = Field(
        default=None,
        description=(
            "How to handle a new run when the thread already has an in-flight run "
            "(double-texting). 'reject' (server default when omitted) → 409; "
            "'interrupt' → cancel the in-flight run, keep its state; 'rollback' → "
            "cancel and discard its state; 'enqueue' → run after the in-flight one "
            "finishes. At most one run executes per thread at a time."
        ),
    )

    webhook: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Optional http(s) URL POSTed with the final Run payload when this "
            "run reaches a terminal state. Hosts resolving to private/reserved "
            "addresses are rejected unless WEBHOOK_ALLOW_PRIVATE_IPS is set."
        ),
    )

    durability: Literal["sync", "async", "exit"] | None = Field(
        default=None,
        description=(
            "When checkpoints persist: 'async' (default, persist in background), "
            "'sync' (persist before proceeding), 'exit' (persist only at the end). "
            "Forwarded to the LangGraph runtime."
        ),
    )

    checkpoint_during: bool | None = Field(
        default=None,
        description="Deprecated alias for durability: true → 'async', false → 'exit'.",
    )

    if_not_exists: Literal["create", "reject"] = Field(
        default="reject",
        description=(
            "Behavior when the target thread does not exist: 'reject' (default) "
            "returns 404; 'create' creates the thread first."
        ),
    )

    after_seconds: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Seconds to wait before starting the run. The run is created "
            "immediately in pending state and executed once the delay elapses."
        ),
    )

    # Human-in-the-loop fields (core HITL functionality)
    command: dict[str, Any] | None = Field(
        default=None,
        description="Command for resuming interrupted runs with state updates or navigation",
    )
    interrupt_before: str | list[str] | None = Field(
        default=None,
        description="Nodes to interrupt immediately before they get executed. Use '*' for all nodes.",
    )
    interrupt_after: str | list[str] | None = Field(
        default=None,
        description="Nodes to interrupt immediately after they get executed. Use '*' for all nodes.",
    )

    # Subgraph configuration
    stream_subgraphs: bool | None = Field(
        default=False,
        description="Whether to include subgraph events in streaming. When True, includes events from all subgraphs. When False (default when None), excludes subgraph events. Defaults to False for backwards compatibility.",
    )

    # Arbitrary JSON metadata, matching the SDK's ``Json`` type. Primitive
    # values reach OTEL trace attributes (``langfuse.trace.metadata.<key>``);
    # nested values are dropped there but stored/returned intact.
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary JSON metadata associated with the run and returned on the Run entity.",
    )

    @field_validator("metadata", mode="after")
    @classmethod
    def cap_metadata_size(cls, metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject metadata that isn't JSON-serializable or exceeds the anti-DoS byte cap."""
        if metadata is None:
            return metadata
        try:
            size = len(json.dumps(metadata).encode("utf-8"))
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        if size > _METADATA_MAX_BYTES:
            raise ValueError(f"metadata exceeds {_METADATA_MAX_BYTES} bytes")
        return metadata

    @model_validator(mode="after")
    def map_checkpoint_during(self) -> Self:
        """Fold the deprecated ``checkpoint_during`` flag into ``durability``."""
        if self.checkpoint_during is not None and self.durability is None:
            self.durability = "async" if self.checkpoint_during else "exit"
        self.checkpoint_during = None
        return self

    @model_validator(mode="after")
    def validate_input_command_exclusivity(self) -> Self:
        """Ensure input and command are mutually exclusive."""
        # Empty input dict alongside command: drop it for frontend compatibility.
        if self.input is not None and self.command is not None:
            if self.input == {}:
                self.input = None
            else:
                raise ValueError("Cannot specify both 'input' and 'command' - they are mutually exclusive")
        # Checkpoint-only resume keeps input=None so Pregel resumes from next=[...]
        # instead of restarting from __start__ with an empty input.
        if self.input is None and self.command is None and self.checkpoint is None:
            raise ValueError("Must specify at least one of 'input', 'command', or 'checkpoint'")
        return self


class RunsCancelRequest(BaseModel):
    """Body for bulk cancel (``POST /runs/cancel``).

    Either ``status`` alone (cancel every owned run in that state) or
    ``thread_id`` + ``run_ids`` (cancel those specific runs).
    """

    run_ids: list[str] | None = Field(default=None, description="Run ids to cancel (requires thread_id).")
    thread_id: str | None = Field(default=None, description="Thread scoping the run_ids.")
    status: Literal["pending", "running", "all"] | None = Field(
        default=None,
        description="Cancel every owned run with this status; mutually exclusive with thread_id/run_ids.",
    )

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.status is not None:
            if self.thread_id or self.run_ids:
                raise ValueError("When providing 'status', 'thread_id' and 'run_ids' must be omitted")
            return self
        if not self.thread_id or not self.run_ids:
            raise ValueError("Must provide either a status or both 'thread_id' and 'run_ids'")
        return self


class Run(BaseModel):
    """Run entity model

    Status values: pending, running, error, success, timeout, interrupted
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    run_id: str = Field(..., description="Unique identifier for the run.")
    thread_id: str = Field(..., description="Thread this run belongs to.")
    assistant_id: str = Field(..., description="Assistant that is executing this run.")
    status: str = Field(
        default="pending", description="Current run status: pending, running, error, success, timeout, or interrupted."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="metadata_dict",
        description="Arbitrary JSON metadata associated with the run.",
    )
    multitask_strategy: str = Field(
        default="reject", description="Strategy used to handle concurrent runs on the same thread."
    )
    input: dict[str, Any] | None = Field(
        default=None, description="Input data provided to the run. None for checkpoint-only resume."
    )
    output: dict[str, Any] | None = Field(
        default=None, description="Final output produced by the run, or null if not yet complete."
    )
    error_message: str | None = Field(default=None, description="Error message if the run failed.")
    config: dict[str, Any] | None = Field(
        default_factory=dict, description="Configuration passed to the graph at runtime."
    )
    context: dict[str, Any] | None = Field(
        default_factory=dict, description="Context variables available during execution."
    )
    user_id: str = Field(..., description="Identifier of the user who owns this run.")
    created_at: datetime = Field(..., description="Timestamp when the run was created.")
    updated_at: datetime = Field(..., description="Timestamp when the run was last updated.")

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status conforms to API specification."""
        if not isinstance(v, str):
            raise ValueError(f"Status must be a string, got {type(v)}")
        return validate_run_status(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def default_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any]:
        """A run row before its value is materialized carries NULL; treat as empty."""
        return v or {}

    @field_validator("multitask_strategy", mode="before")
    @classmethod
    def default_multitask_strategy(cls, v: str | None) -> str:
        """Rows created before this column existed have NULL; treat as the default."""
        return v or "reject"


class RunStatus(BaseModel):
    """Simple run status response"""

    run_id: str = Field(..., description="Unique identifier for the run.")
    status: str = Field(..., description="Current run status value.")

    message: str | None = Field(default=None, description="Optional human-readable status message.")
