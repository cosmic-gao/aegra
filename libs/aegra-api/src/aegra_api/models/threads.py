"""Thread-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aegra_api.utils.status_compat import validate_thread_status

# SDK ThreadSelectField values; fields Aegra does not store are omitted from rows.
ThreadSelectField = Literal[
    "thread_id", "created_at", "updated_at", "metadata", "config", "context", "status", "values", "interrupts"
]


def _normalize_ttl(value: Any) -> dict[str, Any] | None:
    """Normalize the SDK's ttl input (minutes number or config dict) to a config dict.

    Runs as a ``mode="before"`` validator so it sees the raw input and can reject a
    JSON boolean before Pydantic coerces it to 0/1 (bool is an ``int`` subclass).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("ttl must be a number of minutes or a config dict")
    if isinstance(value, (int, float)):
        value = {"ttl": value, "strategy": "delete"}
    if not isinstance(value, dict):
        raise ValueError("ttl must be a number of minutes or a config dict")
    ttl = value.get("ttl")
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl <= 0:
        raise ValueError("ttl must be a positive number of minutes")
    if value.get("strategy", "delete") != "delete":
        raise ValueError("ttl strategy must be 'delete'")
    return {"ttl": float(ttl), "strategy": "delete"}


class ThreadCreate(BaseModel):
    """Request model for creating threads"""

    model_config = ConfigDict(populate_by_name=True)

    metadata: dict[str, Any] | None = Field(default=None, description="Thread metadata")
    initial_state: dict[str, Any] | None = Field(default=None, description="LangGraph initial state")
    thread_id: str | None = Field(
        default=None,
        alias="threadId",
        description="Optional client-provided thread ID for idempotent creation",
    )
    if_exists: str | None = Field(
        default="raise",
        alias="ifExists",
        description="Behavior when thread exists: 'raise' (default) or 'do_nothing'",
    )
    graph_id: str | None = Field(default=None, description="Graph to associate with the thread (metadata.graph_id)")
    ttl: int | float | dict[str, Any] | None = Field(
        default=None,
        description="Per-thread retention: minutes, or {'ttl': minutes, 'strategy': 'delete'}.",
    )
    supersteps: list[dict[str, Any]] | None = Field(
        default=None,
        description="State updates applied to seed the new thread; each item holds {'updates': [{values, as_node, command?}]}.",
    )

    @field_validator("ttl", mode="before")
    @classmethod
    def validate_ttl(cls, v: Any) -> dict[str, Any] | None:
        return _normalize_ttl(v)


class ThreadUpdate(BaseModel):
    """Request model for updating threads"""

    metadata: dict[str, Any] | None = Field(default=None, description="Thread metadata to update")
    ttl: int | float | dict[str, Any] | None = Field(
        default=None,
        description="Per-thread retention: minutes, or {'ttl': minutes, 'strategy': 'delete'}.",
    )

    @field_validator("ttl", mode="before")
    @classmethod
    def validate_ttl(cls, v: Any) -> dict[str, Any] | None:
        return _normalize_ttl(v)


class Thread(BaseModel):
    """Thread entity model

    Status values: idle, busy, interrupted, error
    """

    model_config = ConfigDict(from_attributes=True)

    thread_id: str = Field(..., description="Unique identifier for the thread.")
    status: str = Field(default="idle", description="Current thread status: idle, busy, interrupted, or error.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata attached to the thread.")
    values: dict[str, Any] | None = Field(default=None, description="Latest state values, materialized on finalize.")
    interrupts: dict[str, list[dict[str, Any]]] | None = Field(
        default=None, description="Pending interrupts from the latest state, keyed by task id."
    )
    ttl: dict[str, Any] | None = Field(default=None, description="Per-thread retention config (include=ttl).")
    user_id: str = Field(..., description="Identifier of the user who owns this thread.")
    created_at: datetime = Field(..., description="Timestamp when the thread was created.")
    updated_at: datetime = Field(..., description="Timestamp when the thread was last updated.")

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status conforms to API specification."""
        if not isinstance(v, str):
            raise ValueError(f"Status must be a string, got {type(v)}")
        return validate_thread_status(v)


class ThreadList(BaseModel):
    """Response model for listing threads"""

    threads: list[Thread]
    total: int


class ThreadSearchRequest(BaseModel):
    """Request model for thread search"""

    metadata: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    values: dict[str, Any] | None = Field(
        default=None, description="Filter on the thread's latest state values (JSONB containment)."
    )
    ids: list[str] | None = Field(default=None, description="Restrict to these thread ids.")
    status: str | None = Field(default=None, description="Thread status filter (idle, busy, interrupted, error)")
    limit: int | None = Field(default=20, le=100, ge=1, description="Maximum results")
    offset: int | None = Field(default=0, ge=0, description="Results offset")
    sort_by: Literal["thread_id", "status", "created_at", "updated_at", "state_updated_at"] | None = Field(
        default=None,
        description="Field to sort by (SDK-compatible). 'state_updated_at' maps to updated_at.",
    )
    sort_order: Literal["asc", "desc"] | None = Field(
        default=None,
        description="Sort direction (SDK-compatible). Defaults to 'desc' when sort_by is set.",
    )
    select: list[ThreadSelectField] | None = Field(default=None, description="Return only these thread fields.")
    extract: dict[str, str] | None = Field(
        default=None,
        description="Extra keys extracted from values/metadata/interrupts via dot/bracket paths, e.g. {'last': 'values.messages[-1].content'}.",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        """Validate status filter conforms to API specification."""
        if v is not None:
            return validate_thread_status(v)
        return v


class ThreadPruneRequest(BaseModel):
    """Request body for ``POST /threads/prune``."""

    thread_ids: list[str] = Field(..., description="Threads to prune.")
    strategy: Literal["delete", "keep_latest"] = Field(
        default="delete",
        description="'delete' removes the threads entirely; 'keep_latest' keeps only the newest checkpoint per namespace.",
    )


class ThreadSearchResponse(BaseModel):
    """Response model for thread search"""

    threads: list[Thread]
    total: int
    limit: int
    offset: int


class ThreadCheckpoint(BaseModel):
    """Checkpoint identifier for thread history"""

    checkpoint_id: str | None = None
    thread_id: str | None = None
    checkpoint_ns: str | None = ""


class ThreadCheckpointPostRequest(BaseModel):
    """Request model for fetching thread checkpoint"""

    checkpoint: ThreadCheckpoint = Field(description="Checkpoint to fetch")
    subgraphs: bool | None = Field(default=False, description="Include subgraph states")


class ThreadState(BaseModel):
    """Thread state model for history endpoint"""

    values: dict[str, Any] = Field(description="Channel values (messages, etc.)")
    next: list[str] = Field(default_factory=list, description="Next nodes to execute")
    tasks: list[dict[str, Any]] = Field(default_factory=list, description="Tasks to execute")
    interrupts: list[dict[str, Any]] = Field(default_factory=list, description="Interrupt data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Checkpoint metadata")
    created_at: datetime | None = Field(default=None, description="Timestamp of state creation")
    checkpoint: ThreadCheckpoint = Field(description="Current checkpoint")
    parent_checkpoint: ThreadCheckpoint | None = Field(default=None, description="Parent checkpoint")
    checkpoint_id: str | None = Field(default=None, description="Checkpoint ID (for backward compatibility)")
    parent_checkpoint_id: str | None = Field(
        default=None, description="Parent checkpoint ID (for backward compatibility)"
    )


class ThreadStateUpdate(BaseModel):
    """Request model for updating thread state"""

    values: dict[str, Any] | list[dict[str, Any]] | None = Field(
        default=None, description="The values to update the state with"
    )
    checkpoint: dict[str, Any] | None = Field(default=None, description="The checkpoint to update the state of")
    checkpoint_id: str | None = Field(default=None, description="Optional checkpoint ID to update from")
    as_node: str | None = Field(default=None, description="Update the state as if this node had just executed")
    # Also support query-like parameters for GET-like behavior via POST
    subgraphs: bool | None = Field(default=False, description="Include states from subgraphs")
    checkpoint_ns: str | None = Field(default=None, description="Checkpoint namespace")


class ThreadStateUpdateResponse(BaseModel):
    """Response model for thread state update"""

    checkpoint: dict[str, Any] = Field(description="The checkpoint that was created/updated")


class ThreadHistoryRequest(BaseModel):
    """Request model for thread history endpoint"""

    limit: int | None = Field(default=10, ge=1, le=1000, description="Number of states to return")
    before: dict[str, Any] | str | None = Field(
        default=None,
        description="Return states before this checkpoint (checkpoint ID string, raw checkpoint dict, or RunnableConfig with 'configurable' key)",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Filter by metadata")
    checkpoint: dict[str, Any] | None = Field(default=None, description="Checkpoint for subgraph filtering")
    subgraphs: bool | None = Field(default=False, description="Include states from subgraphs")
    checkpoint_ns: str | None = Field(default=None, description="Checkpoint namespace")
