"""Assistant-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aegra_api.models.enums import OnConflictBehavior


class AssistantCreate(BaseModel):
    """Request model for creating assistants"""

    assistant_id: str | None = Field(
        default=None, description="Unique assistant identifier (auto-generated if not provided)"
    )
    name: str | None = Field(
        default=None,
        description="Human-readable assistant name (auto-generated if not provided)",
    )
    description: str | None = Field(default=None, description="Assistant description")
    config: dict[str, Any] | None = Field(default_factory=dict, description="Assistant configuration")
    context: dict[str, Any] | None = Field(default_factory=dict, description="Assistant context")
    graph_id: str = Field(..., description="LangGraph graph ID from aegra.json")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Metadata to use for searching and filtering assistants."
    )
    if_exists: OnConflictBehavior | None = Field(
        default="raise", description="What to do if the assistant exists: 'raise' (default) or 'do_nothing'."
    )


class Assistant(BaseModel):
    """Assistant entity model"""

    assistant_id: str = Field(..., description="Unique identifier for the assistant.")
    name: str = Field(..., description="Human-readable name of the assistant.")
    description: str | None = Field(default=None, description="Optional description of the assistant's purpose.")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration passed to the graph at runtime.")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Context variables available to the graph during execution."
    )
    graph_id: str = Field(..., description="Identifier of the graph this assistant executes.")
    user_id: str = Field(..., description="Identifier of the user who owns this assistant.")
    version: int = Field(..., description="The version of the assistant.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, alias="metadata_dict", description="Arbitrary metadata for searching and filtering."
    )
    created_at: datetime = Field(..., description="Timestamp when the assistant was created.")
    updated_at: datetime = Field(..., description="Timestamp when the assistant was last updated.")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class AssistantUpdate(BaseModel):
    """Request model for creating assistants"""

    name: str | None = Field(default=None, description="The name of the assistant (auto-generated if not provided)")
    description: str | None = Field(default=None, description="The description of the assistant. Defaults to null.")
    config: dict[str, Any] | None = Field(default_factory=dict, description="Configuration to use for the graph.")
    graph_id: str = Field(default="agent", description="The ID of the graph")
    context: dict[str, Any] | None = Field(
        default_factory=dict,
        description="The context to use for the graph. Useful when graph is configurable.",
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Metadata to use for searching and filtering assistants."
    )


class AssistantList(BaseModel):
    """Response model for listing assistants"""

    assistants: list[Assistant]
    total: int


class AssistantSearchRequest(BaseModel):
    """Request model for assistant search"""

    name: str | None = Field(default=None, description="Filter by assistant name")
    description: str | None = Field(default=None, description="Filter by assistant description")
    graph_id: str | None = Field(default=None, description="Filter by graph ID")
    limit: int | None = Field(default=20, le=100, ge=1, description="Maximum results")
    offset: int | None = Field(default=0, ge=0, description="Results offset")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Metadata to use for searching and filtering assistants.",
    )
    sort_by: Literal["assistant_id", "name", "graph_id", "created_at", "updated_at"] | None = Field(
        default=None,
        description="Field to sort by (SDK-compatible).",
    )
    sort_order: Literal["asc", "desc"] | None = Field(
        default=None,
        description="Sort direction (SDK-compatible). Defaults to 'desc' when sort_by is set.",
    )
    select: (
        list[
            Literal[
                "assistant_id",
                "graph_id",
                "name",
                "description",
                "config",
                "context",
                "created_at",
                "updated_at",
                "metadata",
                "version",
            ]
        ]
        | None
    ) = Field(
        default=None,
        description="Fields to return for each assistant (SDK-compatible). None returns full assistants.",
    )


class AgentSchemas(BaseModel):
    """Agent schema definitions for client integration.

    Schema fields are nullable: a graph may not expose a JSON schema for every
    slot, and the SDK's GraphSchema types each as ``dict | None``.
    """

    graph_id: str = Field(..., description="Identifier of the graph these schemas describe.")
    input_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema for agent inputs.")
    output_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema for agent outputs.")
    state_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema for agent state.")
    config_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema for agent config.")
    context_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema for agent context.")
