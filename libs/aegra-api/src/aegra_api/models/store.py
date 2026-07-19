"""Store-related Pydantic models for Agent Protocol"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class StorePutRequest(BaseModel):
    """Request model for storing items"""

    namespace: list[str] = Field(..., description="Storage namespace")
    key: str = Field(..., description="Item key")
    value: dict[str, Any] = Field(..., description="Item value (must be a JSON object)")
    index: Literal[False] | list[str] | None = Field(
        default=None,
        description="Fields to embed for semantic search: false disables, a list selects paths, null uses store defaults.",
    )
    ttl: float | None = Field(
        default=None, ge=0, description="Item time-to-live in minutes; null keeps the store default."
    )

    @field_validator("value", mode="before")
    @classmethod
    def validate_value_is_dict(cls, v: Any) -> dict[str, Any]:
        """Validate that value is a dictionary.

        LangGraph store requires values to be dictionaries for proper
        serialization and search functionality.
        """
        if not isinstance(v, dict):
            raise ValueError(f"Value must be a dictionary (JSON object), got {type(v).__name__}")
        return v


class StoreGetResponse(BaseModel):
    """Response model for getting items"""

    key: str = Field(..., description="The item's key within its namespace.")
    value: Any = Field(..., description="The stored value.")
    namespace: list[str] = Field(..., description="The namespace path where this item is stored.")
    created_at: datetime = Field(..., description="Timestamp when the item was created.")
    updated_at: datetime = Field(..., description="Timestamp when the item was last updated.")


class StoreSearchRequest(BaseModel):
    """Request model for searching store items"""

    namespace_prefix: list[str] = Field(..., description="Namespace prefix to search")
    filter: dict[str, Any] | None = Field(
        default=None, description="Optional dictionary of key-value pairs to filter results."
    )
    query: str | None = Field(default=None, description="Search query")
    limit: int | None = Field(default=20, le=100, ge=1, description="Maximum results")
    offset: int | None = Field(default=0, ge=0, description="Results offset")
    refresh_ttl: bool | None = Field(
        default=None, description="Whether matching items refresh their TTL; null uses the store default."
    )


class StoreItem(BaseModel):
    """Store item model"""

    key: str = Field(..., description="The item's key within its namespace.")
    value: Any = Field(..., description="The stored value.")
    namespace: list[str] = Field(..., description="The namespace path where this item is stored.")
    created_at: datetime = Field(..., description="Timestamp when the item was created.")
    updated_at: datetime = Field(..., description="Timestamp when the item was last updated.")


class StoreSearchItem(StoreItem):
    """Store item with an optional relevance score from a semantic search."""

    score: float | None = Field(default=None, description="Relevance score from a semantic query, if applicable.")


class StoreSearchResponse(BaseModel):
    """Response model for store search"""

    items: list[StoreSearchItem]
    total: int
    limit: int
    offset: int


class StoreDeleteRequest(BaseModel):
    """Request body for deleting store items (SDK-compatible)."""

    namespace: list[str] = Field(..., description="Namespace path of the item to delete.")
    key: str = Field(..., description="Key of the item to delete.")


class StoreListNamespacesRequest(BaseModel):
    """Request model for listing store namespaces"""

    prefix: list[str] | None = Field(default=None, description="Filter by namespace prefix")
    suffix: list[str] | None = Field(default=None, description="Filter by namespace suffix")
    max_depth: int | None = Field(default=None, le=100, ge=1, description="Maximum namespace depth to return")
    limit: int = Field(default=100, le=1000, ge=1, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results offset")


class StoreListNamespacesResponse(BaseModel):
    """Response model for listing store namespaces"""

    namespaces: list[list[str]]
