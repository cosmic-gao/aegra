"""Pydantic models for cron job endpoints."""

from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegra_api.models.enums import MultitaskStrategy
from aegra_api.settings import settings

# Field length caps. Keep these conservative; cron metadata is small by nature.
_SCHEDULE_MAX_LEN = 256
_TIMEZONE_MAX_LEN = 64
_WEBHOOK_MAX_LEN = 2048
_STR_FIELD_MAX_LEN = 256

OnRunCompleted = Literal["delete", "keep"]


def _validate_webhook_url(value: str | None) -> str | None:
    """Reject malformed or non-http(s) webhook URLs at the API boundary.

    Cron-fired runs forward this ``webhook`` to the run, which delivers it on
    terminal state (services/webhooks.py) after a full SSRF re-check.
    """
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("webhook must use http or https scheme")
    if not parsed.netloc:
        raise ValueError("webhook must include a host")
    return value


def _map_checkpoint_during(model: "CronCreate | CronUpdate") -> None:
    """Fold the deprecated ``checkpoint_during`` flag into ``durability``."""
    if model.checkpoint_during is not None and model.durability is None:
        model.durability = "async" if model.checkpoint_during else "exit"
    model.checkpoint_during = None


def _validate_payload_size(model: BaseModel) -> None:
    """Reject payloads whose serialized JSON exceeds the configured cap."""
    cap = settings.cron.CRON_MAX_PAYLOAD_BYTES
    serialized = model.model_dump_json()
    if len(serialized.encode("utf-8")) > cap:
        raise ValueError(f"cron payload exceeds {cap} bytes")


class CronCreate(BaseModel):
    """Request body for creating a cron job (stateless or thread-bound)."""

    assistant_id: str = Field(..., max_length=_STR_FIELD_MAX_LEN)
    schedule: str = Field(..., max_length=_SCHEDULE_MAX_LEN)
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    webhook: str | None = Field(default=None, max_length=_WEBHOOK_MAX_LEN)
    on_run_completed: OnRunCompleted | None = None
    multitask_strategy: MultitaskStrategy | None = None
    end_time: datetime | None = None
    enabled: bool | None = None
    stream_mode: str | list[str] | None = None
    stream_subgraphs: bool | None = None
    timezone: str | None = Field(default=None, max_length=_TIMEZONE_MAX_LEN)
    command: dict[str, Any] | None = None
    durability: Literal["sync", "async", "exit"] | None = None
    checkpoint_during: bool | None = Field(
        default=None,
        description="Deprecated alias for durability: true → 'async', false → 'exit'.",
    )
    after_seconds: int | None = Field(default=None, ge=0)
    # Transport-only RunCreate params (stream, on_disconnect) are not exposed:
    # cron fires background runs with no client. on_completion is expressed as
    # on_run_completed above.

    @model_validator(mode="after")
    def _check(self) -> "CronCreate":
        self.webhook = _validate_webhook_url(self.webhook)
        _map_checkpoint_during(self)
        if self.end_time is not None:
            now = datetime.now(UTC)
            end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=UTC)
            if end <= now:
                raise ValueError("end_time must be in the future")
        _validate_payload_size(self)
        return self


class CronResponse(BaseModel):
    """Response model matching the SDK ``Cron`` TypedDict."""

    model_config = ConfigDict(from_attributes=True)

    cron_id: str
    assistant_id: str
    thread_id: str | None = None
    on_run_completed: OnRunCompleted | None = None
    end_time: datetime | None = None
    schedule: str
    timezone: str | None = None
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    next_run_date: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class CronUpdate(BaseModel):
    """Request body for updating an existing cron job."""

    schedule: str | None = Field(default=None, max_length=_SCHEDULE_MAX_LEN)
    end_time: datetime | None = None
    input: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    webhook: str | None = Field(default=None, max_length=_WEBHOOK_MAX_LEN)
    interrupt_before: Literal["*"] | list[str] | None = None
    interrupt_after: Literal["*"] | list[str] | None = None
    on_run_completed: OnRunCompleted | None = None
    multitask_strategy: MultitaskStrategy | None = None
    enabled: bool | None = None
    stream_mode: str | list[str] | None = None
    stream_subgraphs: bool | None = None
    timezone: str | None = Field(default=None, max_length=_TIMEZONE_MAX_LEN)
    command: dict[str, Any] | None = None
    durability: Literal["sync", "async", "exit"] | None = None
    checkpoint_during: bool | None = Field(
        default=None,
        description="Deprecated alias for durability: true → 'async', false → 'exit'.",
    )
    after_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "CronUpdate":
        self.webhook = _validate_webhook_url(self.webhook)
        _map_checkpoint_during(self)
        if self.end_time is not None:
            now = datetime.now(UTC)
            end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=UTC)
            if end <= now:
                raise ValueError("end_time must be in the future")
        _validate_payload_size(self)
        return self


class CronSearchRequest(BaseModel):
    """Request body for searching cron jobs."""

    assistant_id: str | None = None
    thread_id: str | None = None
    enabled: bool | None = None
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    metadata: dict[str, Any] | None = None
    sort_by: (
        Literal["cron_id", "assistant_id", "thread_id", "created_at", "updated_at", "next_run_date", "end_time"] | None
    ) = None
    sort_order: Literal["asc", "desc"] | None = None
    select: (
        list[
            Literal[
                "cron_id",
                "assistant_id",
                "thread_id",
                "end_time",
                "schedule",
                "timezone",
                "created_at",
                "updated_at",
                "user_id",
                "payload",
                "next_run_date",
                "metadata",
                "enabled",
            ]
        ]
        | None
    ) = None


class CronCountRequest(BaseModel):
    """Request body for counting cron jobs."""

    assistant_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] | None = None
