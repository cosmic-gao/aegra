"""Thread endpoints for Agent Protocol"""

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.active_runs import active_runs
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.database import db_manager
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import ThreadState as ThreadStateORM
from aegra_api.core.orm import get_session
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.core.serializers.langgraph import LangGraphSerializer
from aegra_api.models import (
    Thread,
    ThreadCheckpoint,
    ThreadCheckpointPostRequest,
    ThreadCreate,
    ThreadHistoryRequest,
    ThreadList,
    ThreadSearchRequest,
    ThreadState,
    ThreadStateUpdate,
    ThreadStateUpdateResponse,
    ThreadUpdate,
    User,
)
from aegra_api.models.errors import CONFLICT, NOT_FOUND
from aegra_api.models.threads import ThreadPruneRequest
from aegra_api.services.run_cleanup import delete_thread_by_id
from aegra_api.services.run_status import materialize_thread_state
from aegra_api.services.streaming_service import streaming_service
from aegra_api.services.thread_state_service import ThreadStateService
from aegra_api.utils.extract import extract_path_value, validate_extract
from aegra_api.utils.run_utils import strip_pinned_config_keys

router = APIRouter(tags=["Threads"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)

thread_state_service = ThreadStateService()


# --- Sort resolution for /threads/search ---

_DEFAULT_SORT_FIELD = "created_at"
_DEFAULT_SORT_ASC = False


def _resolve_sort(request: ThreadSearchRequest) -> tuple[Any, bool]:
    """Resolve (ORM column, is_ascending) for /threads/search.

    ``sort_by`` is Pydantic-validated against the column Literal;
    ``state_updated_at`` maps to updated_at (values materialize on finalize,
    which touches updated_at in the same transaction).
    """
    if request.sort_by:
        field = "updated_at" if request.sort_by == "state_updated_at" else request.sort_by
        asc = (request.sort_order or "desc").lower() == "asc"
        return getattr(ThreadORM, field), asc
    return getattr(ThreadORM, _DEFAULT_SORT_FIELD), _DEFAULT_SORT_ASC


# --- Helper for safe ORM -> Pydantic conversion (Test/Mock compatible) ---


def _serialize_thread(
    thread_orm: ThreadORM,
    default_metadata: dict[str, Any] | None = None,
    *,
    include_ttl: bool = False,
    state: ThreadStateORM | None = None,
) -> Thread:
    """
    Safely converts ThreadORM to Thread model using dictionary construction.
    This handles None values and MagicMocks that appear in tests, preventing
    Pydantic V2 ValidationErrors.
    """

    def _coerce_str(val: Any, default: str) -> str:
        try:
            s = str(val)
            # Handle MagicMock objects in tests converting to strings like "<MagicMock...>"
            return default if "MagicMock" in s else s
        except Exception:
            return default

    def _coerce_dict(val: Any, default: dict[str, Any]) -> dict[str, Any]:
        if val is None:
            return default
        if isinstance(val, dict):
            return val
        # Try to convert dict-like objects (mocks)
        with contextlib.suppress(Exception):
            if hasattr(val, "items"):
                return dict(val.items())
        return default

    # 1. ID
    t_id = _coerce_str(getattr(thread_orm, "thread_id", None), "unknown")

    # 2. Status
    status = _coerce_str(getattr(thread_orm, "status", "idle"), "idle")

    # 3. User ID
    u_id = _coerce_str(getattr(thread_orm, "user_id", ""), "")

    # 4. Metadata (map metadata_json -> metadata)
    # Use provided default if ORM is None (e.g. during creation before refresh)
    meta_source = getattr(thread_orm, "metadata_json", None)
    if meta_source is None and default_metadata is not None:
        meta_source = default_metadata
    metadata = _coerce_dict(meta_source, {})

    # 5. Timestamps (Default to NOW if None/Mock fails)
    c_at = getattr(thread_orm, "created_at", None)
    if not isinstance(c_at, datetime):
        c_at = datetime.now(UTC)

    u_at = getattr(thread_orm, "updated_at", None)
    if not isinstance(u_at, datetime):
        u_at = datetime.now(UTC)

    # Latest state lives in thread_state (1:1); present only when the caller joined it.
    values = getattr(state, "values", None) if state is not None else None
    interrupts = getattr(state, "interrupts", None) if state is not None else None
    ttl = getattr(thread_orm, "ttl", None) if include_ttl else None

    # Validate from dict (more robust than validate(orm_obj) for partial mocks)
    return Thread.model_validate(
        {
            "thread_id": t_id,
            "status": status,
            "metadata": metadata,
            "values": values if isinstance(values, dict) else None,
            "interrupts": interrupts if isinstance(interrupts, dict) else None,
            "ttl": ttl if isinstance(ttl, dict) else None,
            "user_id": u_id,
            "created_at": c_at,
            "updated_at": u_at,
        }
    )


async def _apply_supersteps(
    session: AsyncSession,
    thread: ThreadORM,
    supersteps: list[dict[str, Any]],
    user: User,
) -> None:
    """Seed a new thread by applying superstep state updates in order.

    Each superstep item is ``{"updates": [{"values", "as_node"?}]}`` (SDK
    shape). Requires the thread to carry a ``graph_id`` so state updates can
    resolve the graph's channels.
    """
    graph_id = (thread.metadata_json or {}).get("graph_id")
    if not graph_id:
        raise HTTPException(422, "supersteps require a graph_id on the thread")

    from aegra_api.services.langgraph_service import create_thread_config, get_langgraph_service

    service = get_langgraph_service()
    raw_config = create_thread_config(thread.thread_id, user)
    config = cast("RunnableConfig", raw_config)
    async with service.get_graph(graph_id, config=raw_config, access_context="threads.update", user=user) as agent:
        agent = agent.with_config(config)
        for step in supersteps:
            updates = step.get("updates") or []
            for item in updates:
                if item.get("command") is not None:
                    raise HTTPException(422, "supersteps with 'command' are not supported")
                await agent.aupdate_state(config, item.get("values"), as_node=item.get("as_node"))
    await _materialize_thread_state(session, thread, user)


async def _materialize_thread_state(session: AsyncSession, thread: ThreadORM, user: User) -> None:
    """Best-effort refresh of the thread row's materialized values/interrupts.

    Feeds thread search's ``values`` filter and ``select``. A read failure must
    not fail the state update that triggered it, so errors are swallowed.
    """
    graph_id = (thread.metadata_json or {}).get("graph_id")
    if not graph_id:
        return
    from aegra_api.services.langgraph_service import create_thread_config, get_langgraph_service

    serializer = GeneralSerializer()
    service = get_langgraph_service()
    raw_config = create_thread_config(thread.thread_id, user)
    config = cast("RunnableConfig", raw_config)
    try:
        async with service.get_graph(graph_id, config=raw_config, access_context="threads.read", user=user) as agent:
            state = await agent.aget_state(config)
        values = serializer.serialize(state.values) if state.values is not None else None
        interrupts = LangGraphSerializer().build_interrupts_map(state)
    except Exception as exc:
        logger.warning("Could not materialize thread state", thread_id=thread.thread_id, error=str(exc))
        return
    if isinstance(values, dict):
        await materialize_thread_state(session, thread.thread_id, values, interrupts)
    thread.updated_at = datetime.now(UTC)
    await session.commit()


# --- Endpoints ---


@router.post("/threads", response_model=Thread, responses={**CONFLICT})
async def create_thread(
    request: ThreadCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Thread:
    """Create a new conversation thread.

    Threads hold conversation state and checkpoint history. Provide a
    `thread_id` for idempotent creation, or let the server generate one.
    Set `if_exists` to `"do_nothing"` to return the existing thread when the
    ID already exists instead of raising a 409 conflict.
    """
    # Authorization check
    ctx = build_auth_context(user, "threads", "create")
    value = request.model_dump()
    filters = await handle_event(ctx, value)

    # If handler modified metadata, update request
    if filters and "metadata" in filters:
        handler_meta = filters["metadata"]
        if isinstance(handler_meta, dict):
            request.metadata = {**(request.metadata or {}), **handler_meta}
    elif value.get("metadata"):
        # Handler may have modified value dict directly
        handler_meta = value["metadata"]
        if isinstance(handler_meta, dict):
            request.metadata = {**(request.metadata or {}), **handler_meta}

    thread_id = request.thread_id or str(uuid4())

    if request.thread_id:
        existing_stmt = select(ThreadORM).where(
            ThreadORM.thread_id == thread_id,
            ThreadORM.user_id == user.identity,
        )
        existing = await session.scalar(existing_stmt)

        if existing:
            if request.if_exists == "do_nothing":
                return _serialize_thread(existing)
            else:
                raise HTTPException(409, f"Thread '{thread_id}' already exists")

    metadata = request.metadata or {}
    # Always enforce owner from authenticated user
    metadata["owner"] = user.identity
    # Preserve client-provided values; only set defaults if missing.
    metadata.setdefault("assistant_id", None)
    metadata.setdefault("graph_id", request.graph_id)
    metadata.setdefault("thread_name", "")

    thread_orm = ThreadORM(
        thread_id=thread_id,
        status="idle",
        metadata_json=metadata,
        ttl=request.ttl,
        user_id=user.identity,
    )

    session.add(thread_orm)
    await session.commit()

    if request.supersteps:
        await _apply_supersteps(session, thread_orm, request.supersteps, user)

    with contextlib.suppress(Exception):
        await session.refresh(thread_orm)

    # Pass metadata explicitly in case refresh failed (tests/mocks)
    return _serialize_thread(thread_orm, default_metadata=metadata)


@router.get("/threads", response_model=ThreadList)
async def list_threads(
    user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)
) -> ThreadList:
    """List all threads owned by the authenticated user.

    Returns every thread without filtering. Use the search endpoint for
    filtered queries.
    """
    # Authorization check (search action for listing)
    ctx = build_auth_context(user, "threads", "search")
    value = {}
    filters = await handle_event(ctx, value)

    # Build query with filters if provided
    stmt = select(ThreadORM).where(ThreadORM.user_id == user.identity)
    if filters:
        # Apply filters from authorization handler
        # For now, we'll apply user_id filter which is already there
        # Additional filters can be added here based on handler response
        pass
    result = await session.scalars(stmt)
    rows = result.all()

    # Use safe serialization
    user_threads = [_serialize_thread(t) for t in rows]
    return ThreadList(threads=user_threads, total=len(user_threads))


@router.get("/threads/{thread_id}", response_model=Thread, responses={**NOT_FOUND})
async def get_thread(
    thread_id: str,
    include: list[str] | None = Query(None, description="Extra fields to include; supports 'ttl'."),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Thread:
    """Get a thread by its ID.

    Returns 404 if the thread does not exist or does not belong to the
    authenticated user.
    """
    # Authorization check
    ctx = build_auth_context(user, "threads", "read")
    value = {"thread_id": thread_id}
    await handle_event(ctx, value)

    stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    thread = await session.scalar(stmt)
    if not thread:
        raise HTTPException(404, f"Thread '{thread_id}' not found")

    # The SDK sends include as a comma-joined single param.
    include_parts = {p for raw in include or [] for p in raw.split(",") if p}
    # Latest values/interrupts live in thread_state (1:1); fetch for the detail view.
    state = await session.scalar(select(ThreadStateORM).where(ThreadStateORM.thread_id == thread_id))
    return _serialize_thread(thread, include_ttl="ttl" in include_parts, state=state)


@router.patch("/threads/{thread_id}", response_model=Thread, responses={**NOT_FOUND})
async def update_thread(
    thread_id: str,
    request: ThreadUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Thread:
    """Update a thread's metadata.

    Merges the provided metadata with the existing metadata (shallow merge).
    """
    # Authorization check
    ctx = build_auth_context(user, "threads", "update")
    value = {**request.model_dump(), "thread_id": thread_id}
    filters = await handle_event(ctx, value)

    # If handler modified metadata, update request
    if filters and "metadata" in filters:
        handler_meta = filters["metadata"]
        if isinstance(handler_meta, dict):
            request.metadata = {**(request.metadata or {}), **handler_meta}
    elif value.get("metadata"):
        handler_meta = value["metadata"]
        if isinstance(handler_meta, dict):
            request.metadata = {**(request.metadata or {}), **handler_meta}

    stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    thread = await session.scalar(stmt)

    if not thread:
        raise HTTPException(404, f"Thread '{thread_id}' not found")

    thread.updated_at = datetime.now(UTC)

    if request.metadata:
        current_metadata = dict(thread.metadata_json or {})
        current_metadata.update(request.metadata)
        thread.metadata_json = current_metadata
    if isinstance(request.ttl, dict):
        # The Pydantic validator normalizes int minutes to the config dict.
        thread.ttl = request.ttl

    await session.commit()
    await session.refresh(thread)

    return _serialize_thread(thread, include_ttl=request.ttl is not None)


@router.get("/threads/{thread_id}/state", response_model=ThreadState, responses={**NOT_FOUND})
async def get_thread_state(
    thread_id: str,
    subgraphs: bool = Query(False, description="Include states from subgraphs"),
    checkpoint_ns: str | None = Query(None, description="Checkpoint namespace to scope lookup"),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ThreadState:
    """Get the current state of a thread.

    Returns the latest checkpoint's values, pending next nodes, interrupt
    data, and metadata. If the thread has no associated graph yet (no runs
    executed), returns an empty state.
    """
    try:
        stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
        thread = await session.scalar(stmt)
        if not thread:
            raise HTTPException(404, f"Thread '{thread_id}' not found")

        thread_metadata = thread.metadata_json or {}
        graph_id = thread_metadata.get("graph_id")
        if not graph_id:
            logger.info(
                "state GET: no graph_id set for thread %s, returning empty state",
                thread_id,
            )
            empty_checkpoint = ThreadCheckpoint(
                checkpoint_id=None,
                thread_id=thread_id,
                checkpoint_ns="",
            )
            return ThreadState(
                values={},
                next=[],
                tasks=[],
                interrupts=[],
                metadata={},
                created_at=None,
                checkpoint=empty_checkpoint,
                parent_checkpoint=None,
                checkpoint_id=None,
                parent_checkpoint_id=None,
            )

        from aegra_api.services.langgraph_service import (
            create_thread_config,
            get_langgraph_service,
        )

        langgraph_service = get_langgraph_service()
        config: dict[str, Any] = create_thread_config(thread_id, user)
        if checkpoint_ns:
            config["configurable"]["checkpoint_ns"] = checkpoint_ns

        try:
            async with langgraph_service.get_graph(
                graph_id,
                config=config,
                access_context="threads.read",
                user=user,
            ) as agent:
                agent = agent.with_config(config)
                # NOTE: LangGraph only exposes subgraph checkpoints while the run is
                # interrupted. See https://docs.langchain.com/oss/python/langgraph/use-subgraphs#view-subgraph-state
                state_snapshot = await agent.aget_state(config, subgraphs=subgraphs)

                if not state_snapshot:
                    logger.info(
                        "state GET: no checkpoint found for thread %s (checkpoint_ns=%s)",
                        thread_id,
                        checkpoint_ns,
                    )
                    raise HTTPException(404, f"No state found for thread '{thread_id}'")

                thread_state = thread_state_service.convert_snapshot_to_thread_state(
                    state_snapshot, thread_id, subgraphs=subgraphs
                )

                logger.debug(
                    "state GET: thread_id=%s checkpoint_id=%s subgraphs=%s checkpoint_ns=%s",
                    thread_id,
                    thread_state.checkpoint.checkpoint_id,
                    subgraphs,
                    checkpoint_ns,
                )

                return thread_state
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to retrieve latest state for thread '%s'", thread_id)
            raise HTTPException(500, f"Failed to retrieve thread state: {str(e)}") from e

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving latest state for thread '%s'", thread_id)
        raise HTTPException(500, f"Error retrieving thread state: {str(e)}") from e


@router.post("/threads/{thread_id}/state", responses={**NOT_FOUND})
async def update_thread_state(
    thread_id: str,
    request: ThreadStateUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ThreadState | ThreadStateUpdateResponse:
    """Update thread state or retrieve it via POST.

    When `values` is provided, creates a new checkpoint with the updated state.
    Use `as_node` to attribute the update to a specific graph node. When
    `values` is null AND `as_node` is not provided, this endpoint acts as a
    POST-based alternative to the GET state endpoint (useful when passing
    complex checkpoint/subgraph parameters in the request body).

    When `values` is null AND `as_node` is provided (e.g. ``as_node="__copy__"``
    as LangGraph Studio sends for "Re-run from here"), this creates a new
    checkpoint derived from the supplied ``checkpoint_id`` without applying
    any state change — used to anchor a subsequent run as a fork of that
    checkpoint rather than of the thread's latest state.
    """
    # GET-shim only fires when body has no mutation or checkpoint targeting.
    if request.values is None and request.as_node is None and request.checkpoint_id is None and not request.checkpoint:
        return await get_thread_state(
            thread_id=thread_id,
            subgraphs=request.subgraphs or False,
            checkpoint_ns=request.checkpoint_ns,
            user=user,
            session=session,
        )

    try:
        stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
        thread = await session.scalar(stmt)
        if not thread:
            raise HTTPException(404, f"Thread '{thread_id}' not found")

        thread_metadata = thread.metadata_json or {}
        graph_id = thread_metadata.get("graph_id")
        if not graph_id:
            raise HTTPException(
                400,
                f"Thread '{thread_id}' has no associated graph. Cannot update state.",
            )

        from aegra_api.services.langgraph_service import (
            create_thread_config,
            get_langgraph_service,
        )

        langgraph_service = get_langgraph_service()
        config: dict[str, Any] = create_thread_config(thread_id, user)

        if request.checkpoint_id:
            config["configurable"]["checkpoint_id"] = request.checkpoint_id
        if request.checkpoint:
            config["configurable"].update(strip_pinned_config_keys(request.checkpoint))
        if request.checkpoint_ns:
            config["configurable"]["checkpoint_ns"] = request.checkpoint_ns

        try:
            async with langgraph_service.get_graph(
                graph_id,
                config=config,
                access_context="threads.update",
                user=user,
            ) as agent:
                # Update state using aupdate_state method
                # This creates a new checkpoint with the updated values
                agent = agent.with_config(config)

                # Handle values - can be dict or list of dicts
                update_values = request.values
                if isinstance(update_values, list):
                    # If it's a list, use the first dict or convert to dict
                    if update_values and isinstance(update_values[0], dict):
                        # Merge all dicts in the list
                        merged = {}
                        for item in update_values:
                            if isinstance(item, dict):
                                merged.update(item)
                        update_values = merged
                    else:
                        update_values = update_values[0] if update_values else None

                # Always pass as_node: without it the graph may resume execution
                # instead of only updating state, which can fail if state doesn't match graph flow.
                try:
                    updated_config = await agent.aupdate_state(config, update_values, as_node=request.as_node)
                except Exception as update_error:
                    logger.exception(
                        "aupdate_state failed for thread %s: %s",
                        thread_id,
                        update_error,
                        exc_info=True,
                    )
                    raise

                # Extract checkpoint info from the updated config
                # aupdate_state returns the updated config dict
                if not isinstance(updated_config, dict):
                    logger.error(
                        "aupdate_state returned non-dict: %s (type: %s)",
                        updated_config,
                        type(updated_config),
                    )
                    raise HTTPException(
                        500,
                        f"Unexpected return type from aupdate_state: {type(updated_config)}",
                    )

                checkpoint_info = {
                    "checkpoint_id": updated_config.get("configurable", {}).get("checkpoint_id"),
                    "thread_id": thread_id,
                    "checkpoint_ns": updated_config.get("configurable", {}).get("checkpoint_ns", ""),
                }

                logger.info(
                    "state POST: updated state for thread %s checkpoint_id=%s",
                    thread_id,
                    checkpoint_info.get("checkpoint_id"),
                )

                await _materialize_thread_state(session, thread, user)
                return ThreadStateUpdateResponse(checkpoint=checkpoint_info)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to update state for thread '%s'", thread_id)
            raise HTTPException(500, f"Failed to update thread state: {str(e)}") from e

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error updating state for thread '%s'", thread_id)
        raise HTTPException(500, f"Error updating thread state: {str(e)}") from e


@router.get("/threads/{thread_id}/state/{checkpoint_id}", response_model=ThreadState, responses={**NOT_FOUND})
async def get_thread_state_at_checkpoint(
    thread_id: str,
    checkpoint_id: str,
    subgraphs: bool | None = Query(False, description="Include states from subgraphs"),
    checkpoint_ns: str | None = Query(None, description="Checkpoint namespace to scope lookup"),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ThreadState:
    """Get the thread state at a specific checkpoint.

    Use this to inspect historical state at any point in the thread's
    execution history. Returns 404 if the checkpoint does not exist.
    """
    try:
        stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
        thread = await session.scalar(stmt)
        if not thread:
            raise HTTPException(404, f"Thread '{thread_id}' not found")

        thread_metadata = thread.metadata_json or {}
        graph_id = thread_metadata.get("graph_id")
        if not graph_id:
            raise HTTPException(404, f"Thread '{thread_id}' has no associated graph")

        from aegra_api.services.langgraph_service import (
            create_thread_config,
            get_langgraph_service,
        )

        langgraph_service = get_langgraph_service()

        config: dict[str, Any] = create_thread_config(thread_id, user)
        config["configurable"]["checkpoint_id"] = checkpoint_id
        if checkpoint_ns:
            config["configurable"]["checkpoint_ns"] = checkpoint_ns

        try:
            async with langgraph_service.get_graph(
                graph_id,
                config=config,
                access_context="threads.read",
                user=user,
            ) as agent:
                agent = agent.with_config(config)
                state_snapshot = await agent.aget_state(config, subgraphs=subgraphs or False)

                if not state_snapshot:
                    raise HTTPException(
                        404,
                        f"No state found at checkpoint '{checkpoint_id}' for thread '{thread_id}'",
                    )

                # Convert snapshot to ThreadCheckpoint using service
                thread_checkpoint = thread_state_service.convert_snapshot_to_thread_state(
                    state_snapshot,
                    thread_id,
                    subgraphs=subgraphs or False,
                )

                return thread_checkpoint
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(
                "Failed to retrieve state at checkpoint '%s' for thread '%s'",
                checkpoint_id,
                thread_id,
            )
            raise HTTPException(
                500,
                f"Failed to retrieve state at checkpoint '{checkpoint_id}': {str(e)}",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving checkpoint '%s' for thread '%s'", checkpoint_id, thread_id)
        raise HTTPException(500, f"Error retrieving checkpoint '{checkpoint_id}': {str(e)}") from e


@router.post("/threads/{thread_id}/state/checkpoint", response_model=ThreadState, responses={**NOT_FOUND})
async def get_thread_state_at_checkpoint_post(
    thread_id: str,
    request: ThreadCheckpointPostRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ThreadState:
    """Get the thread state at a specific checkpoint (POST variant).

    Identical to the GET checkpoint endpoint but accepts the checkpoint
    configuration in the request body. Useful when the checkpoint namespace
    contains characters that are awkward in URL paths.
    """
    checkpoint = request.checkpoint
    if not checkpoint.checkpoint_id:
        raise HTTPException(400, "checkpoint_id is required in checkpoint configuration")

    subgraphs = request.subgraphs
    checkpoint_ns = checkpoint.checkpoint_ns if checkpoint.checkpoint_ns else None

    output = await get_thread_state_at_checkpoint(
        thread_id,
        checkpoint.checkpoint_id,
        subgraphs,
        checkpoint_ns,
        user,
        session,
    )
    return output


@router.post("/threads/{thread_id}/history", response_model=list[ThreadState], responses={**NOT_FOUND})
async def get_thread_history_post(
    thread_id: str,
    request: ThreadHistoryRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[ThreadState]:
    """Get the checkpoint history for a thread (POST variant).

    Returns a list of past states ordered from newest to oldest. Use `limit`
    to control how many states are returned and `before` to paginate.
    """
    try:
        limit = request.limit or 10
        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise HTTPException(422, "Invalid limit; must be an integer between 1 and 1000")

        before = request.before
        metadata = request.metadata
        checkpoint = request.checkpoint or {}
        subgraphs = bool(request.subgraphs) if request.subgraphs is not None else False
        checkpoint_ns = request.checkpoint_ns

        stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
        thread = await session.scalar(stmt)
        if not thread:
            raise HTTPException(404, f"Thread '{thread_id}' not found")

        thread_metadata = thread.metadata_json or {}
        graph_id = thread_metadata.get("graph_id")
        if not graph_id:
            logger.info(f"history POST: no graph_id set for thread {thread_id}")
            return []

        from aegra_api.services.langgraph_service import (
            create_thread_config,
            get_langgraph_service,
        )

        langgraph_service = get_langgraph_service()

        config: dict[str, Any] = create_thread_config(thread_id, user)
        if checkpoint:
            cfg_cp = strip_pinned_config_keys(checkpoint)
            if checkpoint_ns is not None:
                cfg_cp.setdefault("checkpoint_ns", checkpoint_ns)
            config["configurable"].update(cfg_cp)
        elif checkpoint_ns is not None:
            config["configurable"]["checkpoint_ns"] = checkpoint_ns

        # Convert `before` to a RunnableConfig for aget_state_history.
        # The SDK sends `before` as either a checkpoint ID string, a raw
        # checkpoint dict, or a full RunnableConfig with a "configurable" key.
        # No thread_id scrub here: aget_state_history reads only checkpoint_id
        # from `before` (the thread comes from the main config, pinned above).
        before_config: dict[str, Any] | None = None
        if isinstance(before, str):
            before_config = {"configurable": {"checkpoint_id": before}}
        elif isinstance(before, dict):
            before_config = before if "configurable" in before else {"configurable": before}

        state_snapshots = []
        kwargs: dict[str, Any] = {
            "limit": limit,
            "before": before_config,
        }
        if metadata is not None:
            kwargs["metadata"] = metadata

        async with langgraph_service.get_graph(
            graph_id,
            config=config,
            access_context="threads.read",
            user=user,
        ) as agent:
            # Some LangGraph versions support subgraphs flag; pass if available
            try:
                async for snapshot in agent.aget_state_history(config, subgraphs=subgraphs, **kwargs):
                    state_snapshots.append(snapshot)
            except TypeError:
                # Fallback if subgraphs not supported in this version
                async for snapshot in agent.aget_state_history(config, **kwargs):
                    state_snapshots.append(snapshot)

        # Convert outside the async with so the graph context is closed first
        thread_states = thread_state_service.convert_snapshots_to_thread_states(state_snapshots, thread_id)

        return thread_states

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in history POST for thread %s", thread_id)
        msg = str(e).lower()
        if "not found" in msg or "no checkpoint" in msg:
            return []
        raise HTTPException(500, f"Error retrieving thread history: {str(e)}") from e


@router.get("/threads/{thread_id}/history", response_model=list[ThreadState], responses={**NOT_FOUND})
async def get_thread_history_get(
    thread_id: str,
    limit: int = Query(10, ge=1, le=1000, description="Number of states to return"),
    before: str | None = Query(None, description="Return states before this checkpoint ID"),
    subgraphs: bool | None = Query(False, description="Include states from subgraphs"),
    checkpoint_ns: str | None = Query(None, description="Checkpoint namespace"),
    metadata: str | None = Query(None, description="JSON-encoded metadata filter"),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[ThreadState]:
    """Get the checkpoint history for a thread.

    Returns a list of past states ordered from newest to oldest. Use `limit`
    to control how many states are returned and `before` to paginate.
    """
    parsed_metadata: dict[str, Any] | None = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValueError("metadata must be a JSON object")
        except Exception as e:
            raise HTTPException(422, f"Invalid metadata query param: {e}") from e
    req = ThreadHistoryRequest(
        limit=limit,
        before=before,
        metadata=parsed_metadata,
        checkpoint=None,
        subgraphs=subgraphs,
        checkpoint_ns=checkpoint_ns,
    )
    return await get_thread_history_post(thread_id, req, user, session)


@router.delete("/threads/{thread_id}", responses={**NOT_FOUND})
async def delete_thread(
    thread_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Delete a thread by its ID.

    Permanently removes the thread and its metadata. Any active runs on the
    thread are automatically cancelled before deletion. Checkpoint history
    stored in the graph backend is not affected.
    """
    # Authorization check
    ctx = build_auth_context(user, "threads", "delete")
    value = {"thread_id": thread_id}
    await handle_event(ctx, value)

    stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    thread = await session.scalar(stmt)
    if not thread:
        raise HTTPException(404, f"Thread '{thread_id}' not found")

    active_runs_stmt = select(RunORM).where(
        RunORM.thread_id == thread_id,
        RunORM.user_id == user.identity,
        RunORM.status.in_(["pending", "running"]),
    )
    active_runs_list = (await session.scalars(active_runs_stmt)).all()

    if active_runs_list:
        logger.info(f"Cancelling {len(active_runs_list)} active runs for thread {thread_id}")
        for run in active_runs_list:
            run_id = run.run_id
            await streaming_service.cancel_run(run_id)
            task = active_runs.pop(run_id, None)
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    await session.delete(thread)
    await session.commit()

    return {"status": "deleted"}


def _wants_state(request: ThreadSearchRequest) -> bool:
    """Whether the search needs thread_state joined (values filter / projection)."""
    if request.values or request.extract:
        return True
    return bool(request.select) and bool({"values", "interrupts"} & set(request.select))


def _search_filters(request: ThreadSearchRequest, user: User) -> list[Any]:
    """Shared WHERE predicates for /threads/search and /threads/count.

    The ``values`` filter targets ``thread_state`` (the caller joins it when
    ``_wants_state`` is true).
    """
    where: list[Any] = [ThreadORM.user_id == user.identity]
    if request.status:
        where.append(ThreadORM.status == request.status)
    if request.metadata:
        # JSONB containment: type-correct, deep-nested, GIN-indexable. Mirrors
        # AssistantService.search_assistants for cross-endpoint consistency.
        where.append(ThreadORM.metadata_json.op("@>")(request.metadata))
    if request.values:
        where.append(ThreadStateORM.values.op("@>")(request.values))
    if request.ids:
        where.append(ThreadORM.thread_id.in_(request.ids))
    return where


# response_model=None: with `select`/`extract` items become partial dicts.
@router.post("/threads/search", response_model=None)
async def search_threads(
    request: ThreadSearchRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[Thread] | list[dict[str, Any]]:
    """Search threads with filters.

    Filter by status, metadata, latest state values, or explicit ids. Results
    are paginated via `limit`/`offset`; `select` projects fields and `extract`
    adds keys pulled from values/metadata via dot/bracket paths.
    """
    # Authorization check
    ctx = build_auth_context(user, "threads", "search")
    value = request.model_dump()
    filters = await handle_event(ctx, value)

    # Merge handler filters with request metadata
    # Note: ThreadSearchRequest doesn't have a filters field,
    # so we merge authorization filters into metadata if needed
    if filters and "metadata" in filters:
        # If filters contain metadata, merge with request metadata
        handler_meta = filters["metadata"]
        if isinstance(handler_meta, dict):
            request.metadata = {**(request.metadata or {}), **handler_meta}
        # Other filter types can be handled here if needed
    extract = validate_extract(request.extract) if request.extract else None

    # Only join thread_state when the query actually needs values (filter/projection);
    # plain list/search scans the narrow thread table without the large state blob.
    need_state = _wants_state(request)
    if need_state:
        stmt = (
            select(ThreadORM, ThreadStateORM)
            .outerjoin(ThreadStateORM, ThreadStateORM.thread_id == ThreadORM.thread_id)
            .where(*_search_filters(request, user))
        )
    else:
        stmt = select(ThreadORM).where(*_search_filters(request, user))
    offset = request.offset or 0
    limit = request.limit or 20
    column, asc = _resolve_sort(request)
    direction = column.asc() if asc else column.desc()
    # Secondary sort on thread_id keeps offset pagination stable when the
    # primary sort key has duplicates (status buckets, microsecond ties).
    stmt = stmt.order_by(direction, ThreadORM.thread_id.asc()).offset(offset).limit(limit)

    if need_state:
        threads_models = [_serialize_thread(t, state=s) for t, s in (await session.execute(stmt)).all()]
    else:
        threads_models = [_serialize_thread(t) for t in (await session.scalars(stmt)).all()]

    if not request.select and not extract:
        return threads_models

    wanted = set(request.select) if request.select else None
    projected: list[dict[str, Any]] = []
    for model in threads_models:
        data = model.model_dump(mode="json")
        row = {k: v for k, v in data.items() if k in wanted} if wanted else dict(data)
        if extract:
            for alias, path in extract.items():
                row[alias] = extract_path_value(data, path)
        projected.append(row)
    return projected


@router.post("/threads/count", response_model=int)
async def count_threads(
    request: ThreadSearchRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> int:
    """Count threads matching the given filters.

    Accepts the same filters as `/threads/search` (status, metadata, values,
    ids) but returns only the total count.
    """
    ctx = build_auth_context(user, "threads", "search")
    value = request.model_dump()
    filters = await handle_event(ctx, value)
    if filters and "metadata" in filters:
        handler_meta = filters["metadata"]
        if isinstance(handler_meta, dict):
            request.metadata = {**(request.metadata or {}), **handler_meta}

    stmt = select(func.count()).select_from(ThreadORM)
    if request.values:  # values filter targets thread_state
        stmt = stmt.join(ThreadStateORM, ThreadStateORM.thread_id == ThreadORM.thread_id)
    stmt = stmt.where(*_search_filters(request, user))
    return await session.scalar(stmt) or 0


@router.post("/threads/prune")
async def prune_threads(
    request: ThreadPruneRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    """Prune threads by id.

    ``delete`` removes each thread entirely (runs, checkpoints, row);
    ``keep_latest`` keeps the thread but drops all but the newest checkpoint
    per namespace. Missing/unowned threads are skipped, not errors.
    """
    ctx = build_auth_context(user, "threads", "delete")
    await handle_event(ctx, {"thread_ids": request.thread_ids, "strategy": request.strategy})

    if not request.thread_ids:
        return {"pruned_count": 0}

    owned = list(
        (
            await session.scalars(
                select(ThreadORM.thread_id).where(
                    ThreadORM.thread_id.in_(request.thread_ids),
                    ThreadORM.user_id == user.identity,
                )
            )
        ).all()
    )
    if request.strategy == "keep_latest":
        await db_manager.get_checkpointer().aprune_keep_latest(owned)
        return {"pruned_count": len(owned)}

    pruned = 0
    for thread_id in owned:
        try:
            await delete_thread_by_id(thread_id, user.identity)
            pruned += 1
        except HTTPException as exc:
            # Reference behavior: skip silently, count only successes.
            logger.debug("Prune skipped thread", thread_id=thread_id, detail=exc.detail)
    return {"pruned_count": pruned}


@router.post("/threads/{thread_id}/copy", response_model=Thread, responses={**NOT_FOUND})
async def copy_thread(
    thread_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Thread:
    """Copy a thread and its full checkpoint history into a new thread.

    Creates a new idle thread owned by the caller, duplicating the source
    thread's metadata and every checkpoint. The original is unchanged. Cost is
    O(checkpoint history) — avoid on very large threads.
    """
    ctx = build_auth_context(user, "threads", "create")
    await handle_event(ctx, {"thread_id": thread_id})

    src = await session.scalar(
        select(ThreadORM).where(ThreadORM.thread_id == thread_id, ThreadORM.user_id == user.identity)
    )
    if not src:
        raise HTTPException(404, f"Thread '{thread_id}' not found")

    new_thread_id = str(uuid4())
    new_thread = ThreadORM(
        thread_id=new_thread_id,
        status="idle",
        metadata_json=dict(src.metadata_json or {}),
        user_id=user.identity,
    )
    session.add(new_thread)
    # Carry the materialized state over too (checkpointer rows copied below).
    src_state = await session.scalar(select(ThreadStateORM).where(ThreadStateORM.thread_id == thread_id))
    if src_state is not None:
        session.add(
            ThreadStateORM(
                thread_id=new_thread_id,
                values=src_state.values,
                interrupts=src_state.interrupts,
                values_hash=src_state.values_hash,
            )
        )
    await session.commit()

    await _copy_thread_checkpoints(thread_id, new_thread_id)

    return _serialize_thread(new_thread)


# Explicit column lists — the saver has no copy API, so we duplicate its rows
# directly (same coupling as adelete_thread) with the thread_id remapped.
_COPY_STATEMENTS = (
    "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata) "
    "SELECT %s, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata "
    "FROM checkpoints WHERE thread_id = %s",
    "INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob) "
    "SELECT %s, checkpoint_ns, channel, version, type, blob "
    "FROM checkpoint_blobs WHERE thread_id = %s",
    "INSERT INTO checkpoint_writes "
    "(thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob) "
    "SELECT %s, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob "
    "FROM checkpoint_writes WHERE thread_id = %s",
)


async def _copy_thread_checkpoints(src_thread_id: str, dst_thread_id: str) -> None:
    """Duplicate all checkpoint rows from src to dst thread."""
    pool = db_manager.lg_pool
    if pool is None:
        raise HTTPException(503, "Checkpoint store not initialized")
    # One transaction so a partial failure never leaves half-copied checkpoints,
    # regardless of the pool's autocommit setting.
    async with pool.connection() as conn, conn.transaction():
        for stmt in _COPY_STATEMENTS:
            await conn.execute(stmt, (dst_thread_id, src_thread_id))
