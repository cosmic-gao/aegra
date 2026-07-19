"""MCP server exposing this deployment's graphs as tools over Streamable HTTP.

Mounted at ``/mcp`` by ``main.create_app`` when ``MCP_ENABLED``. Mirrors the
LangGraph Platform MCP endpoint: **each graph is exposed as its own tool** whose
name is the graph id and whose input schema is the graph's input schema, rather
than a pair of generic ``list_assistants``/``run_assistant`` tools. Tools run
under the caller's identity (same auth backend as the REST API), each call is
stateless (a fresh thread per invocation), and ``@auth.on`` authorization
applies. A no-auth deployment resolves to the anonymous user.

FastMCP infers a tool's schema from a Python signature, so to advertise each
graph's own JSON schema we register ``list_tools``/``call_tool`` handlers on the
underlying low-level server (``mcp._mcp_server``) while keeping FastMCP's
Streamable HTTP transport (``streamable_http_app``).
"""

import asyncio
import json
from typing import Any
from uuid import uuid4

import structlog
from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from mcp import types
from mcp.server.fastmcp import FastMCP

from aegra_api.core.auth_deps import require_auth
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.models.auth import User
from aegra_api.services.langgraph_service import get_langgraph_service

logger = structlog.getLogger(__name__)

_serializer = GeneralSerializer()

mcp = FastMCP("aegra", stateless_http=True)

# MCP requires an object input schema; used when a graph can't produce one.
_PERMISSIVE_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}

# Deriving a graph's input schema recompiles factory graphs, so cache per graph
# and invalidate on hot-reload (langgraph_service.invalidate_cache calls us).
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def invalidate_schema_cache(graph_id: str | None = None) -> None:
    """Drop cached MCP input schemas for a graph, or all when ``graph_id`` is None."""
    if graph_id is None:
        _SCHEMA_CACHE.clear()
    else:
        _SCHEMA_CACHE.pop(graph_id, None)


async def _authenticate() -> User:
    """Resolve the caller via the REST auth backend, or raise (unauthorized)."""
    try:
        request = getattr(mcp._mcp_server.request_context, "request", None)
    except LookupError:
        request = None
    if request is None:
        raise ValueError("MCP request has no HTTP context to authenticate")
    try:
        return await require_auth(request)
    except HTTPException as exc:
        raise ValueError(f"Unauthorized: {exc.detail}") from exc


async def _input_schema(graph_id: str, user: User) -> dict[str, Any]:
    """The graph's input JSON schema (cached), or a permissive object schema on failure."""
    cached = _SCHEMA_CACHE.get(graph_id)
    if cached is not None:
        return cached
    try:
        graph = await get_langgraph_service().get_graph_for_validation(graph_id, user=user)
        schema = graph.get_input_jsonschema()
    except Exception as exc:
        logger.warning("Could not derive MCP input schema", graph_id=graph_id, error=str(exc))
        return dict(_PERMISSIVE_SCHEMA)  # transient — don't cache the fallback
    result = schema if isinstance(schema, dict) and schema.get("type") == "object" else dict(_PERMISSIVE_SCHEMA)
    _SCHEMA_CACHE[graph_id] = result
    return result


@mcp._mcp_server.list_tools()
async def _list_tools() -> list[types.Tool]:
    """Expose each graph as its own tool (name = graph id, schema = input schema)."""
    user = await _authenticate()
    graph_ids = list(get_langgraph_service().list_graphs())
    schemas = await asyncio.gather(*(_input_schema(graph_id, user) for graph_id in graph_ids))
    return [
        types.Tool(
            name=graph_id,
            description=f"Run the '{graph_id}' agent and return its final state.",
            inputSchema=schema,
        )
        for graph_id, schema in zip(graph_ids, schemas, strict=True)
    ]


@mcp._mcp_server.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Run the named graph with the given input; return its serialized final state."""
    user = await _authenticate()
    service = get_langgraph_service()
    if name not in service.list_graphs():
        raise ValueError(f"Unknown assistant '{name}'")
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    async with service.get_graph(name, user=user) as graph:
        result = await graph.ainvoke(arguments, config)
    payload = _serializer.serialize(result)
    return [types.TextContent(type="text", text=json.dumps(payload, default=str))]


# Built at import so the session manager is initialized before create_app mounts
# it; main's lifespan runs ``mcp.session_manager.run()``.
mcp_app = mcp.streamable_http_app()
