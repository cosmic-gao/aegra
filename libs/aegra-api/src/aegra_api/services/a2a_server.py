"""A2A server exposing this deployment's assistants over JSON-RPC.

Mirrors LangGraph Platform's A2A surface: a per-assistant JSON-RPC endpoint at
``/a2a/{assistant_id}`` plus agent-card discovery routes. Built on the official
``a2a-sdk`` (``JsonRpcDispatcher``, ``DefaultRequestHandler``, ``InMemoryTaskStore``)
with v0.3 compatibility enabled so platform-era clients (``message/send``) work.

Every request authenticates through the same auth backend as the REST API
(``require_auth``), and graphs execute under the caller's identity. Routes are
returned by :func:`a2a_routes` for the app factory to append to the router.
"""

import functools
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import structlog
from a2a.auth.user import UnauthenticatedUser
from a2a.auth.user import User as A2AUser
from a2a.helpers import new_data_part, new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.server.routes.jsonrpc_dispatcher import JsonRpcDispatcher
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    TaskState,
)
from a2a.utils.constants import (
    PROTOCOL_VERSION_0_3,
    PROTOCOL_VERSION_1_0,
    TransportProtocol,
)
from fastapi import HTTPException
from langchain_core.runnables import RunnableConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import BaseRoute, Route

from aegra_api import __version__
from aegra_api.core.auth_deps import require_auth
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.models.auth import User
from aegra_api.services.langgraph_service import get_langgraph_service

logger = structlog.getLogger(__name__)

_serializer = GeneralSerializer()

_A2A_PREFIX = "/a2a"
_TEXT_MODE = "text/plain"
_JSON_MODE = "application/json"

# One dispatcher (handler + task store) per assistant, created lazily and kept
# so tasks/get can find tasks created by earlier message/send calls.
_dispatchers: dict[str, JsonRpcDispatcher] = {}


class _A2AUser(A2AUser):
    """Adapts Aegra's authenticated user to the A2A SDK user interface."""

    def __init__(self, user: User) -> None:
        self._user = user

    @property
    def is_authenticated(self) -> bool:
        return self._user.is_authenticated

    @property
    def user_name(self) -> str:
        return self._user.identity


class _AegraCallContextBuilder(DefaultServerCallContextBuilder):
    """Builds SDK call contexts carrying the Aegra user and target graph."""

    def build(self, request: Request) -> ServerCallContext:
        context = super().build(request)
        user: User | None = getattr(request.state, "aegra_user", None)
        if user is not None:
            context.state["aegra_user"] = user
        assistant_id: str | None = request.path_params.get("assistant_id")
        if assistant_id:
            context.state["graph_id"] = assistant_id
        return context

    def build_user(self, request: Request) -> A2AUser:
        user: User | None = getattr(request.state, "aegra_user", None)
        if user is None:
            return UnauthenticatedUser()
        return _A2AUser(user)


_context_builder = _AegraCallContextBuilder()


def _final_text(result: Any) -> str | None:
    """Extract the last message's text content from a graph result, if any."""
    if not isinstance(result, dict):
        return None
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
    if isinstance(content, str) and content:
        return content
    return None


def _result_parts(result: Any) -> list[Part]:
    text = _final_text(result)
    if text is not None:
        return [new_text_part(text)]
    return [new_data_part(_serializer.serialize(result), media_type=_JSON_MODE)]


class AegraAgentExecutor(AgentExecutor):
    """Runs the target Aegra graph and publishes the result as an A2A task."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        state = context.call_context.state
        graph_id: str | None = state.get("graph_id")
        user: User | None = state.get("aegra_user")
        service = get_langgraph_service()
        if not graph_id or graph_id not in service.list_graphs():
            raise ValueError(f"Unknown assistant '{graph_id}'")

        task = context.current_task
        if task is None:
            history = [context.message] if context.message is not None else None
            task = new_task(
                task_id=context.task_id or str(uuid4()),
                context_id=context.context_id or str(uuid4()),
                state=TaskState.TASK_STATE_SUBMITTED,
                history=history,
            )
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        graph_input = {"messages": [{"role": "user", "content": context.get_user_input()}]}
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
        try:
            async with service.get_graph(graph_id, user=user) as graph:
                result = await graph.ainvoke(graph_input, config)
        except Exception as exc:  # graph code may raise anything; surface a failed task
            logger.exception("a2a_graph_execution_failed", assistant=graph_id)
            await updater.failed(message=updater.new_agent_message([new_text_part(str(exc))]))
            return

        await updater.add_artifact(_result_parts(result), name="result")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        task_id = context.task_id or (task.id if task else None)
        context_id = context.context_id or (task.context_id if task else None)
        if not task_id or not context_id:
            logger.warning("a2a_cancel_missing_task_ids")
            return
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()


def _interfaces(endpoint: str) -> list[AgentInterface]:
    """Advertise the endpoint for protocol 1.0 and 0.3 (legacy ``url`` field)."""
    return [
        AgentInterface(
            url=endpoint,
            protocol_binding=TransportProtocol.JSONRPC.value,
            protocol_version=PROTOCOL_VERSION_1_0,
        ),
        AgentInterface(
            url=endpoint,
            protocol_binding=TransportProtocol.JSONRPC.value,
            protocol_version=PROTOCOL_VERSION_0_3,
        ),
    ]


def _build_card(graph_id: str, base_url: str) -> AgentCard:
    endpoint = f"{base_url}{_A2A_PREFIX}/{graph_id}"
    return AgentCard(
        name=graph_id,
        description=f"LangGraph assistant '{graph_id}' served by Aegra over the A2A protocol.",
        version=__version__,
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=[_TEXT_MODE],
        default_output_modes=[_TEXT_MODE, _JSON_MODE],
        skills=[
            AgentSkill(
                id=f"{graph_id}-run",
                name=graph_id,
                description=f"Send a message to the '{graph_id}' assistant and receive its final response.",
                tags=["langgraph", "aegra"],
            )
        ],
        supported_interfaces=_interfaces(endpoint),
    )


def _build_root_card(graph_ids: list[str], base_url: str) -> AgentCard:
    return AgentCard(
        name="Aegra",
        description=(
            "Self-hosted Agent Protocol server. Each assistant has its own A2A "
            f"JSON-RPC endpoint at {_A2A_PREFIX}/{{assistant_id}}."
        ),
        version=__version__,
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=[_TEXT_MODE],
        default_output_modes=[_TEXT_MODE, _JSON_MODE],
        skills=[
            AgentSkill(
                id=graph_id,
                name=graph_id,
                description=f"Run the '{graph_id}' assistant via {_A2A_PREFIX}/{graph_id}.",
                tags=["langgraph", "aegra"],
            )
            for graph_id in graph_ids
        ],
        supported_interfaces=_interfaces(base_url or "/"),
    )


def _get_dispatcher(graph_id: str) -> JsonRpcDispatcher:
    dispatcher = _dispatchers.get(graph_id)
    if dispatcher is not None:
        return dispatcher
    handler = DefaultRequestHandler(
        agent_executor=AegraAgentExecutor(),
        task_store=InMemoryTaskStore(),
        # Path-only URL: the handler card only gates capabilities (streaming=False);
        # cards served over HTTP are built per-request with the absolute base URL.
        agent_card=_build_card(graph_id, base_url=""),
    )
    dispatcher = JsonRpcDispatcher(
        request_handler=handler,
        context_builder=_context_builder,
        enable_v0_3_compat=True,
    )
    _dispatchers[graph_id] = dispatcher
    return dispatcher


def _authed(
    endpoint: Callable[[Request], Awaitable[Response]],
) -> Callable[[Request], Awaitable[Response]]:
    """Resolve the caller via the REST auth backend before running the endpoint."""

    @functools.wraps(endpoint)
    async def wrapper(request: Request) -> Response:
        try:
            user = await require_auth(request)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        request.state.aegra_user = user
        return await endpoint(request)

    return wrapper


def _not_found(assistant_id: str) -> JSONResponse:
    return JSONResponse({"detail": f"Assistant '{assistant_id}' not found"}, status_code=404)


def _card_response(graph_id: str, request: Request) -> JSONResponse:
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse(agent_card_to_dict(_build_card(graph_id, base_url)))


@_authed
async def _rpc_endpoint(request: Request) -> Response:
    assistant_id: str = request.path_params["assistant_id"]
    if assistant_id not in get_langgraph_service().list_graphs():
        return _not_found(assistant_id)
    if request.method == "GET":
        return _card_response(assistant_id, request)
    if request.method == "DELETE":
        # A2A's JSON-RPC transport is stateless: cancellation/deletion happen via
        # JSON-RPC methods, so DELETE (accepted for platform parity) is a 405.
        return JSONResponse(
            {"detail": "Method Not Allowed"},
            status_code=405,
            headers={"Allow": "GET, POST"},
        )
    return await _get_dispatcher(assistant_id).handle_requests(request)


@_authed
async def _assistant_card_endpoint(request: Request) -> Response:
    assistant_id: str = request.path_params["assistant_id"]
    if assistant_id not in get_langgraph_service().list_graphs():
        return _not_found(assistant_id)
    return _card_response(assistant_id, request)


@_authed
async def _root_card_endpoint(request: Request) -> Response:
    base_url = str(request.base_url).rstrip("/")
    graph_ids = sorted(get_langgraph_service().list_graphs())
    return JSONResponse(agent_card_to_dict(_build_root_card(graph_ids, base_url)))


def a2a_routes() -> list[BaseRoute]:
    """Starlette routes for the A2A surface, ready to append to the app router."""
    return [
        Route(
            f"{_A2A_PREFIX}/{{assistant_id}}",
            _rpc_endpoint,
            methods=["GET", "POST", "DELETE"],
        ),
        Route(
            f"{_A2A_PREFIX}/{{assistant_id}}/.well-known/agent-card.json",
            _assistant_card_endpoint,
            methods=["GET"],
        ),
        Route(
            f"{_A2A_PREFIX}/{{assistant_id}}/.well-known/agent.json",
            _assistant_card_endpoint,
            methods=["GET"],
        ),
        Route("/.well-known/agent-card.json", _root_card_endpoint, methods=["GET"]),
    ]
