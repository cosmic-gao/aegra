"""Translate raw langgraph stream events into Agent Protocol v2 channel events.

The broker holds raw ``(mode, payload)`` tuples (the same ones the legacy
SSE path consumes). This turns them into v2 channel-event ``params`` dicts.

Message streaming is the involved case: langgraph emits a sequence of
``(AIMessageChunk, metadata)`` tuples per assistant message. We project
that into the protocol's content-block lifecycle —
``message-start`` → ``content-block-delta`` (one per token) →
``message-finish`` — keyed by message id, so the wire output is the same
shape the LangGraph JS/Python SDKs consume regardless of how the model
chunks its tokens. Text and reasoning are separate content blocks, so a
thinking model's reasoning streams as ``reasoning-delta`` on its own index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, BaseMessageChunk

# Stable content-block indices per message: text and reasoning are distinct
# blocks, so a client renders the chain-of-thought separately from the answer.
_TEXT_BLOCK_INDEX = 0
_REASONING_BLOCK_INDEX = 1

# (delta type, content-block index) for each kind of message delta.
_DELTA_KINDS: dict[str, tuple[str, int]] = {
    "text": ("text-delta", _TEXT_BLOCK_INDEX),
    "reasoning": ("reasoning-delta", _REASONING_BLOCK_INDEX),
}


def _message_deltas(message: Any) -> list[tuple[str, str]]:
    """Ordered ``(kind, text)`` deltas from a chunk, where kind is text/reasoning.

    Pulls text and reasoning out of the chunk's content (str or block list)
    and the ``reasoning_content`` additional_kwarg some providers use.
    """
    deltas: list[tuple[str, str]] = []
    content = getattr(message, "content", "")
    if isinstance(content, str):
        if content:
            deltas.append(("text", content))
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "reasoning" and (text := block.get("reasoning")):
                deltas.append(("reasoning", text))
            elif text := block.get("text"):
                deltas.append(("text", text))

    extra = getattr(message, "additional_kwargs", None)
    if isinstance(extra, dict) and (reasoning := extra.get("reasoning_content")):
        deltas.append(("reasoning", reasoning))

    return deltas


def _message_role(message: Any) -> str:
    """Protocol role for a message chunk (ai/human/system; default ai)."""
    msg_type = getattr(message, "type", "") or ""
    if msg_type.startswith("human"):
        return "human"
    if msg_type.startswith("system"):
        return "system"
    return "ai"


@dataclass
class _MessageState:
    """Per-message accumulation: whether we've opened it and the block index."""

    started: bool = False


@dataclass
class EventTranslator:
    """Stateful translator for one run's raw events.

    Holds per-message state so a token stream becomes a correct
    start/delta/finish sequence. One translator per session.
    """

    _messages: dict[str, _MessageState] = field(default_factory=dict)

    def translate(self, mode: str, payload: Any) -> list[tuple[str, dict[str, Any], list[str]]]:
        """Map one raw ``(mode, payload)`` event to zero or more channel events.

        Returns ``(channel, data, namespace)`` triples: ``channel`` is the
        protocol event method, ``data`` the per-channel payload (becomes
        ``params.data``), ``namespace`` the subgraph path (``[]`` at root).
        Unhandled modes (``metadata``, ``debug``, ``end``, ``error``) return
        nothing — the session handles lifecycle separately.
        """
        if mode == "messages":
            return self._translate_message(payload)
        if mode == "values":
            return [("values", payload if isinstance(payload, dict) else {"value": payload}, [])]
        if mode == "updates":
            return self._translate_updates(payload)
        if mode == "tools":
            return [("tools", payload if isinstance(payload, dict) else {"payload": payload}, [])]
        if mode == "custom":
            return [("custom", {"payload": payload}, [])]
        if mode == "tasks":
            return [("tasks", payload if isinstance(payload, dict) else {"payload": payload}, [])]
        if mode == "checkpoints":
            return [("checkpoints", payload if isinstance(payload, dict) else {"payload": payload}, [])]
        return []

    def _translate_message(self, payload: Any) -> list[tuple[str, dict[str, Any], list[str]]]:
        """Project an ``(AIMessageChunk, metadata)`` tuple into message events."""
        if not (isinstance(payload, (tuple, list)) and len(payload) == 2):
            return []
        message, metadata = payload
        if not isinstance(message, BaseMessage):
            return []

        msg_id = message.id or ""
        if not msg_id:
            return []

        events: list[tuple[str, dict[str, Any], list[str]]] = []
        state = self._messages.get(msg_id)
        if state is None:
            state = _MessageState(started=True)
            self._messages[msg_id] = state
            events.append(
                (
                    "messages",
                    {
                        "event": "message-start",
                        "role": _message_role(message),
                        "id": msg_id,
                        "metadata": _wire_metadata(metadata),
                    },
                    [],
                )
            )

        for kind, text in _message_deltas(message):
            delta_type, index = _DELTA_KINDS[kind]
            events.append(
                (
                    "messages",
                    {
                        "event": "content-block-delta",
                        "index": index,
                        "delta": {"type": delta_type, kind: text},
                    },
                    [],
                )
            )

        if _is_final_chunk(message):
            events.append(("messages", {"event": "message-finish"}, []))
            del self._messages[msg_id]

        return events

    def _translate_updates(self, payload: Any) -> list[tuple[str, dict[str, Any], list[str]]]:
        """Map an updates chunk (``{node: values}``) to one event per node."""
        if not isinstance(payload, dict):
            return [("updates", {"values": payload}, [])]
        return [("updates", {"node": node, "values": values}, []) for node, values in payload.items()]


def _is_final_chunk(message: Any) -> bool:
    """True when a message is terminal: the last stream chunk, or a complete
    (non-chunk) message that arrives whole from a non-streaming model."""
    if getattr(message, "chunk_position", None) == "last":
        return True
    return not isinstance(message, BaseMessageChunk)


def _wire_metadata(metadata: Any) -> dict[str, Any]:
    """Subset of langgraph chunk metadata mapped to protocol MessageMetadata."""
    if not isinstance(metadata, dict):
        return {}
    out: dict[str, Any] = {}
    if model := metadata.get("ls_model_name"):
        out["model"] = model
    if provider := metadata.get("ls_provider"):
        out["provider"] = provider
    return out
