"""Tests for raw langgraph event → v2 channel event translation."""

from langchain_core.messages import AIMessageChunk, HumanMessage

from aegra_api.services.event_streaming.translator import EventTranslator


def _chunk(text: str, *, msg_id: str = "m1", last: bool = False) -> AIMessageChunk:
    chunk = AIMessageChunk(content=text, id=msg_id)
    if last:
        chunk.chunk_position = "last"
    return chunk


class TestMessageTranslation:
    def test_first_chunk_emits_start_then_delta(self) -> None:
        t = EventTranslator()
        events = t.translate("messages", (_chunk("hello"), {"ls_model_name": "gpt", "ls_provider": "openai"}))
        assert [e[1]["event"] for e in events] == ["message-start", "content-block-delta"]
        start = events[0][1]
        assert start == {
            "event": "message-start",
            "role": "ai",
            "id": "m1",
            "metadata": {"model": "gpt", "provider": "openai"},
        }
        delta = events[1][1]
        assert delta == {"event": "content-block-delta", "index": 0, "delta": {"type": "text-delta", "text": "hello"}}

    def test_subsequent_chunk_emits_only_delta(self) -> None:
        t = EventTranslator()
        t.translate("messages", (_chunk("hello"), {}))
        events = t.translate("messages", (_chunk(" world"), {}))
        assert [e[1]["event"] for e in events] == ["content-block-delta"]
        assert events[0][1]["delta"]["text"] == " world"

    def test_final_chunk_emits_finish(self) -> None:
        t = EventTranslator()
        t.translate("messages", (_chunk("hi"), {}))
        events = t.translate("messages", (_chunk("!", last=True), {}))
        assert [e[1]["event"] for e in events] == ["content-block-delta", "message-finish"]

    def test_empty_text_chunk_skips_delta(self) -> None:
        t = EventTranslator()
        t.translate("messages", (_chunk("first"), {}))
        events = t.translate("messages", (_chunk("", last=True), {}))
        assert [e[1]["event"] for e in events] == ["message-finish"]

    def test_complete_message_emits_start_delta_finish(self) -> None:
        """A whole (non-chunk) message from a non-streaming model finishes immediately."""
        from langchain_core.messages import AIMessage

        t = EventTranslator()
        events = t.translate("messages", (AIMessage(content="done", id="c1"), {}))
        assert [e[1]["event"] for e in events] == ["message-start", "content-block-delta", "message-finish"]

    def test_message_state_is_cleared_after_finish(self) -> None:
        """Per-message state does not accumulate across finished messages."""
        t = EventTranslator()
        t.translate("messages", (_chunk("a", msg_id="x", last=True), {}))
        assert "x" not in t._messages

    def test_human_message_role(self) -> None:
        t = EventTranslator()
        msg = HumanMessage(content="hey", id="h1")
        events = t.translate("messages", (msg, {}))
        assert events[0][1]["role"] == "human"

    def test_channel_is_messages(self) -> None:
        t = EventTranslator()
        events = t.translate("messages", (_chunk("x"), {}))
        assert all(triple[0] == "messages" for triple in events)

    def test_message_without_id_is_skipped(self) -> None:
        t = EventTranslator()
        assert t.translate("messages", (AIMessageChunk(content="x", id=None), {})) == []

    def test_malformed_message_payload_skipped(self) -> None:
        t = EventTranslator()
        assert t.translate("messages", "not a tuple") == []


class TestOtherChannels:
    def test_values_carries_state_dict_directly(self) -> None:
        t = EventTranslator()
        assert t.translate("values", {"count": 1}) == [("values", {"count": 1}, [])]

    def test_updates_one_event_per_node(self) -> None:
        t = EventTranslator()
        events = t.translate("updates", {"node_a": {"x": 1}, "node_b": {"y": 2}})
        assert ("updates", {"node": "node_a", "values": {"x": 1}}, []) in events
        assert ("updates", {"node": "node_b", "values": {"y": 2}}, []) in events

    def test_custom(self) -> None:
        t = EventTranslator()
        assert t.translate("custom", {"foo": "bar"}) == [("custom", {"payload": {"foo": "bar"}}, [])]

    def test_tools_passthrough(self) -> None:
        t = EventTranslator()
        assert t.translate("tools", {"tool_call_id": "c1"}) == [("tools", {"tool_call_id": "c1"}, [])]

    def test_checkpoints_passthrough(self) -> None:
        t = EventTranslator()
        assert t.translate("checkpoints", {"id": "ck1"}) == [("checkpoints", {"id": "ck1"}, [])]

    def test_unhandled_modes_return_nothing(self) -> None:
        t = EventTranslator()
        for mode in ("metadata", "debug", "end", "error"):
            assert t.translate(mode, {"anything": True}) == []


class TestReasoning:
    def test_reasoning_block_emits_reasoning_delta_on_its_own_index(self) -> None:
        """A reasoning content block streams as reasoning-delta at index 1."""
        t = EventTranslator()
        chunk = AIMessageChunk(content=[{"type": "reasoning", "reasoning": "thinking..."}], id="m1")
        events = t.translate("messages", (chunk, {}))
        deltas = [e[1] for e in events if e[1]["event"] == "content-block-delta"]
        assert deltas == [
            {
                "event": "content-block-delta",
                "index": 1,
                "delta": {"type": "reasoning-delta", "reasoning": "thinking..."},
            }
        ]

    def test_text_and_reasoning_are_separate_blocks(self) -> None:
        t = EventTranslator()
        chunk = AIMessageChunk(
            content=[{"type": "reasoning", "reasoning": "hmm"}, {"type": "text", "text": "answer"}],
            id="m1",
        )
        events = t.translate("messages", (chunk, {}))
        deltas = [(e[1]["index"], e[1]["delta"]["type"]) for e in events if e[1]["event"] == "content-block-delta"]
        assert (1, "reasoning-delta") in deltas
        assert (0, "text-delta") in deltas

    def test_reasoning_content_kwarg_is_picked_up(self) -> None:
        """Providers that put reasoning in additional_kwargs are handled too."""
        t = EventTranslator()
        chunk = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "deep thought"}, id="m1")
        events = t.translate("messages", (chunk, {}))
        deltas = [e[1]["delta"] for e in events if e[1]["event"] == "content-block-delta"]
        assert {"type": "reasoning-delta", "reasoning": "deep thought"} in deltas


class TestChannelTripleShape:
    def test_message_events_carry_empty_namespace(self) -> None:
        t = EventTranslator()
        events = t.translate("messages", (_chunk("x"), {}))
        assert all(len(triple) == 3 and triple[2] == [] for triple in events)
