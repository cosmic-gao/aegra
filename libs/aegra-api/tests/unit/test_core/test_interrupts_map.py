"""Unit tests for LangGraphSerializer.build_interrupts_map.

The map keys interrupts by task id to match the SDK ``Thread.interrupts`` shape
(``dict[str, list[Interrupt]]``).
"""

import pytest

from aegra_api.core.serializers.langgraph import LangGraphSerializer
from tests.fixtures.langgraph import make_interrupt, make_snapshot, make_task

pytestmark = pytest.mark.unit


def _snapshot(tasks):
    return make_snapshot({"messages": []}, {"configurable": {"thread_id": "t1"}}, tasks=tasks)


def test_groups_interrupts_by_task_id():
    task = make_task(id="task-A", interrupts=(make_interrupt(value="hi", interrupt_id="int-1"),))
    result = LangGraphSerializer().build_interrupts_map(_snapshot([task]))
    assert set(result) == {"task-A"}
    assert result["task-A"] == [{"value": "hi", "id": "int-1"}]


def test_multiple_tasks_keys_do_not_collapse():
    tasks = [
        make_task(id="task-A", interrupts=(make_interrupt(interrupt_id="a"),)),
        make_task(id="task-B", interrupts=(make_interrupt(interrupt_id="b"),)),
    ]
    result = LangGraphSerializer().build_interrupts_map(_snapshot(tasks))
    assert set(result) == {"task-A", "task-B"}


def test_task_without_interrupts_skipped():
    tasks = [
        make_task(id="task-A", interrupts=()),
        make_task(id="task-B", interrupts=(make_interrupt(),)),
    ]
    result = LangGraphSerializer().build_interrupts_map(_snapshot(tasks))
    assert set(result) == {"task-B"}


def test_no_tasks_returns_empty_dict():
    assert LangGraphSerializer().build_interrupts_map(_snapshot(None)) == {}
