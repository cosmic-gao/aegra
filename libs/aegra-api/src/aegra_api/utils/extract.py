"""Dot/bracket path extraction for thread search ``extract``.

Mirrors the LangGraph Platform semantics: paths like
``values.messages[-1].content`` are resolved against the thread's
materialized columns and returned as extra response keys.
"""

import re
from typing import Any

from fastapi import HTTPException

# Columns that support extraction path syntax.
EXTRACTABLE_COLUMNS = frozenset({"values", "metadata", "config", "interrupts"})

# Reserved thread field names that extract aliases must not shadow.
THREAD_FIELDS = frozenset(
    {"thread_id", "created_at", "updated_at", "metadata", "config", "context", "status", "values", "interrupts"}
)

MAX_EXTRACT_PATHS = 10

_ALIAS_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(-?\d+)\]")


def validate_extract(extract: dict[str, str]) -> dict[str, str]:
    """Validate the ``extract`` parameter; raises 422 on any violation."""
    if len(extract) > MAX_EXTRACT_PATHS:
        raise HTTPException(422, f"Maximum of {MAX_EXTRACT_PATHS} extract paths allowed, got {len(extract)}")
    reserved = extract.keys() & THREAD_FIELDS
    if reserved:
        raise HTTPException(422, f"Extract keys cannot use reserved field names: {sorted(reserved)}")
    for alias, path in extract.items():
        if not _ALIAS_RE.match(alias):
            raise HTTPException(422, f"Extract key '{alias}' must be a valid identifier")
        if not path or not isinstance(path, str):
            raise HTTPException(422, f"Extract path for '{alias}' must be a non-empty string")
        root = path.split(".")[0].split("[")[0]
        if root not in EXTRACTABLE_COLUMNS:
            raise HTTPException(
                422, f"Extract path '{path}' must start with one of: {', '.join(sorted(EXTRACTABLE_COLUMNS))}"
            )
    return extract


def extract_path_value(data: dict[str, Any], path: str) -> Any:
    """Resolve ``column.key1[idx].key2`` against a thread dict; None when missing."""
    current: Any = data
    for match in _TOKEN_RE.finditer(path):
        key, index = match.group(1), match.group(2)
        try:
            if index is not None:
                if not isinstance(current, (list, tuple)):
                    return None
                current = current[int(index)]
            elif isinstance(current, dict):
                current = current[key]
            else:
                current = getattr(current, key)
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
        if current is None:
            return None
    return current
