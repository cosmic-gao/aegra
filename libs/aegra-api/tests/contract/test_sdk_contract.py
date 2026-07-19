"""SDK contract tests — lock Aegra's wire contract to what ``langgraph-sdk`` expects.

Source of truth is the pinned ``langgraph-sdk`` (the client Aegra advertises as a
drop-in target). These tests run offline (no server, no DB) so they gate every PR
in the unit job: a response model that drops a key the SDK requires, or an enum
that drifts from the SDK's literal set, fails here. Bumping ``langgraph-sdk``
requires re-checking this file against the new schema.
"""

from __future__ import annotations

from typing import get_args

import pytest
import typing_extensions as te
from langgraph_sdk import schema as sdk
from pydantic import BaseModel

from aegra_api.models import enums
from aegra_api.models.assistants import AgentSchemas, Assistant
from aegra_api.models.crons import CronResponse
from aegra_api.models.runs import Run
from aegra_api.models.store import StoreGetResponse, StoreItem
from aegra_api.models.threads import Thread, ThreadState

pytestmark = pytest.mark.contract


def _wire_keys(model: type[BaseModel]) -> set[str]:
    """Keys the model serializes on the wire.

    Routes dump with ``by_alias=False``, so the wire key is the field name, not
    its alias (e.g. Assistant emits ``metadata``, not ``metadata_dict``).
    """
    return set(model.model_fields.keys())


def _sdk_required_keys(td: type) -> set[str]:
    """Keys the SDK requires the server to always emit.

    ``__required_keys__`` is unreliable here: langgraph-sdk stringizes its
    annotations (``from __future__ import annotations``), under which TypedDict
    miscounts ``NotRequired`` keys as required. Resolve the hints and drop them.
    """
    hints = te.get_type_hints(td, include_extras=True)
    return {key for key, hint in hints.items() if te.get_origin(hint) is not te.NotRequired}


# (Aegra response model, SDK TypedDict): Aegra must emit every SDK-required key.
# Store items pair with ``Item`` (the required shape); the search-only ``score``
# is optional in the SDK and asserted separately in the store unit tests.
_MODEL_PAIRS: list[tuple[type[BaseModel], type]] = [
    (Run, sdk.Run),
    (Thread, sdk.Thread),
    (ThreadState, sdk.ThreadState),
    (StoreItem, sdk.Item),
    (StoreGetResponse, sdk.Item),
    (AgentSchemas, sdk.GraphSchema),
    (CronResponse, sdk.Cron),
    (Assistant, sdk.Assistant),
]
_MODEL_IDS = [f"{model.__name__}->{sdk_type.__name__}" for model, sdk_type in _MODEL_PAIRS]


@pytest.mark.parametrize("model, sdk_type", _MODEL_PAIRS, ids=_MODEL_IDS)
def test_response_model_covers_sdk_required_keys(model: type[BaseModel], sdk_type: type) -> None:
    missing = _sdk_required_keys(sdk_type) - _wire_keys(model)
    assert not missing, f"{model.__name__} misses SDK-required keys {sorted(missing)} of {sdk_type.__name__}"


# (name, Aegra enum, SDK Literal): value sets must match exactly.
_ENUM_PAIRS: list[tuple[str, object, object]] = [
    ("RunStatus", enums.RunStatus, sdk.RunStatus),
    ("ThreadStatus", enums.ThreadStatus, sdk.ThreadStatus),
    ("MultitaskStrategy", enums.MultitaskStrategy, sdk.MultitaskStrategy),
    ("StreamMode", enums.StreamMode, sdk.StreamMode),
    ("DisconnectMode", enums.DisconnectMode, sdk.DisconnectMode),
    ("OnCompletionBehavior", enums.OnCompletionBehavior, sdk.OnCompletionBehavior),
    ("Durability", enums.Durability, sdk.Durability),
    ("OnConflictBehavior", enums.OnConflictBehavior, sdk.OnConflictBehavior),
    ("IfNotExists", enums.IfNotExists, sdk.IfNotExists),
    ("CancelAction", enums.CancelAction, sdk.CancelAction),
    ("PruneStrategy", enums.PruneStrategy, sdk.PruneStrategy),
    ("BulkCancelRunsStatus", enums.BulkCancelRunsStatus, sdk.BulkCancelRunsStatus),
]


@pytest.mark.parametrize("name, aegra_enum, sdk_enum", _ENUM_PAIRS, ids=[pair[0] for pair in _ENUM_PAIRS])
def test_enum_matches_sdk(name: str, aegra_enum: object, sdk_enum: object) -> None:
    assert set(get_args(aegra_enum)) == set(get_args(sdk_enum)), f"{name} enum drifted from langgraph-sdk"
