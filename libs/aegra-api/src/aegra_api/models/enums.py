"""Status enums for Aegra API specification."""

from typing import Literal

# Run status enum
RunStatus = Literal[
    "pending",
    "running",
    "error",
    "success",
    "timeout",
    "interrupted",
]

# Thread status enum
ThreadStatus = Literal[
    "idle",
    "busy",
    "interrupted",
    "error",
]

# Single source of truth for terminal run states, so waiters/streaming/executor
# and finalize's compare-and-set can't drift (dropping "timeout" would hang /join).
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"success", "error", "interrupted", "timeout"})

# Multitask strategy enum
MultitaskStrategy = Literal[
    "reject",
    "rollback",
    "interrupt",
    "enqueue",
]

# The literals below mirror langgraph-sdk exactly and are the single source of
# truth for these value sets; models/routes reference them instead of inlining.
# tests/contract/test_sdk_contract.py fails if any drifts from the SDK.

# Stream mode enum (SDK StreamMode)
StreamMode = Literal[
    "values",
    "messages",
    "updates",
    "events",
    "tasks",
    "checkpoints",
    "debug",
    "custom",
    "messages-tuple",
]

# Behavior on client disconnect (SDK DisconnectMode)
DisconnectMode = Literal["cancel", "continue"]

# Behavior after a stateless run completes (SDK OnCompletionBehavior)
OnCompletionBehavior = Literal["delete", "keep"]

# Checkpoint durability (SDK Durability)
Durability = Literal["sync", "async", "exit"]

# Behavior when a resource already exists (SDK OnConflictBehavior; the if_exists param)
OnConflictBehavior = Literal["raise", "do_nothing"]

# Behavior when the target thread is missing (SDK IfNotExists)
IfNotExists = Literal["create", "reject"]

# Run cancel action (SDK CancelAction)
CancelAction = Literal["interrupt", "rollback"]

# Thread prune strategy (SDK PruneStrategy)
PruneStrategy = Literal["delete", "keep_latest"]

# Status filter for bulk run cancel (SDK BulkCancelRunsStatus)
BulkCancelRunsStatus = Literal["pending", "running", "all"]
