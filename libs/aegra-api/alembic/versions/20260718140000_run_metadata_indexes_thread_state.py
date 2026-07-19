"""Runtime-alignment + perf schema changes (single consolidated migration).

Three coupled changes in one migration, so a partial apply can't leave the ORM
and schema disagreeing:
1. ``runs`` gains ``metadata`` (JSONB) + ``multitask_strategy`` (TEXT) — the SDK
   ``Run`` contract needs both as real, selectable columns.
2. Composite + GIN indexes for the list/search hot paths (``thread``, ``runs``,
   ``crons`` metadata, ``thread_state`` values).
3. ``thread_state`` (1:1 with ``thread``): the latest materialized state moves
   out of the wide ``thread`` row into a narrow table (backfilled, then the old
   ``thread.values``/``interrupts`` columns are dropped). Checkpointer stays the
   source of truth.

Defensive (``IF EXISTS`` / ``IF NOT EXISTS`` + a column guard on the backfill)
so it is idempotent and resumable.

Revision ID: e5b9d2f7a3c1
Revises: b88bb61be638
Create Date: 2026-07-18 14:00:00.000000
"""

import sqlalchemy as sa

from alembic import op

revision = "e5b9d2f7a3c1"
down_revision = "b88bb61be638"
branch_labels = None
depends_on = None

_BACKFILL = sa.text(
    """
    INSERT INTO thread_state (thread_id, "values", interrupts, values_hash, updated_at)
    -- values_hash left NULL: SQL md5(jsonb::text) can't match the app's
    -- md5(json.dumps(sort_keys=True)), so let the first materialize recompute it.
    SELECT t.thread_id, t."values", t.interrupts, NULL, t.updated_at
    FROM thread t
    WHERE (t."values" IS NOT NULL OR t.interrupts IS NOT NULL)
      AND NOT EXISTS (SELECT 1 FROM thread_state ts WHERE ts.thread_id = t.thread_id)
    LIMIT 1000
    """
)

_INDEX_CREATE = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thread_user_created ON thread (user_id, created_at DESC)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_runs_thread_created ON runs (thread_id, created_at DESC)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cron_metadata_gin ON crons USING gin (metadata jsonb_path_ops)",
    'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thread_state_values_gin '
    'ON thread_state USING gin ("values" jsonb_path_ops)',
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_runs_scheduled ON runs (status, scheduled_at)",
)
_INDEX_NAMES = (
    "idx_thread_user_created",
    "idx_runs_thread_created",
    "idx_cron_metadata_gin",
    "idx_thread_state_values_gin",
    "idx_runs_scheduled",
)

_THREAD_HAS_VALUES = sa.text(
    "SELECT 1 FROM information_schema.columns WHERE table_name = 'thread' AND column_name = 'values'"
)


def upgrade() -> None:
    # 1. runs metadata + multitask_strategy, plus runs.scheduled_at (delayed runs)
    #    and thread.ttl (per-thread retention) — their standalone migrations were
    #    folded into this consolidated one.
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS multitask_strategy TEXT")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS scheduled_at TIMESTAMPTZ")
    op.execute("ALTER TABLE thread ADD COLUMN IF NOT EXISTS ttl JSONB")

    # 2. thread_state table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_state (
            thread_id   TEXT PRIMARY KEY REFERENCES thread(thread_id) ON DELETE CASCADE,
            "values"    JSONB,
            interrupts  JSONB,
            values_hash TEXT,
            updated_at  TIMESTAMPTZ DEFAULT now()
        )
        """
    )

    # 3. backfill (only if the old wide columns exist) + indexes, in autocommit so
    #    batches commit incrementally and CONCURRENTLY builds don't block writes.
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        if bind.execute(_THREAD_HAS_VALUES).first():
            while bind.execute(_BACKFILL).rowcount:
                pass
        for stmt in _INDEX_CREATE:
            op.execute(stmt)

    # 4. drop the now-migrated wide columns
    op.execute('ALTER TABLE thread DROP COLUMN IF EXISTS "values"')
    op.execute("ALTER TABLE thread DROP COLUMN IF EXISTS interrupts")


def downgrade() -> None:
    op.execute('ALTER TABLE thread ADD COLUMN IF NOT EXISTS interrupts JSONB')
    op.execute('ALTER TABLE thread ADD COLUMN IF NOT EXISTS "values" JSONB')
    op.execute(
        'UPDATE thread t SET "values" = ts."values", interrupts = ts.interrupts '
        "FROM thread_state ts WHERE ts.thread_id = t.thread_id"
    )
    with op.get_context().autocommit_block():
        for name in _INDEX_NAMES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
    op.execute("DROP TABLE IF EXISTS thread_state")
    op.execute("ALTER TABLE thread DROP COLUMN IF EXISTS ttl")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS scheduled_at")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS multitask_strategy")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS metadata")
