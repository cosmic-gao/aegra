"""Add the one-running-per-thread unique index + prune redundant single-col indexes.

Two coupled index-hygiene changes:

1. ``uq_runs_one_running_per_thread`` — the partial unique index
   ``runs (thread_id) WHERE status='running'``. The ORM has declared it as a
   hard multitask-serialization invariant for a while (``core/orm.py``) and the
   worker's atomic claim path relies on it as the last-resort dedup backstop
   (``worker_executor.py`` catches its ``IntegrityError``), but no migration ever
   created it — so production DBs run without the backstop and a claim race can
   put two runs of the same thread into ``running`` at once. This creates it for
   real. Pre-existing duplicate running rows (the illegal state) are resolved
   first — newest kept, older ones demoted to ``error`` — otherwise the unique
   build would fail and leave an INVALID index.

2. Drop three single-column indexes that are exact left-prefixes of an existing
   composite and so serve no query the composite doesn't (equality on the lead
   column + FK cascade lookups both use the composite's left prefix):
     - ``idx_thread_user (user_id)``         ⊂ ``idx_thread_user_created (user_id, created_at)``
     - ``idx_runs_thread_id (thread_id)``    ⊂ ``idx_runs_thread_created (thread_id, created_at)``
     - ``idx_assistant_user (user_id)``      ⊂ ``idx_assistant_user_assistant (user_id, assistant_id)``
   Removing them cuts write amplification and storage with no read regression.

All index DDL runs ``CONCURRENTLY`` inside ``autocommit_block`` (only a
``SHARE UPDATE EXCLUSIVE`` lock, no write stall) and is idempotent /
resumable via ``IF [NOT] EXISTS`` — including after an interrupted
``CONCURRENTLY`` build that left an INVALID index behind.

Revision ID: a7c3e1f9b2d4
Revises: e5b9d2f7a3c1
Create Date: 2026-07-19 00:00:00.000000
"""

from alembic import op

revision = "a7c3e1f9b2d4"
down_revision = "e5b9d2f7a3c1"
branch_labels = None
depends_on = None

_UQ_NAME = "uq_runs_one_running_per_thread"

# Keep the most recent running run per thread; demote the rest. Only touches
# threads that already hold >1 running row (the invariant this index enforces).
_DEDUP_RUNNING = """
    UPDATE runs SET
        status = 'error',
        error_message = COALESCE(error_message, 'superseded: duplicate running run per thread'),
        updated_at = now()
    WHERE run_id IN (
        SELECT run_id FROM (
            SELECT run_id,
                   row_number() OVER (PARTITION BY thread_id ORDER BY created_at DESC, run_id DESC) AS rn
            FROM runs
            WHERE status = 'running'
        ) ranked
        WHERE ranked.rn > 1
    )
"""

# (index_name, create_stmt) for the three redundant single-col indexes, so
# upgrade drops them and downgrade recreates them from the same source.
_REDUNDANT = (
    ("idx_thread_user", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thread_user ON thread (user_id)"),
    ("idx_runs_thread_id", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_runs_thread_id ON runs (thread_id)"),
    ("idx_assistant_user", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assistant_user ON assistant (user_id)"),
)


def upgrade() -> None:
    # Resolve illegal duplicates transactionally before the concurrent build.
    op.execute(_DEDUP_RUNNING)
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_UQ_NAME}")
        op.execute(
            f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS {_UQ_NAME} "
            "ON runs (thread_id) WHERE status = 'running'"
        )
        for name, _ in _REDUNDANT:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for _, create_stmt in _REDUNDANT:
            op.execute(create_stmt)
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_UQ_NAME}")
