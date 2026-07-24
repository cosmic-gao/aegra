"""Add a partial index supporting per-run TTL sweeping.

The run TTL sweeper deletes terminal runs older than a cutoff. This partial
index on ``updated_at`` (restricted to terminal statuses) keeps that scan cheap
without indexing the hot pending/running rows.

Index-only, no data migration. Built CONCURRENTLY in an autocommit block so it
never locks the (potentially large) runs table; IF NOT EXISTS keeps it resumable.

Revision ID: d1f7b3a9c5e2
Revises: c9e5a3f1d2b7
Create Date: 2026-07-19 03:00:00.000000
"""

from alembic import op

revision = "d1f7b3a9c5e2"
down_revision = "c9e5a3f1d2b7"
branch_labels = None
depends_on = None

_INDEX = "idx_runs_ttl_sweep"
_TERMINAL = "('success', 'error', 'interrupted', 'timeout')"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX} "
            f"ON runs (updated_at) WHERE status IN {_TERMINAL}"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX}")
