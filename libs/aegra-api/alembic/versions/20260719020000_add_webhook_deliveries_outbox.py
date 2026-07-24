"""Add the webhook_deliveries transactional outbox.

Run-completion webhooks were delivered by an in-process fire-and-forget task, so
a worker crash during the retry window silently lost the notification. This
outbox row is inserted in the same transaction as the run's terminal status, so
a durable ``pending`` delivery always exists once a run finalizes; a background
deliverer claims, POSTs, retries with backoff, and dead-letters exhausted rows.

New empty table + index — pure additive DDL, no data migration. IF NOT EXISTS
keeps it resumable.

Revision ID: c9e5a3f1d2b7
Revises: b8d4f2a1c3e5
Create Date: 2026-07-19 02:00:00.000000
"""

from alembic import op

revision = "c9e5a3f1d2b7"
down_revision = "b8d4f2a1c3e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS webhook_deliveries (
            id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            url TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_due ON webhook_deliveries (status, next_attempt_at)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS webhook_deliveries")
