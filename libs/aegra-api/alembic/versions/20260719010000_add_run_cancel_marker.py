"""Add a persistent cancel marker to the runs table.

Cancellation previously relied solely on Redis pub/sub — a single fire-and-forget
publish with no ack. When that message is dropped, the owning worker keeps
heartbeating and runs to completion, and ``finalize_run`` overwrites the
API-written ``interrupted`` back to ``success`` (the cancel silently fails).

``cancel_requested`` persists the intent in Postgres so the worker's heartbeat
loop (which already round-trips the DB every ~10s to validate its lease) reads
it in the same query and stops the job. Cancellation is now durable,
cross-instance, and independent of pub/sub (kept only as an accelerator). It
also survives a worker crash: the reaper re-enqueues the run with the marker
intact, so the next worker honors it.

No index: the heartbeat reads the flag by ``run_id`` (primary key) in its
existing lease-extension UPDATE, so no additional scan path is introduced.

Revision ID: b8d4f2a1c3e5
Revises: a7c3e1f9b2d4
Create Date: 2026-07-19 01:00:00.000000
"""

from alembic import op

revision = "b8d4f2a1c3e5"
down_revision = "a7c3e1f9b2d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT false")


def downgrade() -> None:
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS cancel_requested")
