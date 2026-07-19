"""Postgres checkpointer with per-run deletion.

Extends langgraph's ``AsyncPostgresSaver``, which through checkpoint-postgres
3.1.0 implements only whole-thread deletion (``adelete_thread``).
``adelete_for_runs`` is declared on the base saver but left
``NotImplementedError``; this fills it in for Aegra's rollback double-texting,
deleting exactly the checkpoints a run produced while keeping blobs that
surviving checkpoints still reference.
"""

import os
from collections.abc import Sequence

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from psycopg_pool import AsyncConnectionPool


def build_encrypted_serde() -> SerializerProtocol | None:
    """Return an AES checkpoint serializer when ``LANGGRAPH_AES_KEY`` is set.

    Encryption at rest for checkpoint state/blobs via langgraph's native
    ``EncryptedSerializer``. None (plaintext) when unset — plaintext rows still
    decrypt after enabling it since the serializer keys off the cipher tag.
    """
    if not os.getenv("LANGGRAPH_AES_KEY"):
        return None
    return EncryptedSerializer.from_pycryptodome_aes()


# Checkpoints carry their creating run in ``metadata->>'run_id'`` — langgraph
# writes it from the top-level config ``run_id``, which Aegra pins to its own
# run id (see langgraph_service.create_run_config).
_AFFECTED_THREADS_SQL = "SELECT DISTINCT thread_id FROM checkpoints WHERE metadata->>'run_id' = ANY(%s)"

_DELETE_WRITES_SQL = (
    "DELETE FROM checkpoint_writes cw USING checkpoints c "
    "WHERE cw.thread_id = c.thread_id AND cw.checkpoint_ns = c.checkpoint_ns "
    "AND cw.checkpoint_id = c.checkpoint_id AND c.metadata->>'run_id' = ANY(%s)"
)

_DELETE_CHECKPOINTS_SQL = "DELETE FROM checkpoints WHERE metadata->>'run_id' = ANY(%s)"

# GC blobs no surviving checkpoint references, matching the saver's own
# checkpoint->blob join on (channel, version): correct across namespaces and
# delta channels without version arithmetic.
_GC_BLOBS_SQL = (
    "DELETE FROM checkpoint_blobs b WHERE b.thread_id = ANY(%s) AND NOT EXISTS ("
    " SELECT 1 FROM checkpoints c"
    " JOIN jsonb_each_text(c.checkpoint -> 'channel_versions') cv"
    " ON cv.key = b.channel AND cv.value = b.version"
    " WHERE c.thread_id = b.thread_id AND c.checkpoint_ns = b.checkpoint_ns"
    ")"
)

# keep_latest prune: a checkpoint dies when a newer one exists in the same
# namespace (checkpoint ids are time-ordered uuid6).
_PRUNE_STALE_WRITES_SQL = (
    "DELETE FROM checkpoint_writes cw USING checkpoints c "
    "WHERE cw.thread_id = c.thread_id AND cw.checkpoint_ns = c.checkpoint_ns "
    "AND cw.checkpoint_id = c.checkpoint_id AND c.thread_id = ANY(%s) AND EXISTS ("
    " SELECT 1 FROM checkpoints newer WHERE newer.thread_id = c.thread_id"
    " AND newer.checkpoint_ns = c.checkpoint_ns AND newer.checkpoint_id > c.checkpoint_id"
    ")"
)
_PRUNE_STALE_CHECKPOINTS_SQL = (
    "DELETE FROM checkpoints c WHERE c.thread_id = ANY(%s) AND EXISTS ("
    " SELECT 1 FROM checkpoints newer WHERE newer.thread_id = c.thread_id"
    " AND newer.checkpoint_ns = c.checkpoint_ns AND newer.checkpoint_id > c.checkpoint_id"
    ")"
)


class AegraPostgresSaver(AsyncPostgresSaver):
    """``AsyncPostgresSaver`` plus per-run checkpoint deletion."""

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Delete every checkpoint (and its writes) produced by the given runs.

        Matches checkpoints via ``metadata->>'run_id'``, drops their pending
        writes, then GCs blobs no surviving checkpoint still references. Runs
        as one transaction so the blob GC observes the checkpoint deletions.

        Delta channels: deleting a run whose checkpoints hold the only
        ``_DeltaSnapshot`` blob — or whose writes a still-live descendant
        depends on — can break ``DeltaChannel`` reconstruction, per the base
        saver's contract. Safe for rollback: the run's checkpoints are the
        thread's forward tail, so nothing surviving depends on them.
        """
        ids = [str(rid) for rid in run_ids]
        if not ids:
            return
        pool = self.conn
        if not isinstance(pool, AsyncConnectionPool):
            raise TypeError("AegraPostgresSaver requires a connection pool")
        async with pool.connection() as conn, conn.transaction():
            cur = await conn.execute(_AFFECTED_THREADS_SQL, (ids,))
            thread_ids = [row["thread_id"] for row in await cur.fetchall()]
            if not thread_ids:
                return
            await conn.execute(_DELETE_WRITES_SQL, (ids,))
            await conn.execute(_DELETE_CHECKPOINTS_SQL, (ids,))
            await conn.execute(_GC_BLOBS_SQL, (thread_ids,))

    async def aprune_keep_latest(self, thread_ids: Sequence[str]) -> None:
        """Keep only the latest checkpoint per namespace for the given threads.

        Deletes every superseded checkpoint plus its writes, then GCs blobs no
        surviving checkpoint references, in one transaction.

        Delta channels: dropping ancestor checkpoints can sever ``DeltaChannel``
        reconstruction (per the base saver's ``prune`` contract) — threads whose
        graphs use delta channels should not be pruned with this strategy.
        """
        ids = [str(tid) for tid in thread_ids]
        if not ids:
            return
        pool = self.conn
        if not isinstance(pool, AsyncConnectionPool):
            raise TypeError("AegraPostgresSaver requires a connection pool")
        async with pool.connection() as conn, conn.transaction():
            await conn.execute(_PRUNE_STALE_WRITES_SQL, (ids,))
            await conn.execute(_PRUNE_STALE_CHECKPOINTS_SQL, (ids,))
            await conn.execute(_GC_BLOBS_SQL, (ids,))
