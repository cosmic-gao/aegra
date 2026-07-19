"""Database manager with LangGraph integration"""

from typing import cast

import structlog
from langgraph.store.base import TTLConfig
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aegra_api.config import load_store_config
from aegra_api.core.checkpointer import AegraPostgresSaver, build_encrypted_serde
from aegra_api.settings import settings

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages database connections and LangGraph persistence components"""

    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None

        # Shared pool for LangGraph components (Checkpointer + Store). Typed as
        # DictRow because the pool is opened with row_factory=dict_row below,
        # which is what the langgraph savers/stores require.
        self.lg_pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None
        self._checkpointer: AegraPostgresSaver | None = None
        self._store: AsyncPostgresStore | None = None
        self._database_url = settings.db.database_url

    async def initialize(self) -> None:
        """Initialize database connections and LangGraph components"""
        # Idempotency check: if already initialized, do nothing
        if self.engine:
            return

        # 1. SQLAlchemy Engine (app metadata, uses asyncpg)
        # We strictly limit this pool because the main load
        # is handled by LangGraph components.
        self.engine = create_async_engine(
            self._database_url,
            pool_size=settings.pool.SQLALCHEMY_POOL_SIZE,
            max_overflow=settings.pool.SQLALCHEMY_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=settings.db.DB_ECHO_LOG,
            connect_args={"prepared_statement_cache_size": 0},  # PgBouncer compatibility
        )

        lg_max = settings.pool.LANGGRAPH_MAX_POOL_SIZE
        lg_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,  # Disable prepared statements for PgBouncer compatibility
            "row_factory": dict_row,  # LangGraph requires dictionary rows, not tuples
        }

        # Create a single shared pool.
        # 'open=False' is important to avoid RuntimeWarning; we open it explicitly below.
        self.lg_pool = cast(
            "AsyncConnectionPool[AsyncConnection[DictRow]]",
            AsyncConnectionPool(
                conninfo=settings.db.database_url_sync,
                min_size=settings.pool.LANGGRAPH_MIN_POOL_SIZE,
                max_size=lg_max,
                open=False,
                kwargs=lg_kwargs,
                check=AsyncConnectionPool.check_connection,
            ),
        )

        # Explicitly open the pool
        await self.lg_pool.open()

        # 2. Initialize LangGraph components using the shared pool
        # Passing 'conn=self.lg_pool' prevents components from creating their own pools.

        logger.info(f"Initializing LangGraph components with shared pool (max {lg_max} conns)...")

        # Encrypt checkpoint state at rest when LANGGRAPH_AES_KEY is configured.
        self._checkpointer = AegraPostgresSaver(conn=self.lg_pool, serde=build_encrypted_serde())
        await self._checkpointer.setup()  # Ensure tables exist

        # Load store configuration for semantic search + TTL (if configured).
        # Aegra's config TypedDicts mirror langgraph's shape; cast at the boundary.
        store_config = load_store_config()
        index_config = cast(PostgresIndexConfig | None, store_config.get("index") if store_config else None)
        ttl_config = cast(TTLConfig | None, store_config.get("ttl") if store_config else None)

        self._store = AsyncPostgresStore(conn=self.lg_pool, index=index_config, ttl=ttl_config)
        await self._store.setup()  # Ensure tables exist

        if index_config:
            embed_model = index_config.get("embed", "unknown")
            logger.info(f"Semantic store enabled with embeddings: {embed_model}")
        if ttl_config:
            # No-ops when ttl_config has no sweep interval; starts the background
            # sweeper otherwise.
            await self._store.start_ttl_sweeper()
            logger.info(f"Store TTL enabled: {ttl_config}")

        logger.info("✅ Database and LangGraph components initialized")

    async def close(self) -> None:
        """Close database connections"""
        # Stop the store TTL sweeper before tearing down the pool it uses.
        if self._store is not None:
            try:
                await self._store.stop_ttl_sweeper()
            except Exception as exc:
                logger.warning("Failed to stop store TTL sweeper", error=str(exc))

        # Close SQLAlchemy engine
        if self.engine:
            await self.engine.dispose()
            self.engine = None

        # Close shared LangGraph pool
        if self.lg_pool:
            await self.lg_pool.close()
            self.lg_pool = None
            self._checkpointer = None
            self._store = None

        logger.info("✅ Database connections closed")

    def get_checkpointer(self) -> AegraPostgresSaver:
        """Return the live checkpointer instance."""
        if self._checkpointer is None:
            raise RuntimeError("Database not initialized")
        return self._checkpointer

    def get_store(self) -> AsyncPostgresStore:
        """Return the live AsyncPostgresStore instance."""
        if self._store is None:
            raise RuntimeError("Database not initialized")
        return self._store

    def get_engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine for metadata tables"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        return self.engine


# Global database manager instance
db_manager = DatabaseManager()
