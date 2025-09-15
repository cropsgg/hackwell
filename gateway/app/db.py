"""Database connection and session management."""

import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool
import structlog

from .config import settings, get_database_config

logger = structlog.get_logger()

# Global database engine
engine = None
async_session_factory = None


def get_engine():
    """Get or create the database engine."""
    global engine
    if engine is None:
        config = get_database_config()
        engine = create_async_engine(
            config["url"],
            echo=config["echo"],
            pool_size=config["pool_size"],
            max_overflow=config["max_overflow"],
            pool_pre_ping=config["pool_pre_ping"],
            pool_recycle=config["pool_recycle"],
            poolclass=NullPool if settings.environment == "test" else None,
        )
    return engine


def get_session_factory():
    """Get or create the session factory."""
    global async_session_factory
    if async_session_factory is None:
        async_session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """Context manager for database sessions."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


class DatabaseManager:
    """Database connection and query management."""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        
    async def connect(self):
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'jit': 'off'
                }
            )
            logger.info("Database connection pool created")
        except Exception as e:
            logger.error("Failed to create database pool", error=str(e))
            raise
    
    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def execute_query(self, query: str, *args):
        """Execute a query and return results."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            try:
                result = await connection.fetch(query, *args)
                return result
            except Exception as e:
                logger.error("Query execution failed", query=query, error=str(e))
                raise
    
    async def execute_command(self, command: str, *args):
        """Execute a command (INSERT, UPDATE, DELETE)."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            try:
                result = await connection.execute(command, *args)
                return result
            except Exception as e:
                logger.error("Command execution failed", command=command, error=str(e))
                raise
    
    async def call_function(self, function_name: str, *args):
        """Call a database function."""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            try:
                result = await connection.fetchval(f"SELECT {function_name}($1)", *args)
                return result
            except Exception as e:
                logger.error("Function call failed", function=function_name, error=str(e))
                raise


# Global database manager instance
db_manager = DatabaseManager()


async def init_database():
    """Initialize database connections on startup."""
    await db_manager.connect()
    logger.info("Database initialized")


async def close_database():
    """Close database connections on shutdown."""
    await db_manager.disconnect()
    if engine:
        await engine.dispose()
    logger.info("Database connections closed")


# Raw SQL query helpers for performance-critical operations
class QueryBuilder:
    """Helper class for building SQL queries."""
    
    @staticmethod
    def get_patient_summary(patient_id: str) -> str:
        """Get comprehensive patient summary query."""
        return """
        SELECT 
            p.id,
            p.demographics,
            p.consent,
            p.created_at,
            get_patient_risk_summary(p.id) as risk_summary
        FROM patients p
        WHERE p.id = $1
        """
    
    @staticmethod
    def get_patient_vitals(patient_id: str, limit: int = 100) -> str:
        """Get recent patient vitals query."""
        return """
        SELECT 
            type,
            value,
            unit,
            ts,
            source
        FROM vitals
        WHERE patient_id = $1
        ORDER BY ts DESC
        LIMIT $2
        """
    
    @staticmethod
    def get_recommendations_with_evidence(patient_id: str) -> str:
        """Get patient recommendations with evidence links."""
        return """
        SELECT 
            r.id,
            r.snapshot_ts,
            r.careplan,
            r.explainer,
            r.model_version,
            r.status,
            r.risk_score,
            r.risk_category,
            r.created_at,
            COALESCE(
                json_agg(
                    json_build_object(
                        'source_type', e.source_type,
                        'url', e.url,
                        'title', e.title,
                        'weight', e.weight,
                        'snippet', e.snippet
                    )
                ) FILTER (WHERE e.id IS NOT NULL),
                '[]'::json
            ) as evidence_links
        FROM recommendations r
        LEFT JOIN evidence_links e ON e.recommendation_id = r.id
        WHERE r.patient_id = $1
        GROUP BY r.id, r.snapshot_ts, r.careplan, r.explainer, 
                 r.model_version, r.status, r.risk_score, r.risk_category, r.created_at
        ORDER BY r.created_at DESC
        """
    
    @staticmethod
    def get_clinician_patients(clinician_id: str) -> str:
        """Get patients assigned to a clinician."""
        return """
        SELECT 
            p.id,
            p.demographics,
            COALESCE(
                (
                    SELECT r.risk_score 
                    FROM recommendations r 
                    WHERE r.patient_id = p.id 
                    AND r.status = 'approved'
                    ORDER BY r.created_at DESC 
                    LIMIT 1
                ), 0
            ) as latest_risk_score,
            COALESCE(
                (
                    SELECT r.risk_category 
                    FROM recommendations r 
                    WHERE r.patient_id = p.id 
                    AND r.status = 'approved'
                    ORDER BY r.created_at DESC 
                    LIMIT 1
                ), 'unknown'
            ) as latest_risk_category
        FROM patients p
        JOIN clinician_patients cp ON cp.patient_id = p.id
        WHERE cp.clinician_user_id = $1 
        AND cp.active = true
        ORDER BY latest_risk_score DESC
        """


# Health check queries
HEALTH_CHECK_QUERIES = {
    "basic": "SELECT 1",
    "tables": "SELECT COUNT(*) FROM patients",
    "functions": "SELECT get_patient_risk_summary($1)",
}


async def health_check() -> dict:
    """Perform database health check."""
    results = {}
    
    try:
        # Basic connectivity
        result = await db_manager.execute_query(HEALTH_CHECK_QUERIES["basic"])
        results["connectivity"] = "ok"
        
        # Table access
        result = await db_manager.execute_query(HEALTH_CHECK_QUERIES["tables"])
        results["tables"] = "ok"
        
        # Function availability (with dummy UUID)
        from uuid import uuid4
        dummy_id = str(uuid4())
        try:
            await db_manager.execute_query(
                HEALTH_CHECK_QUERIES["functions"], 
                dummy_id
            )
            results["functions"] = "ok"
        except Exception:
            results["functions"] = "degraded"
        
        results["status"] = "healthy"
        
    except Exception as e:
        results["status"] = "unhealthy"
        results["error"] = str(e)
        logger.error("Database health check failed", error=str(e))
    
    return results
