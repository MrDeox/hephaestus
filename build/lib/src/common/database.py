"""
Database utilities with connection pooling for Hephaestus RSI.

Provides SQLite and async database operations with automatic
connection pool management and performance optimization.
"""

import asyncio
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from .performance import AsyncConnectionPool, get_performance_monitor
from .exceptions import StorageError, create_error_context

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections with pooling and optimization."""
    
    def __init__(
        self,
        database_path: Union[str, Path] = "data/hephaestus.db",
        pool_size: int = 10,
        enable_wal: bool = True,
        enable_foreign_keys: bool = True
    ):
        self.database_path = Path(database_path)
        self.pool_size = pool_size
        self.enable_wal = enable_wal
        self.enable_foreign_keys = enable_foreign_keys
        
        # Ensure database directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        self.pool: Optional[AsyncConnectionPool] = None
        self.sync_connections: Dict[int, sqlite3.Connection] = {}
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.monitor = get_performance_monitor()
    
    async def initialize(self) -> None:
        """Initialize database and connection pool."""
        # Create connection pool
        self.pool = AsyncConnectionPool(
            factory=self._create_connection,
            min_size=2,
            max_size=self.pool_size,
            max_idle_time=300.0
        )
        
        await self.pool.initialize()
        
        # Initialize database schema
        await self._initialize_schema()
    
    async def close(self) -> None:
        """Close database connections and pool."""
        if self.pool:
            await self.pool.close()
        
        # Close sync connections
        with self.lock:
            for conn in self.sync_connections.values():
                conn.close()
            self.sync_connections.clear()
    
    async def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimizations."""
        conn = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
            timeout=30.0
        )
        
        # Enable optimizations
        if self.enable_wal:
            conn.execute("PRAGMA journal_mode=WAL")
        
        if self.enable_foreign_keys:
            conn.execute("PRAGMA foreign_keys=ON")
        
        # Performance optimizations
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        conn.row_factory = sqlite3.Row
        return conn
    
    async def _initialize_schema(self) -> None:
        """Initialize database schema if needed."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS system_state (
            id INTEGER PRIMARY KEY,
            state_data TEXT NOT NULL,
            checksum TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS model_metadata (
            model_id TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            version TEXT NOT NULL,
            file_path TEXT,
            metrics TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY,
            event_type TEXT NOT NULL,
            component TEXT NOT NULL,
            event_data TEXT NOT NULL,
            checksum TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY,
            operation_name TEXT NOT NULL,
            duration_ms REAL NOT NULL,
            memory_mb REAL,
            cpu_percent REAL,
            success BOOLEAN NOT NULL,
            metadata TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_performance_operation ON performance_metrics(operation_name);
        """
        
        async with self.connection() as conn:
            for statement in schema_sql.split(';'):
                if statement.strip():
                    await self._execute(conn, statement)
    
    @asynccontextmanager
    async def connection(self):
        """Async context manager for database connections."""
        if not self.pool:
            raise StorageError(
                "Database pool not initialized",
                context=create_error_context("connection_acquire")
            )
        
        async with self.pool.connection() as conn:
            yield conn
    
    @contextmanager
    def sync_connection(self):
        """Sync context manager for database connections."""
        thread_id = threading.get_ident()
        
        with self.lock:
            if thread_id not in self.sync_connections:
                self.sync_connections[thread_id] = sqlite3.connect(
                    str(self.database_path),
                    check_same_thread=False
                )
                
                conn = self.sync_connections[thread_id]
                if self.enable_wal:
                    conn.execute("PRAGMA journal_mode=WAL")
                if self.enable_foreign_keys:
                    conn.execute("PRAGMA foreign_keys=ON")
                
                conn.row_factory = sqlite3.Row
            
            conn = self.sync_connections[thread_id]
        
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
    
    async def _execute(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: Optional[Tuple] = None
    ) -> sqlite3.Cursor:
        """Execute SQL with performance monitoring."""
        with self.monitor.measure_operation(f"db_execute"):
            try:
                if params:
                    cursor = conn.execute(sql, params)
                else:
                    cursor = conn.execute(sql)
                conn.commit()
                return cursor
            except Exception as e:
                conn.rollback()
                raise StorageError(
                    f"Database execution failed: {e}",
                    context=create_error_context("database_execute", sql=sql)
                )
    
    async def execute(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> sqlite3.Cursor:
        """Execute SQL statement."""
        async with self.connection() as conn:
            return await self._execute(conn, sql, params)
    
    async def execute_many(
        self,
        sql: str,
        params_list: List[Tuple]
    ) -> None:
        """Execute SQL statement with multiple parameter sets."""
        async with self.connection() as conn:
            with self.monitor.measure_operation(f"db_execute_many"):
                try:
                    conn.executemany(sql, params_list)
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise StorageError(
                        f"Database batch execution failed: {e}",
                        context=create_error_context("database_execute_many", sql=sql)
                    )
    
    async def fetch_one(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> Optional[sqlite3.Row]:
        """Fetch one row."""
        cursor = await self.execute(sql, params)
        return cursor.fetchone()
    
    async def fetch_all(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> List[sqlite3.Row]:
        """Fetch all rows."""
        cursor = await self.execute(sql, params)
        return cursor.fetchall()
    
    async def fetch_many(
        self,
        sql: str,
        size: int,
        params: Optional[Tuple] = None
    ) -> List[sqlite3.Row]:
        """Fetch specified number of rows."""
        cursor = await self.execute(sql, params)
        return cursor.fetchmany(size)
    
    # Specific data access methods
    
    async def save_system_state(
        self,
        state_data: str,
        checksum: str
    ) -> int:
        """Save system state with checksum."""
        cursor = await self.execute(
            "INSERT INTO system_state (state_data, checksum) VALUES (?, ?)",
            (state_data, checksum)
        )
        return cursor.lastrowid
    
    async def get_latest_system_state(self) -> Optional[Dict[str, Any]]:
        """Get the latest system state."""
        row = await self.fetch_one(
            "SELECT * FROM system_state ORDER BY created_at DESC LIMIT 1"
        )
        
        if row:
            return {
                "id": row["id"],
                "state_data": row["state_data"],
                "checksum": row["checksum"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        return None
    
    async def save_model_metadata(
        self,
        model_id: str,
        model_type: str,
        version: str,
        file_path: Optional[str] = None,
        metrics: Optional[str] = None
    ) -> None:
        """Save model metadata."""
        await self.execute(
            """INSERT OR REPLACE INTO model_metadata 
               (model_id, model_type, version, file_path, metrics, updated_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (model_id, model_type, version, file_path, metrics)
        )
    
    async def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by ID."""
        row = await self.fetch_one(
            "SELECT * FROM model_metadata WHERE model_id = ?",
            (model_id,)
        )
        
        if row:
            return {
                "model_id": row["model_id"],
                "model_type": row["model_type"],
                "version": row["version"],
                "file_path": row["file_path"],
                "metrics": row["metrics"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        return None
    
    async def log_audit_event(
        self,
        event_type: str,
        component: str,
        event_data: str,
        checksum: str
    ) -> int:
        """Log audit event."""
        cursor = await self.execute(
            "INSERT INTO audit_log (event_type, component, event_data, checksum) VALUES (?, ?, ?, ?)",
            (event_type, component, event_data, checksum)
        )
        return cursor.lastrowid
    
    async def record_performance_metric(
        self,
        operation_name: str,
        duration_ms: float,
        memory_mb: Optional[float] = None,
        cpu_percent: Optional[float] = None,
        success: bool = True,
        metadata: Optional[str] = None
    ) -> None:
        """Record performance metric."""
        await self.execute(
            """INSERT INTO performance_metrics 
               (operation_name, duration_ms, memory_mb, cpu_percent, success, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (operation_name, duration_ms, memory_mb, cpu_percent, success, metadata)
        )
    
    async def get_performance_stats(
        self,
        operation_name: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance statistics."""
        base_sql = """
        SELECT 
            AVG(duration_ms) as avg_duration,
            MIN(duration_ms) as min_duration,
            MAX(duration_ms) as max_duration,
            COUNT(*) as total_count,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
        FROM performance_metrics 
        WHERE timestamp > datetime('now', '-{} hours')
        """.format(hours)
        
        if operation_name:
            base_sql += " AND operation_name = ?"
            params = (operation_name,)
        else:
            params = None
        
        row = await self.fetch_one(base_sql, params)
        
        if row:
            success_rate = (row["success_count"] / row["total_count"]) if row["total_count"] > 0 else 0
            return {
                "operation_name": operation_name or "all",
                "time_period_hours": hours,
                "avg_duration_ms": row["avg_duration"] or 0,
                "min_duration_ms": row["min_duration"] or 0,
                "max_duration_ms": row["max_duration"] or 0,
                "total_operations": row["total_count"] or 0,
                "success_rate": success_rate
            }
        
        return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database connection pool statistics."""
        if self.pool:
            return {
                "pool_stats": self.pool.get_stats(),
                "database_path": str(self.database_path),
                "database_size_mb": self.database_path.stat().st_size / (1024 * 1024) if self.database_path.exists() else 0
            }
        return {"error": "Pool not initialized"}


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def set_database_manager(manager: DatabaseManager) -> None:
    """Set global database manager instance."""
    global _db_manager
    _db_manager = manager