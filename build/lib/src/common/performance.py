"""
Performance optimization utilities for Hephaestus RSI.

Provides caching, connection pooling, batch processing,
and performance monitoring capabilities.
"""

import asyncio
import functools
import hashlib
import json
import pickle
import threading
import time
import weakref
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
import gc
import psutil
import numpy as np

from ..common.exceptions import ResourceExhaustionError
from config.base_config import get_config


@dataclass
class PerformanceMetrics:
    """Performance metrics data."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_percent: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_before_mb': self.memory_before,
            'memory_after_mb': self.memory_after,
            'memory_delta_mb': self.memory_delta,
            'cpu_percent': self.cpu_percent,
            'success': self.success,
            'metadata': self.metadata
        }


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats['hits'] += 1
                return value
            else:
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
                self.stats['evictions'] += 1
            
            self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': hit_rate,
                'size': len(self.cache),
                'max_size': self.max_size
            }


class AsyncConnectionPool:
    """Async connection pool for database and network connections."""
    
    def __init__(
        self,
        factory: Callable,
        min_size: int = 1,
        max_size: int = 10,
        max_idle_time: float = 300.0,  # 5 minutes
        health_check_interval: float = 60.0  # 1 minute
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        
        self.available_connections: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.active_connections: Set[Any] = set()
        self.connection_times: Dict[Any, float] = {}
        self.lock = asyncio.Lock()
        self.closed = False
        
        # Background task for health checks
        self.health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        # Create minimum connections
        for _ in range(self.min_size):
            try:
                conn = await self.factory()
                await self.available_connections.put(conn)
                self.connection_times[conn] = time.time()
            except Exception as e:
                print(f"Failed to create initial connection: {e}")
        
        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def acquire(self) -> Any:
        """Acquire a connection from the pool."""
        if self.closed:
            raise RuntimeError("Connection pool is closed")
        
        async with self.lock:
            # Try to get available connection
            try:
                conn = self.available_connections.get_nowait()
                self.active_connections.add(conn)
                return conn
            except asyncio.QueueEmpty:
                pass
            
            # Create new connection if under limit
            if len(self.active_connections) < self.max_size:
                try:
                    conn = await self.factory()
                    self.active_connections.add(conn)
                    self.connection_times[conn] = time.time()
                    return conn
                except Exception as e:
                    raise RuntimeError(f"Failed to create connection: {e}")
            
            # Wait for available connection
            conn = await self.available_connections.get()
            self.active_connections.add(conn)
            return conn
    
    async def release(self, conn: Any) -> None:
        """Release a connection back to the pool."""
        async with self.lock:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                
                # Check if connection is still healthy
                if await self._is_connection_healthy(conn):
                    try:
                        self.available_connections.put_nowait(conn)
                        self.connection_times[conn] = time.time()
                    except asyncio.QueueFull:
                        # Pool is full, close the connection
                        await self._close_connection(conn)
                else:
                    # Connection is unhealthy, close it
                    await self._close_connection(conn)
    
    async def close(self) -> None:
        """Close the connection pool."""
        self.closed = True
        
        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self.lock:
            # Close active connections
            for conn in list(self.active_connections):
                await self._close_connection(conn)
            
            # Close available connections
            while not self.available_connections.empty():
                try:
                    conn = self.available_connections.get_nowait()
                    await self._close_connection(conn)
                except asyncio.QueueEmpty:
                    break
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self.closed:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in health check: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on idle connections."""
        current_time = time.time()
        connections_to_remove = []
        
        # Check available connections
        temp_connections = []
        while not self.available_connections.empty():
            try:
                conn = self.available_connections.get_nowait()
                
                # Check if connection is too old
                if (current_time - self.connection_times.get(conn, 0)) > self.max_idle_time:
                    connections_to_remove.append(conn)
                elif not await self._is_connection_healthy(conn):
                    connections_to_remove.append(conn)
                else:
                    temp_connections.append(conn)
            except asyncio.QueueEmpty:
                break
        
        # Put back healthy connections
        for conn in temp_connections:
            try:
                self.available_connections.put_nowait(conn)
            except asyncio.QueueFull:
                await self._close_connection(conn)
        
        # Close unhealthy connections
        for conn in connections_to_remove:
            await self._close_connection(conn)
        
        # Ensure minimum connections
        current_available = self.available_connections.qsize()
        current_active = len(self.active_connections)
        total_connections = current_available + current_active
        
        if total_connections < self.min_size:
            for _ in range(self.min_size - total_connections):
                try:
                    conn = await self.factory()
                    await self.available_connections.put(conn)
                    self.connection_times[conn] = time.time()
                except Exception as e:
                    print(f"Failed to create connection during health check: {e}")
                    break
    
    async def _is_connection_healthy(self, conn: Any) -> bool:
        """Check if a connection is healthy."""
        try:
            # This is a placeholder - implement actual health check
            # For database connections, might execute a simple query
            # For network connections, might ping the server
            return hasattr(conn, 'ping') and await conn.ping()
        except Exception:
            return False
    
    async def _close_connection(self, conn: Any) -> None:
        """Close a connection."""
        try:
            if hasattr(conn, 'close'):
                if asyncio.iscoroutinefunction(conn.close):
                    await conn.close()
                else:
                    conn.close()
        except Exception as e:
            print(f"Error closing connection: {e}")
        finally:
            self.connection_times.pop(conn, None)
    
    @asynccontextmanager
    async def connection(self):
        """Context manager for acquiring and releasing connections."""
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'available_connections': self.available_connections.qsize(),
            'active_connections': len(self.active_connections),
            'total_connections': self.available_connections.qsize() + len(self.active_connections),
            'min_size': self.min_size,
            'max_size': self.max_size,
            'closed': self.closed
        }


class BatchProcessor:
    """Batch processor for efficient bulk operations."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_wait_time: float = 1.0,
        processor_func: Optional[Callable] = None
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor_func = processor_func
        
        self.batch_queue: List[Any] = []
        self.batch_timestamps: List[float] = []
        self.pending_futures: List[asyncio.Future] = []
        self.lock = asyncio.Lock()
        self.processing = False
        
        # Background task for time-based processing
        self.flush_task: Optional[asyncio.Task] = None
        self.start_flush_task()
    
    def start_flush_task(self) -> None:
        """Start the periodic flush task."""
        if self.flush_task is None or self.flush_task.done():
            self.flush_task = asyncio.create_task(self._flush_loop())
    
    async def add(self, item: Any) -> Any:
        """Add item to batch and return result."""
        future = asyncio.Future()
        
        async with self.lock:
            self.batch_queue.append(item)
            self.batch_timestamps.append(time.time())
            self.pending_futures.append(future)
            
            # Process batch if it's full
            if len(self.batch_queue) >= self.batch_size:
                await self._process_batch()
        
        return await future
    
    async def flush(self) -> None:
        """Flush current batch."""
        async with self.lock:
            if self.batch_queue:
                await self._process_batch()
    
    async def _process_batch(self) -> None:
        """Process current batch."""
        if self.processing or not self.batch_queue:
            return
        
        self.processing = True
        
        try:
            # Extract current batch
            batch_items = self.batch_queue.copy()
            batch_futures = self.pending_futures.copy()
            
            # Clear queues
            self.batch_queue.clear()
            self.batch_timestamps.clear()
            self.pending_futures.clear()
            
            # Process batch
            if self.processor_func:
                try:
                    if asyncio.iscoroutinefunction(self.processor_func):
                        results = await self.processor_func(batch_items)
                    else:
                        results = self.processor_func(batch_items)
                    
                    # Set results for futures
                    if isinstance(results, list) and len(results) == len(batch_futures):
                        for future, result in zip(batch_futures, results):
                            if not future.cancelled():
                                future.set_result(result)
                    else:
                        # Single result for all
                        for future in batch_futures:
                            if not future.cancelled():
                                future.set_result(results)
                
                except Exception as e:
                    # Set exception for all futures
                    for future in batch_futures:
                        if not future.cancelled():
                            future.set_exception(e)
            else:
                # No processor function, return items as-is
                for future, item in zip(batch_futures, batch_items):
                    if not future.cancelled():
                        future.set_result(item)
        
        finally:
            self.processing = False
    
    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while True:
            try:
                await asyncio.sleep(self.max_wait_time)
                
                async with self.lock:
                    if self.batch_queue and not self.processing:
                        # Check if oldest item has exceeded wait time
                        oldest_time = min(self.batch_timestamps) if self.batch_timestamps else time.time()
                        if time.time() - oldest_time >= self.max_wait_time:
                            await self._process_batch()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in flush loop: {e}")
    
    async def close(self) -> None:
        """Close the batch processor."""
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining items
        await self.flush()


class PerformanceMonitor:
    """Performance monitoring and profiling."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats = defaultdict(list)
        self.process = psutil.Process()
        self.lock = threading.Lock()
    
    @contextmanager
    def measure_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for measuring operation performance."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        start_cpu = self.process.cpu_percent()
        
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_before=start_memory,
                memory_after=end_memory,
                memory_delta=end_memory - start_memory,
                cpu_percent=self.process.cpu_percent(),
                success=success,
                metadata=metadata or {}
            )
            
            self.record_metrics(metrics)
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
            self.operation_stats[metrics.operation_name].append(metrics)
            
            # Limit per-operation history
            if len(self.operation_stats[metrics.operation_name]) > 1000:
                self.operation_stats[metrics.operation_name] = \
                    self.operation_stats[metrics.operation_name][-1000:]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self.lock:
            operation_metrics = self.operation_stats.get(operation_name, [])
            
            if not operation_metrics:
                return {}
            
            durations = [m.duration for m in operation_metrics]
            memory_deltas = [m.memory_delta for m in operation_metrics]
            success_count = sum(1 for m in operation_metrics if m.success)
            
            return {
                'operation_name': operation_name,
                'total_calls': len(operation_metrics),
                'success_rate': success_count / len(operation_metrics),
                'duration': {
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'p95': np.percentile(durations, 95),
                    'p99': np.percentile(durations, 99)
                },
                'memory_delta': {
                    'mean': np.mean(memory_deltas),
                    'median': np.median(memory_deltas),
                    'std': np.std(memory_deltas),
                    'min': np.min(memory_deltas),
                    'max': np.max(memory_deltas)
                }
            }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        with self.lock:
            all_metrics = list(self.metrics_history)
            
            if not all_metrics:
                return {}
            
            total_calls = len(all_metrics)
            success_count = sum(1 for m in all_metrics if m.success)
            
            # Group by operation
            operation_counts = defaultdict(int)
            for m in all_metrics:
                operation_counts[m.operation_name] += 1
            
            # Recent performance (last 100 operations)
            recent_metrics = all_metrics[-100:]
            recent_durations = [m.duration for m in recent_metrics]
            
            return {
                'total_operations': total_calls,
                'overall_success_rate': success_count / total_calls,
                'operations_by_type': dict(operation_counts),
                'recent_performance': {
                    'mean_duration': np.mean(recent_durations) if recent_durations else 0,
                    'p95_duration': np.percentile(recent_durations, 95) if recent_durations else 0
                },
                'monitoring_period_hours': (
                    (all_metrics[-1].end_time - all_metrics[0].start_time) / 3600
                    if len(all_metrics) > 1 else 0
                )
            }


# Performance decorators

def cache_result(cache_key_func: Optional[Callable] = None, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size=1000)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(*args, **kwargs)
            else:
                key = hashlib.md5(
                    json.dumps([args, kwargs], sort_keys=True, default=str).encode()
                ).hexdigest()
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                if ttl is None or (time.time() - cached_result['timestamp']) < ttl:
                    return cached_result['value']
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache.put(key, {
                'value': result,
                'timestamp': time.time()
            })
            
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


def monitor_performance(operation_name: Optional[str] = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        monitor = PerformanceMonitor()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with monitor.measure_operation(op_name):
                return func(*args, **kwargs)
        
        wrapper.performance_monitor = monitor
        return wrapper
    
    return decorator


def async_cache_result(cache_key_func: Optional[Callable] = None, ttl: Optional[int] = None):
    """Async decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size=1000)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = await cache_key_func(*args, **kwargs) if asyncio.iscoroutinefunction(cache_key_func) else cache_key_func(*args, **kwargs)
            else:
                key = hashlib.md5(
                    json.dumps([args, kwargs], sort_keys=True, default=str).encode()
                ).hexdigest()
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                if ttl is None or (time.time() - cached_result['timestamp']) < ttl:
                    return cached_result['value']
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.put(key, {
                'value': result,
                'timestamp': time.time()
            })
            
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


# Global performance monitor
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def set_performance_monitor(monitor: PerformanceMonitor) -> None:
    """Set global performance monitor instance."""
    global _performance_monitor
    _performance_monitor = monitor