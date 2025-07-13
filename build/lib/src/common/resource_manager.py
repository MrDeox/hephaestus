"""
Resource Management System for Hephaestus RSI.

Provides comprehensive resource monitoring, allocation, and cleanup
with automatic recovery and leak detection.
"""

import asyncio
import gc
import os
import psutil
import threading
import time
import weakref
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, AsyncIterator
from enum import Enum

from .exceptions import (
    ResourceError, ResourceExhaustionError, MemoryError,
    create_error_context
)
from config.base_config import get_config


class ResourceType(str, Enum):
    """Types of resources managed by the system."""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    FILE_HANDLES = "file_handles"
    THREADS = "threads"
    ASYNC_TASKS = "async_tasks"
    DATABASE_CONNECTIONS = "database_connections"
    CUSTOM = "custom"


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_disk_mb: Optional[int] = None
    max_file_handles: Optional[int] = None
    max_threads: Optional[int] = None
    max_async_tasks: Optional[int] = None
    max_db_connections: Optional[int] = None
    custom_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """Current resource usage information."""
    memory_mb: float
    cpu_percent: float
    disk_mb: float
    file_handles: int
    threads: int
    async_tasks: int
    db_connections: int
    timestamp: datetime
    custom_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring."""
        return {
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "disk_mb": self.disk_mb,
            "file_handles": self.file_handles,
            "threads": self.threads,
            "async_tasks": self.async_tasks,
            "db_connections": self.db_connections,
            "timestamp": self.timestamp.isoformat(),
            "custom_usage": self.custom_usage
        }


@dataclass
class ResourceAllocation:
    """Represents a resource allocation."""
    resource_id: str
    resource_type: ResourceType
    amount: Union[int, float]
    allocated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup_callback: Optional[Callable] = None


class ResourceTracker:
    """Tracks resource allocations and usage."""
    
    def __init__(self):
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.usage_history: List[ResourceUsage] = []
        self.max_history_size = 1000
        self._lock = threading.Lock()
    
    def add_allocation(self, allocation: ResourceAllocation) -> None:
        """Add a resource allocation."""
        with self._lock:
            self.allocations[allocation.resource_id] = allocation
    
    def remove_allocation(self, resource_id: str) -> Optional[ResourceAllocation]:
        """Remove and return a resource allocation."""
        with self._lock:
            return self.allocations.pop(resource_id, None)
    
    def get_allocation(self, resource_id: str) -> Optional[ResourceAllocation]:
        """Get a resource allocation."""
        with self._lock:
            return self.allocations.get(resource_id)
    
    def get_allocations_by_type(self, resource_type: ResourceType) -> List[ResourceAllocation]:
        """Get all allocations of a specific type."""
        with self._lock:
            return [
                allocation for allocation in self.allocations.values()
                if allocation.resource_type == resource_type
            ]
    
    def add_usage_snapshot(self, usage: ResourceUsage) -> None:
        """Add a usage snapshot to history."""
        with self._lock:
            self.usage_history.append(usage)
            
            # Trim history if too large
            if len(self.usage_history) > self.max_history_size:
                self.usage_history = self.usage_history[-self.max_history_size:]
    
    def get_recent_usage(self, count: int = 10) -> List[ResourceUsage]:
        """Get recent usage snapshots."""
        with self._lock:
            return self.usage_history[-count:]
    
    def get_usage_trend(self, minutes: int = 10) -> List[ResourceUsage]:
        """Get usage trend for the last N minutes."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                usage for usage in self.usage_history
                if usage.timestamp >= cutoff_time
            ]


class ResourceManager:
    """Main resource management system."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.tracker = ResourceTracker()
        self.process = psutil.Process()
        self.monitoring_enabled = True
        self.monitoring_interval = 5.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_callbacks: Set[Callable] = set()
        self._custom_resource_trackers: Dict[str, Callable] = {}
        
        # Weak references to track managed objects
        self._managed_objects: Set[weakref.ref] = set()
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load resource limits from configuration."""
        try:
            config = get_config()
            
            # Update limits from config if available
            if hasattr(config, 'resource_limits'):
                resource_config = config.resource_limits
                
                self.limits.max_memory_mb = getattr(resource_config, 'max_memory_mb', self.limits.max_memory_mb)
                self.limits.max_cpu_percent = getattr(resource_config, 'max_cpu_percent', self.limits.max_cpu_percent)
                self.limits.max_disk_mb = getattr(resource_config, 'max_disk_mb', self.limits.max_disk_mb)
                self.limits.max_file_handles = getattr(resource_config, 'max_file_handles', self.limits.max_file_handles)
                self.limits.max_threads = getattr(resource_config, 'max_threads', self.limits.max_threads)
                
        except Exception:
            # Use defaults if config loading fails
            pass
    
    async def start_monitoring(self) -> None:
        """Start resource monitoring task."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring task."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Collect current usage
                usage = await self._collect_resource_usage()
                self.tracker.add_usage_snapshot(usage)
                
                # Check limits
                await self._check_resource_limits(usage)
                
                # Cleanup expired allocations
                await self._cleanup_expired_allocations()
                
                # Garbage collection
                await self._periodic_gc()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                print(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage."""
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Disk usage (approximate)
            disk_mb = 0
            try:
                for fd in self.process.open_files():
                    try:
                        disk_mb += os.path.getsize(fd.path) / (1024 * 1024)
                    except (OSError, AttributeError):
                        pass
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            
            # File handles
            try:
                file_handles = len(self.process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                file_handles = 0
            
            # Threads
            try:
                threads = self.process.num_threads()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                threads = 0
            
            # Async tasks
            async_tasks = len([
                task for task in asyncio.all_tasks()
                if not task.done()
            ])
            
            # Database connections (tracked allocations)
            db_connections = len(self.tracker.get_allocations_by_type(ResourceType.DATABASE_CONNECTIONS))
            
            # Custom resource usage
            custom_usage = {}
            for name, tracker_func in self._custom_resource_trackers.items():
                try:
                    custom_usage[name] = tracker_func()
                except Exception:
                    custom_usage[name] = None
            
            return ResourceUsage(
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                disk_mb=disk_mb,
                file_handles=file_handles,
                threads=threads,
                async_tasks=async_tasks,
                db_connections=db_connections,
                timestamp=datetime.now(timezone.utc),
                custom_usage=custom_usage
            )
            
        except Exception as e:
            # Return empty usage if collection fails
            return ResourceUsage(
                memory_mb=0,
                cpu_percent=0,
                disk_mb=0,
                file_handles=0,
                threads=0,
                async_tasks=0,
                db_connections=0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def _check_resource_limits(self, usage: ResourceUsage) -> None:
        """Check if resource usage exceeds limits."""
        violations = []
        
        if self.limits.max_memory_mb and usage.memory_mb > self.limits.max_memory_mb:
            violations.append(f"Memory usage ({usage.memory_mb:.1f} MB) exceeds limit ({self.limits.max_memory_mb} MB)")
        
        if self.limits.max_cpu_percent and usage.cpu_percent > self.limits.max_cpu_percent:
            violations.append(f"CPU usage ({usage.cpu_percent:.1f}%) exceeds limit ({self.limits.max_cpu_percent}%)")
        
        if self.limits.max_disk_mb and usage.disk_mb > self.limits.max_disk_mb:
            violations.append(f"Disk usage ({usage.disk_mb:.1f} MB) exceeds limit ({self.limits.max_disk_mb} MB)")
        
        if self.limits.max_file_handles and usage.file_handles > self.limits.max_file_handles:
            violations.append(f"File handles ({usage.file_handles}) exceed limit ({self.limits.max_file_handles})")
        
        if self.limits.max_threads and usage.threads > self.limits.max_threads:
            violations.append(f"Threads ({usage.threads}) exceed limit ({self.limits.max_threads})")
        
        if self.limits.max_async_tasks and usage.async_tasks > self.limits.max_async_tasks:
            violations.append(f"Async tasks ({usage.async_tasks}) exceed limit ({self.limits.max_async_tasks})")
        
        if violations:
            # Try automatic cleanup first
            await self._emergency_cleanup()
            
            # Re-check after cleanup
            updated_usage = await self._collect_resource_usage()
            remaining_violations = []
            
            if self.limits.max_memory_mb and updated_usage.memory_mb > self.limits.max_memory_mb:
                remaining_violations.append(f"Memory usage still high after cleanup")
            
            if remaining_violations:
                raise ResourceExhaustionError(
                    f"Resource limits exceeded: {'; '.join(violations)}",
                    resource_type="multiple",
                    current_usage=usage.to_dict(),
                    context=create_error_context("resource_limit_check")
                )
    
    async def _cleanup_expired_allocations(self) -> None:
        """Clean up expired or abandoned resource allocations."""
        current_time = datetime.now(timezone.utc)
        expired_allocations = []
        
        # Find allocations older than 1 hour without recent access
        cutoff_time = current_time - timedelta(hours=1)
        
        for resource_id, allocation in list(self.tracker.allocations.items()):
            if allocation.allocated_at < cutoff_time:
                # Check if allocation has cleanup callback
                if allocation.cleanup_callback:
                    try:
                        if asyncio.iscoroutinefunction(allocation.cleanup_callback):
                            await allocation.cleanup_callback()
                        else:
                            allocation.cleanup_callback()
                    except Exception as e:
                        print(f"Error during cleanup callback for {resource_id}: {e}")
                
                expired_allocations.append(resource_id)
        
        # Remove expired allocations
        for resource_id in expired_allocations:
            self.tracker.remove_allocation(resource_id)
    
    async def _emergency_cleanup(self) -> None:
        """Perform emergency cleanup when resources are exhausted."""
        # Force garbage collection
        gc.collect()
        
        # Cancel non-essential async tasks
        for task in asyncio.all_tasks():
            if (task != asyncio.current_task() and 
                not task.done() and 
                hasattr(task, '_hephaestus_essential') and 
                not task._hephaestus_essential):
                task.cancel()
        
        # Execute cleanup callbacks
        cleanup_callbacks = list(self._cleanup_callbacks)
        for callback in cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                print(f"Error during emergency cleanup: {e}")
        
        # Clean up weak references
        self._cleanup_weak_references()
    
    async def _periodic_gc(self) -> None:
        """Perform periodic garbage collection."""
        # Run garbage collection every 10 monitoring cycles
        if hasattr(self, '_gc_counter'):
            self._gc_counter += 1
        else:
            self._gc_counter = 0
        
        if self._gc_counter >= 10:
            gc.collect()
            self._gc_counter = 0
            
            # Clean up weak references
            self._cleanup_weak_references()
    
    def _cleanup_weak_references(self) -> None:
        """Clean up dead weak references."""
        alive_refs = set()
        for ref in self._managed_objects:
            if ref() is not None:
                alive_refs.add(ref)
        self._managed_objects = alive_refs
    
    # Resource allocation methods
    
    def allocate_resource(
        self,
        resource_type: ResourceType,
        amount: Union[int, float],
        metadata: Optional[Dict[str, Any]] = None,
        cleanup_callback: Optional[Callable] = None
    ) -> str:
        """Allocate a resource and return allocation ID."""
        import uuid
        
        resource_id = str(uuid.uuid4())
        allocation = ResourceAllocation(
            resource_id=resource_id,
            resource_type=resource_type,
            amount=amount,
            allocated_at=datetime.now(timezone.utc),
            metadata=metadata or {},
            cleanup_callback=cleanup_callback
        )
        
        self.tracker.add_allocation(allocation)
        return resource_id
    
    def deallocate_resource(self, resource_id: str) -> bool:
        """Deallocate a resource."""
        allocation = self.tracker.remove_allocation(resource_id)
        if allocation and allocation.cleanup_callback:
            try:
                allocation.cleanup_callback()
            except Exception as e:
                print(f"Error during deallocation cleanup: {e}")
        
        return allocation is not None
    
    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a cleanup callback for emergency situations."""
        self._cleanup_callbacks.add(callback)
    
    def unregister_cleanup_callback(self, callback: Callable) -> None:
        """Unregister a cleanup callback."""
        self._cleanup_callbacks.discard(callback)
    
    def register_custom_tracker(self, name: str, tracker_func: Callable) -> None:
        """Register a custom resource tracker function."""
        self._custom_resource_trackers[name] = tracker_func
    
    def track_object(self, obj: Any) -> None:
        """Track an object for cleanup."""
        weak_ref = weakref.ref(obj)
        self._managed_objects.add(weak_ref)
    
    # Context managers
    
    @contextmanager
    def allocate_memory(self, size_mb: int):
        """Context manager for memory allocation."""
        resource_id = self.allocate_resource(ResourceType.MEMORY, size_mb)
        try:
            yield resource_id
        finally:
            self.deallocate_resource(resource_id)
    
    @asynccontextmanager
    async def allocate_async_resource(
        self,
        resource_type: ResourceType,
        amount: Union[int, float],
        cleanup_func: Optional[Callable] = None
    ):
        """Async context manager for resource allocation."""
        resource_id = self.allocate_resource(
            resource_type, 
            amount, 
            cleanup_callback=cleanup_func
        )
        try:
            yield resource_id
        finally:
            self.deallocate_resource(resource_id)
    
    # Monitoring and reporting
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage synchronously."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._collect_resource_usage())
        except RuntimeError:
            # No event loop, create a new one
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._collect_resource_usage())
            finally:
                loop.close()
    
    def get_usage_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get usage summary for the last N minutes."""
        trend_data = self.tracker.get_usage_trend(minutes)
        
        if not trend_data:
            return {"error": "No usage data available"}
        
        # Calculate statistics
        memory_values = [usage.memory_mb for usage in trend_data]
        cpu_values = [usage.cpu_percent for usage in trend_data]
        
        return {
            "time_period_minutes": minutes,
            "sample_count": len(trend_data),
            "memory": {
                "current_mb": memory_values[-1] if memory_values else 0,
                "average_mb": sum(memory_values) / len(memory_values) if memory_values else 0,
                "peak_mb": max(memory_values) if memory_values else 0,
                "limit_mb": self.limits.max_memory_mb
            },
            "cpu": {
                "current_percent": cpu_values[-1] if cpu_values else 0,
                "average_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "peak_percent": max(cpu_values) if cpu_values else 0,
                "limit_percent": self.limits.max_cpu_percent
            },
            "allocations": {
                "total_count": len(self.tracker.allocations),
                "by_type": {
                    resource_type.value: len(self.tracker.get_allocations_by_type(resource_type))
                    for resource_type in ResourceType
                }
            }
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Check resource manager health."""
        current_usage = self.get_current_usage()
        
        health_status = "healthy"
        issues = []
        
        # Check for resource limit violations
        if self.limits.max_memory_mb and current_usage.memory_mb > self.limits.max_memory_mb * 0.9:
            health_status = "warning"
            issues.append(f"Memory usage at {current_usage.memory_mb:.1f} MB (90% of limit)")
        
        if self.limits.max_cpu_percent and current_usage.cpu_percent > self.limits.max_cpu_percent * 0.9:
            health_status = "warning"
            issues.append(f"CPU usage at {current_usage.cpu_percent:.1f}% (90% of limit)")
        
        # Check for memory violations (critical)
        if self.limits.max_memory_mb and current_usage.memory_mb > self.limits.max_memory_mb:
            health_status = "critical"
        
        return {
            "status": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_usage": current_usage.to_dict(),
            "limits": {
                "max_memory_mb": self.limits.max_memory_mb,
                "max_cpu_percent": self.limits.max_cpu_percent,
                "max_disk_mb": self.limits.max_disk_mb,
                "max_file_handles": self.limits.max_file_handles,
                "max_threads": self.limits.max_threads,
            },
            "issues": issues,
            "monitoring_enabled": self.monitoring_enabled,
            "allocations_count": len(self.tracker.allocations)
        }


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def set_resource_manager(manager: ResourceManager) -> None:
    """Set global resource manager instance."""
    global _resource_manager
    _resource_manager = manager