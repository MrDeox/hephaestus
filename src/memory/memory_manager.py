"""
Memory Manager for RSI AI Systems.
Implements distributed memory coordination, caching, and optimization.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import psutil
import threading
from pathlib import Path
import json
import pickle
import gc

# Optional dependencies
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import blosc
    BLOSC_AVAILABLE = True
except ImportError:
    BLOSC_AVAILABLE = False

logger = logging.getLogger(__name__)


class RSIMemoryManager:
    """
    Memory Manager for RSI AI Systems.
    
    Provides distributed memory coordination with:
    - Redis-based distributed caching
    - Ray object store integration
    - Memory compression and optimization
    - Load balancing across memory systems
    - Performance monitoring and alerting
    """
    
    def __init__(self, ray_object_store_memory: str = "4GB",
                 redis_cluster_nodes: int = 1,
                 compression_algorithm: str = "blosc",
                 max_memory_usage_gb: int = 16):
        self.ray_object_store_memory = ray_object_store_memory
        self.redis_cluster_nodes = redis_cluster_nodes
        self.compression_algorithm = compression_algorithm
        self.max_memory_usage_gb = max_memory_usage_gb
        
        # Initialize distributed systems
        self.redis_client = None
        self.ray_initialized = False
        
        self._initialize_distributed_systems()
        
        # Memory statistics
        self.memory_stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_ratio': 0.0,
            'last_cleanup': None,
            'memory_pressure': 0.0
        }
        
        # Memory pools
        self.memory_pools = {
            'working': {'allocated': 0, 'max_size': 1024 * 1024 * 1024},  # 1GB
            'semantic': {'allocated': 0, 'max_size': 2 * 1024 * 1024 * 1024},  # 2GB
            'episodic': {'allocated': 0, 'max_size': 4 * 1024 * 1024 * 1024},  # 4GB
            'procedural': {'allocated': 0, 'max_size': 1024 * 1024 * 1024}  # 1GB
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("Memory Manager initialized")
    
    def _initialize_distributed_systems(self):
        """Initialize distributed memory systems."""
        try:
            # Initialize Redis if available
            if REDIS_AVAILABLE:
                self._initialize_redis()
            
            # Initialize Ray if available
            if RAY_AVAILABLE:
                self._initialize_ray()
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed systems: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
    
    def _initialize_ray(self):
        """Initialize Ray object store."""
        try:
            if not ray.is_initialized():
                # Parse memory string to bytes for Ray
                memory_bytes = self._parse_memory_string(self.ray_object_store_memory)
                ray.init(
                    object_store_memory=memory_bytes,
                    ignore_reinit_error=True
                )
                self.ray_initialized = True
                logger.info("✅ Ray object store initialized")
            
        except Exception as e:
            logger.warning(f"Ray not available: {e}")
            self.ray_initialized = False
    
    def _parse_memory_string(self, memory_str: str) -> int:
        """Parse memory string like '4GB' to bytes."""
        try:
            if memory_str.endswith('GB'):
                return int(memory_str[:-2]) * 1024 * 1024 * 1024
            elif memory_str.endswith('MB'):
                return int(memory_str[:-2]) * 1024 * 1024
            elif memory_str.endswith('KB'):
                return int(memory_str[:-2]) * 1024
            else:
                # Assume bytes
                return int(memory_str)
        except Exception as e:
            logger.error(f"Failed to parse memory string '{memory_str}': {e}")
            return 4 * 1024 * 1024 * 1024  # Default to 4GB
    
    def _start_monitoring(self):
        """Start memory monitoring task."""
        try:
            asyncio.create_task(self._monitoring_loop())
        except RuntimeError:
            # No event loop running
            pass
    
    async def _monitoring_loop(self):
        """Memory monitoring loop."""
        while True:
            try:
                await self._update_memory_stats()
                await self._check_memory_pressure()
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _update_memory_stats(self):
        """Update memory statistics."""
        try:
            with self.lock:
                # System memory
                memory = psutil.virtual_memory()
                
                # Calculate memory pressure
                self.memory_stats['memory_pressure'] = memory.percent / 100.0
                
                # Update pool statistics
                for pool_name, pool_info in self.memory_pools.items():
                    usage_percent = pool_info['allocated'] / pool_info['max_size'] * 100
                    pool_info['usage_percent'] = usage_percent
                
        except Exception as e:
            logger.error(f"Failed to update memory stats: {e}")
    
    async def _check_memory_pressure(self):
        """Check and respond to memory pressure."""
        try:
            if self.memory_stats['memory_pressure'] > 0.8:  # 80% memory usage
                logger.warning("High memory pressure detected, triggering cleanup")
                await self.cleanup_memory()
                
                # Force garbage collection
                gc.collect()
                
        except Exception as e:
            logger.error(f"Failed to check memory pressure: {e}")
    
    async def allocate_memory(self, memory_type: str, size: int, data: Any = None) -> Optional[str]:
        """
        Allocate memory for a specific memory type.
        
        Args:
            memory_type: Type of memory (working, semantic, episodic, procedural)
            size: Size in bytes
            data: Optional data to store
            
        Returns:
            Memory allocation ID or None if allocation failed
        """
        try:
            with self.lock:
                if memory_type not in self.memory_pools:
                    logger.error(f"Unknown memory type: {memory_type}")
                    return None
                
                pool = self.memory_pools[memory_type]
                
                # Check if allocation would exceed pool limit
                if pool['allocated'] + size > pool['max_size']:
                    logger.warning(f"Memory allocation would exceed {memory_type} pool limit")
                    
                    # Try to free some memory
                    await self._free_memory(memory_type, size)
                    
                    # Check again
                    if pool['allocated'] + size > pool['max_size']:
                        logger.error(f"Cannot allocate {size} bytes in {memory_type} pool")
                        return None
                
                # Generate allocation ID
                allocation_id = f"{memory_type}_{datetime.now(timezone.utc).timestamp()}"
                
                # Allocate memory
                pool['allocated'] += size
                self.memory_stats['total_allocations'] += 1
                
                # Store data if provided
                if data is not None:
                    await self._store_data(allocation_id, data)
                
                return allocation_id
                
        except Exception as e:
            logger.error(f"Failed to allocate memory: {e}")
            return None
    
    async def deallocate_memory(self, memory_type: str, allocation_id: str, size: int) -> bool:
        """
        Deallocate memory.
        
        Args:
            memory_type: Type of memory
            allocation_id: Allocation ID
            size: Size in bytes
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                if memory_type not in self.memory_pools:
                    return False
                
                pool = self.memory_pools[memory_type]
                
                # Deallocate memory
                pool['allocated'] = max(0, pool['allocated'] - size)
                self.memory_stats['total_deallocations'] += 1
                
                # Remove stored data
                await self._remove_data(allocation_id)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to deallocate memory: {e}")
            return False
    
    async def _free_memory(self, memory_type: str, needed_size: int):
        """Free memory in a specific pool."""
        try:
            # This would implement smart memory freeing strategies
            # For now, just a placeholder
            logger.info(f"Freeing memory in {memory_type} pool")
            
        except Exception as e:
            logger.error(f"Failed to free memory: {e}")
    
    async def _store_data(self, allocation_id: str, data: Any):
        """Store data in distributed cache."""
        try:
            # Compress data if compression is available
            if BLOSC_AVAILABLE and self.compression_algorithm == "blosc":
                serialized_data = pickle.dumps(data)
                compressed_data = blosc.compress(serialized_data)
                
                # Update compression ratio
                compression_ratio = len(compressed_data) / len(serialized_data)
                self.memory_stats['compression_ratio'] = (
                    self.memory_stats['compression_ratio'] * 0.9 + compression_ratio * 0.1
                )
                
                data_to_store = compressed_data
            else:
                data_to_store = pickle.dumps(data)
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    self.redis_client.set(
                        allocation_id,
                        data_to_store,
                        ex=3600  # 1 hour expiration
                    )
                    return
                except Exception as e:
                    logger.warning(f"Redis storage failed: {e}")
            
            # Store in Ray object store if available
            if self.ray_initialized:
                try:
                    ray.put(data_to_store)
                    return
                except Exception as e:
                    logger.warning(f"Ray storage failed: {e}")
            
            # Fallback to local storage
            await self._store_locally(allocation_id, data_to_store)
            
        except Exception as e:
            logger.error(f"Failed to store data: {e}")
    
    async def _store_locally(self, allocation_id: str, data: bytes):
        """Store data locally as fallback."""
        try:
            storage_dir = Path("./memory_cache")
            storage_dir.mkdir(exist_ok=True)
            
            file_path = storage_dir / f"{allocation_id}.pkl"
            with open(file_path, 'wb') as f:
                f.write(data)
                
        except Exception as e:
            logger.error(f"Failed to store data locally: {e}")
    
    async def retrieve_data(self, allocation_id: str) -> Optional[Any]:
        """Retrieve data from distributed cache."""
        try:
            # Try Redis first
            if self.redis_client:
                try:
                    data = self.redis_client.get(allocation_id)
                    if data:
                        self.memory_stats['cache_hits'] += 1
                        return await self._decompress_data(data)
                except Exception as e:
                    logger.warning(f"Redis retrieval failed: {e}")
            
            # Try Ray object store
            if self.ray_initialized:
                try:
                    # This would retrieve from Ray
                    # For now, just increment cache miss
                    pass
                except Exception as e:
                    logger.warning(f"Ray retrieval failed: {e}")
            
            # Try local storage
            data = await self._retrieve_locally(allocation_id)
            if data:
                self.memory_stats['cache_hits'] += 1
                return await self._decompress_data(data)
            
            self.memory_stats['cache_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve data: {e}")
            return None
    
    async def _retrieve_locally(self, allocation_id: str) -> Optional[bytes]:
        """Retrieve data from local storage."""
        try:
            storage_dir = Path("./memory_cache")
            file_path = storage_dir / f"{allocation_id}.pkl"
            
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return f.read()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve data locally: {e}")
            return None
    
    async def _decompress_data(self, data: bytes) -> Any:
        """Decompress and deserialize data."""
        try:
            if BLOSC_AVAILABLE and self.compression_algorithm == "blosc":
                try:
                    decompressed_data = blosc.decompress(data)
                    return pickle.loads(decompressed_data)
                except Exception:
                    # Fallback to direct deserialization
                    return pickle.loads(data)
            else:
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Failed to decompress data: {e}")
            return None
    
    async def _remove_data(self, allocation_id: str):
        """Remove data from all storage systems."""
        try:
            # Remove from Redis
            if self.redis_client:
                try:
                    self.redis_client.delete(allocation_id)
                except Exception as e:
                    logger.warning(f"Redis deletion failed: {e}")
            
            # Remove from local storage
            await self._remove_locally(allocation_id)
            
        except Exception as e:
            logger.error(f"Failed to remove data: {e}")
    
    async def _remove_locally(self, allocation_id: str):
        """Remove data from local storage."""
        try:
            storage_dir = Path("./memory_cache")
            file_path = storage_dir / f"{allocation_id}.pkl"
            
            if file_path.exists():
                file_path.unlink()
                
        except Exception as e:
            logger.error(f"Failed to remove data locally: {e}")
    
    async def cleanup_memory(self):
        """Clean up memory caches and optimize allocation."""
        try:
            with self.lock:
                # Clean up Redis cache
                if self.redis_client:
                    try:
                        # Remove expired keys
                        for key in self.redis_client.scan_iter(match="*"):
                            ttl = self.redis_client.ttl(key)
                            if ttl == -1:  # No expiration set
                                self.redis_client.expire(key, 3600)  # Set 1 hour expiration
                    except Exception as e:
                        logger.warning(f"Redis cleanup failed: {e}")
                
                # Clean up local cache
                await self._cleanup_local_cache()
                
                # Update last cleanup time
                self.memory_stats['last_cleanup'] = datetime.now(timezone.utc).isoformat()
                
                logger.info("Memory cleanup completed")
                
        except Exception as e:
            logger.error(f"Failed to cleanup memory: {e}")
    
    async def _cleanup_local_cache(self):
        """Clean up local cache files."""
        try:
            storage_dir = Path("./memory_cache")
            if not storage_dir.exists():
                return
            
            # Remove files older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for file_path in storage_dir.glob("*.pkl"):
                try:
                    if file_path.stat().st_mtime < cutoff_time.timestamp():
                        file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup local cache: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        with self.lock:
            system_memory = psutil.virtual_memory()
            
            usage_stats = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system_memory': {
                    'total_gb': system_memory.total / (1024**3),
                    'used_gb': system_memory.used / (1024**3),
                    'available_gb': system_memory.available / (1024**3),
                    'percent': system_memory.percent
                },
                'memory_pools': self.memory_pools.copy(),
                'statistics': self.memory_stats.copy()
            }
            
            return usage_stats
    
    def get_pool_status(self, memory_type: str) -> Dict[str, Any]:
        """Get status of a specific memory pool."""
        with self.lock:
            if memory_type not in self.memory_pools:
                return {'error': f'Unknown memory type: {memory_type}'}
            
            pool = self.memory_pools[memory_type]
            return {
                'memory_type': memory_type,
                'allocated_bytes': pool['allocated'],
                'max_size_bytes': pool['max_size'],
                'allocated_mb': pool['allocated'] / (1024**2),
                'max_size_mb': pool['max_size'] / (1024**2),
                'usage_percent': (pool['allocated'] / pool['max_size']) * 100,
                'available_bytes': pool['max_size'] - pool['allocated']
            }
    
    async def optimize(self):
        """Optimize memory management."""
        try:
            # Cleanup memory
            await self.cleanup_memory()
            
            # Optimize Redis if available
            if self.redis_client:
                try:
                    # Optimize Redis memory usage
                    self.redis_client.memory_purge()
                except Exception as e:
                    logger.warning(f"Redis optimization failed: {e}")
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        with self.lock:
            stats = self.memory_stats.copy()
            
            # Calculate additional metrics
            total_operations = stats['total_allocations'] + stats['total_deallocations']
            if total_operations > 0:
                stats['allocation_rate'] = stats['total_allocations'] / total_operations
            
            cache_operations = stats['cache_hits'] + stats['cache_misses']
            if cache_operations > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / cache_operations
            
            return stats
    
    async def store_episodic_memory(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Store episodic memory event."""
        try:
            # Create episodic memory entry
            memory_entry = {
                'event_type': event_type,
                'event_data': event_data,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'memory_id': f"episodic_{datetime.now(timezone.utc).timestamp()}"
            }
            
            # Allocate memory for episodic storage
            allocation_id = await self.allocate_memory(
                memory_type='episodic',
                size=len(pickle.dumps(memory_entry)),
                data=memory_entry
            )
            
            if allocation_id:
                logger.debug(f"Stored episodic memory: {event_type}")
                return True
            else:
                logger.warning(f"Failed to store episodic memory: {event_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store episodic memory: {e}")
            return False
    
    async def store_semantic_memory(self, concept: str, knowledge_data: Dict[str, Any]) -> bool:
        """Store semantic memory concept."""
        try:
            memory_entry = {
                'concept': concept,
                'knowledge_data': knowledge_data,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'memory_id': f"semantic_{datetime.now(timezone.utc).timestamp()}"
            }
            
            allocation_id = await self.allocate_memory(
                memory_type='semantic',
                size=len(pickle.dumps(memory_entry)),
                data=memory_entry
            )
            
            return allocation_id is not None
            
        except Exception as e:
            logger.error(f"Failed to store semantic memory: {e}")
            return False
    
    async def store_procedural_memory(self, procedure: str, procedure_data: Dict[str, Any]) -> bool:
        """Store procedural memory."""
        try:
            memory_entry = {
                'procedure': procedure,
                'procedure_data': procedure_data,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'memory_id': f"procedural_{datetime.now(timezone.utc).timestamp()}"
            }
            
            allocation_id = await self.allocate_memory(
                memory_type='procedural',
                size=len(pickle.dumps(memory_entry)),
                data=memory_entry
            )
            
            return allocation_id is not None
            
        except Exception as e:
            logger.error(f"Failed to store procedural memory: {e}")
            return False
    
    async def retrieve_episodic_memories(self, event_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve episodic memories."""
        try:
            # This is a simplified implementation
            # In a full implementation, we would search through stored memories
            logger.debug(f"Retrieving episodic memories for type: {event_type}")
            return []  # Return empty for now
            
        except Exception as e:
            logger.error(f"Failed to retrieve episodic memories: {e}")
            return []

    async def shutdown(self):
        """Shutdown memory manager."""
        try:
            # Final cleanup
            await self.cleanup_memory()
            
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
            
            # Shutdown Ray if we initialized it
            if self.ray_initialized:
                ray.shutdown()
            
            logger.info("Memory manager shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown memory manager: {e}")


# Create factory function for easy instantiation
def create_rsi_memory_manager() -> RSIMemoryManager:
    """Create RSI Memory Manager instance."""
    return RSIMemoryManager()