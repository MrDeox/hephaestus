"""
RSI Memory Hierarchy Implementation.
Implements hierarchical memory architecture with cognitive science principles.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from dataclasses import dataclass
import psutil
import ray

from .working_memory import WorkingMemory
from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory

# Import new components with error handling
try:
    from .procedural_memory import ProceduralMemory
except ImportError as e:
    logger.error(f"Failed to import ProceduralMemory: {e}")
    ProceduralMemory = None

try:
    from .memory_consolidation import MemoryConsolidation
except ImportError as e:
    logger.error(f"Failed to import MemoryConsolidation: {e}")
    MemoryConsolidation = None

try:
    from .retrieval_engine import RetrievalEngine
except ImportError as e:
    logger.error(f"Failed to import RetrievalEngine: {e}")
    RetrievalEngine = None

try:
    from .memory_manager import MemoryManager
except ImportError as e:
    logger.error(f"Failed to import MemoryManager: {e}")
    MemoryManager = None

logger = logging.getLogger(__name__)


class RSIMemoryConfig(BaseModel):
    """Configuration for RSI Memory System."""
    
    # Memory Architecture
    working_memory_capacity: int = Field(default=10000, description="Working memory capacity")
    semantic_memory_backend: str = Field(default="networkx", description="Semantic memory backend")
    episodic_memory_backend: str = Field(default="eventsourcing", description="Episodic memory backend")
    
    # Knowledge Representation
    vector_db_type: str = Field(default="chroma", description="Vector database type")
    graph_db_type: str = Field(default="networkx", description="Graph database type")
    embedding_dimension: int = Field(default=768, description="Embedding dimension")
    
    # Continual Learning
    replay_buffer_size: int = Field(default=5000, description="Replay buffer size")
    ewc_lambda: float = Field(default=0.4, description="EWC lambda parameter")
    meta_learning_enabled: bool = Field(default=True, description="Enable meta-learning")
    
    # Retrieval Systems
    ann_algorithm: str = Field(default="hnsw", description="ANN algorithm")
    index_ef_construction: int = Field(default=200, description="HNSW ef construction")
    index_m: int = Field(default=16, description="HNSW M parameter")
    
    # Memory Management
    ray_object_store_memory: str = Field(default="4GB", description="Ray object store memory")
    redis_cluster_nodes: int = Field(default=1, description="Redis cluster nodes")
    compression_algorithm: str = Field(default="blosc", description="Compression algorithm")
    
    # Performance Tuning
    max_memory_usage_gb: int = Field(default=16, description="Max memory usage in GB")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
    
    # Safety Constraints
    max_working_memory_size: int = Field(default=50000, description="Max working memory size")
    memory_consolidation_threshold: float = Field(default=0.8, description="Memory consolidation threshold")
    automatic_cleanup_enabled: bool = Field(default=True, description="Enable automatic cleanup")


class RSIMemoryHierarchy:
    """
    Advanced Memory Hierarchy for RSI AI Systems.
    
    Implements a cognitive architecture with multiple memory systems:
    - L1: Working Memory (immediate access, temporary storage)
    - L2: Semantic Memory (structured knowledge, concepts)
    - L3: Episodic Memory (temporal sequences, experiences)
    - L4: Procedural Memory (skills, actions, procedures)
    """
    
    def __init__(self, config: RSIMemoryConfig):
        self.config = config
        self.is_initialized = False
        self.memory_usage_stats = {}
        
        # Initialize memory systems
        logger.info("Initializing RSI Memory Hierarchy...")
        self._initialize_memory_systems()
        
        # Initialize support systems
        self._initialize_support_systems()
        
        # Start monitoring if enabled
        if config.monitoring_enabled:
            self._start_monitoring()
        
        self.is_initialized = True
        logger.info("RSI Memory Hierarchy initialized successfully")
    
    def _initialize_memory_systems(self):
        """Initialize all memory systems."""
        try:
            # L1: Working Memory (immediate access)
            self.working_memory = WorkingMemory(
                capacity=self.config.working_memory_capacity,
                max_size=self.config.max_working_memory_size
            )
            
            # L2: Semantic Memory (structured knowledge)
            self.semantic_memory = SemanticMemory(
                backend=self.config.semantic_memory_backend,
                graph_db_type=self.config.graph_db_type,
                embedding_dimension=self.config.embedding_dimension
            )
            
            # L3: Episodic Memory (temporal sequences)
            self.episodic_memory = EpisodicMemory(
                backend=self.config.episodic_memory_backend
            )
            
            # L4: Procedural Memory (skills and actions)
            if ProceduralMemory is not None:
                self.procedural_memory = ProceduralMemory()
            else:
                self.procedural_memory = None
                logger.warning("ProceduralMemory not available")
            
            logger.info("✅ Memory systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory systems: {e}")
            raise
    
    def _initialize_support_systems(self):
        """Initialize support systems for memory management."""
        try:
            # Retrieval System
            if RetrievalEngine is not None:
                self.retrieval_engine = RetrievalEngine(
                    vector_db_type=self.config.vector_db_type,
                    ann_algorithm=self.config.ann_algorithm,
                    index_ef_construction=self.config.index_ef_construction,
                    index_m=self.config.index_m
                )
            else:
                self.retrieval_engine = None
                logger.warning("RetrievalEngine not available")
            
            # Memory Consolidation
            if MemoryConsolidation is not None:
                self.memory_consolidation = MemoryConsolidation(
                    working_memory=self.working_memory,
                    semantic_memory=self.semantic_memory,
                    episodic_memory=self.episodic_memory,
                    procedural_memory=self.procedural_memory,
                    consolidation_threshold=self.config.memory_consolidation_threshold
                )
            else:
                self.memory_consolidation = None
                logger.warning("MemoryConsolidation not available")
            
            # Memory Manager
            if MemoryManager is not None:
                self.memory_manager = MemoryManager(
                    ray_object_store_memory=self.config.ray_object_store_memory,
                    redis_cluster_nodes=self.config.redis_cluster_nodes,
                    compression_algorithm=self.config.compression_algorithm,
                    max_memory_usage_gb=self.config.max_memory_usage_gb
                )
            else:
                self.memory_manager = None
                logger.warning("MemoryManager not available")
            
            logger.info("✅ Support systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize support systems: {e}")
            raise
    
    def _start_monitoring(self):
        """Start memory monitoring."""
        try:
            # Start background monitoring task
            asyncio.create_task(self._monitoring_loop())
            logger.info("✅ Memory monitoring started")
        except Exception as e:
            logger.warning(f"Failed to start monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Update memory usage statistics
                self._update_memory_stats()
                
                # Check for consolidation needs
                if self._should_consolidate():
                    await self.consolidate_memory()
                
                # Check for cleanup needs
                if self._should_cleanup():
                    await self.cleanup_memory()
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _update_memory_stats(self):
        """Update memory usage statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Working memory stats
            working_memory_size = self.working_memory.get_size()
            working_memory_usage = working_memory_size / self.config.working_memory_capacity
            
            # Update stats
            self.memory_usage_stats = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system_memory_percent': memory.percent,
                'system_memory_used_gb': memory.used / (1024**3),
                'working_memory_size': working_memory_size,
                'working_memory_usage_percent': working_memory_usage * 100,
                'semantic_memory_size': self.semantic_memory.get_size(),
                'episodic_memory_size': self.episodic_memory.get_size(),
                'procedural_memory_size': self.procedural_memory.get_size() if self.procedural_memory else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to update memory stats: {e}")
    
    def _should_consolidate(self) -> bool:
        """Check if memory consolidation is needed."""
        try:
            working_memory_usage = (
                self.working_memory.get_size() / self.config.working_memory_capacity
            )
            return working_memory_usage > self.config.memory_consolidation_threshold
        except Exception:
            return False
    
    def _should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        try:
            if not self.config.automatic_cleanup_enabled:
                return False
            
            # Check system memory pressure
            memory = psutil.virtual_memory()
            return memory.percent > 85.0  # Cleanup if system memory > 85%
        except Exception:
            return False
    
    async def store_information(self, information: Dict[str, Any], memory_type: str = "auto") -> bool:
        """
        Store information in appropriate memory system.
        
        Args:
            information: Information to store
            memory_type: Target memory type or "auto" for automatic routing
            
        Returns:
            Success status
        """
        try:
            # First, store in working memory
            await self.working_memory.store(information)
            
            # Route to appropriate long-term memory if specified
            if memory_type == "semantic":
                await self.semantic_memory.store_concept(information)
            elif memory_type == "episodic":
                await self.episodic_memory.record_episode(information)
            elif memory_type == "procedural":
                if self.procedural_memory is not None:
                    await self.procedural_memory.store_skill(information)
                else:
                    logger.warning("ProceduralMemory not available for storage")
            elif memory_type == "auto":
                # Automatic routing based on information characteristics
                if self.memory_consolidation is not None:
                    await self.memory_consolidation.route_information(information)
                else:
                    logger.warning("MemoryConsolidation not available for auto routing")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store information: {e}")
            return False
    
    async def retrieve_information(self, query: Dict[str, Any], memory_types: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve information from memory systems.
        
        Args:
            query: Search query
            memory_types: List of memory types to search (default: all)
            
        Returns:
            Retrieved information
        """
        try:
            if memory_types is None:
                memory_types = ["working", "semantic", "episodic", "procedural"]
            
            results = {}
            
            # Search working memory
            if "working" in memory_types:
                results["working"] = await self.working_memory.retrieve(query)
            
            # Search semantic memory
            if "semantic" in memory_types:
                results["semantic"] = await self.semantic_memory.retrieve_concepts(query)
            
            # Search episodic memory
            if "episodic" in memory_types:
                results["episodic"] = await self.episodic_memory.retrieve_episodes(query)
            
            # Search procedural memory
            if "procedural" in memory_types:
                if self.procedural_memory is not None:
                    results["procedural"] = await self.procedural_memory.retrieve_skills(query)
                else:
                    results["procedural"] = []
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve information: {e}")
            return {}
    
    async def consolidate_memory(self) -> bool:
        """
        Consolidate memory by moving information from working memory to long-term storage.
        
        Returns:
            Success status
        """
        try:
            logger.info("Starting memory consolidation...")
            
            # Run consolidation process
            if self.memory_consolidation is not None:
                results = await self.memory_consolidation.consolidate_batch()
            else:
                results = {"message": "MemoryConsolidation not available"}
            
            # Log results
            logger.info(f"Memory consolidation completed: {results}")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return False
    
    async def cleanup_memory(self) -> bool:
        """
        Clean up memory by removing old or irrelevant information.
        
        Returns:
            Success status
        """
        try:
            logger.info("Starting memory cleanup...")
            
            # Cleanup working memory
            await self.working_memory.cleanup()
            
            # Cleanup semantic memory
            await self.semantic_memory.cleanup()
            
            # Cleanup episodic memory
            await self.episodic_memory.cleanup()
            
            # Cleanup procedural memory
            if self.procedural_memory is not None:
                await self.procedural_memory.cleanup()
            
            logger.info("Memory cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    async def get_memory_status(self) -> Dict[str, Any]:
        """
        Get comprehensive memory system status.
        
        Returns:
            Memory status information
        """
        try:
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'is_initialized': self.is_initialized,
                'config': self.config.model_dump(),
                'memory_usage_stats': self.memory_usage_stats,
                'memory_systems': {
                    'working_memory': {
                        'size': self.working_memory.get_size(),
                        'capacity': self.config.working_memory_capacity,
                        'usage_percent': (self.working_memory.get_size() / self.config.working_memory_capacity) * 100
                    },
                    'semantic_memory': {
                        'size': self.semantic_memory.get_size(),
                        'concepts_count': await self.semantic_memory.get_concepts_count()
                    },
                    'episodic_memory': {
                        'size': self.episodic_memory.get_size(),
                        'episodes_count': await self.episodic_memory.get_episodes_count()
                    },
                    'procedural_memory': {
                        'size': self.procedural_memory.get_size() if self.procedural_memory else 0,
                        'skills_count': await self.procedural_memory.get_skills_count() if self.procedural_memory else 0
                    }
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get memory status: {e}")
            return {'error': str(e)}
    
    async def optimize_memory(self) -> bool:
        """
        Optimize memory systems for better performance.
        
        Returns:
            Success status
        """
        try:
            logger.info("Starting memory optimization...")
            
            # Optimize retrieval engine
            if self.retrieval_engine is not None:
                await self.retrieval_engine.optimize()
            
            # Optimize memory manager
            if self.memory_manager is not None:
                await self.memory_manager.optimize()
            
            # Optimize individual memory systems
            await self.semantic_memory.optimize()
            await self.episodic_memory.optimize()
            if self.procedural_memory is not None:
                await self.procedural_memory.optimize()
            
            logger.info("Memory optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown memory systems."""
        try:
            logger.info("Shutting down memory systems...")
            
            # Final consolidation
            await self.consolidate_memory()
            
            # Shutdown individual systems
            await self.working_memory.shutdown()
            await self.semantic_memory.shutdown()
            await self.episodic_memory.shutdown()
            if self.procedural_memory is not None:
                await self.procedural_memory.shutdown()
            if self.memory_manager is not None:
                await self.memory_manager.shutdown()
            
            logger.info("Memory systems shut down successfully")
            
        except Exception as e:
            logger.error(f"Failed to shutdown memory systems: {e}")


def create_memory_hierarchy(config: Optional[RSIMemoryConfig] = None) -> RSIMemoryHierarchy:
    """
    Factory function to create RSI Memory Hierarchy.
    
    Args:
        config: Optional configuration, uses defaults if not provided
        
    Returns:
        Configured RSI Memory Hierarchy
    """
    if config is None:
        config = RSIMemoryConfig()
    
    return RSIMemoryHierarchy(config)