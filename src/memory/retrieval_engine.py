"""
Retrieval Engine for RSI Memory Systems.
Implements unified cross-memory search and retrieval capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
from enum import Enum
import threading
import heapq

# Optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Types of search operations."""
    EXACT = "exact"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    PROCEDURAL = "procedural"
    CROSS_MODAL = "cross_modal"


@dataclass
class SearchResult:
    """Search result with metadata."""
    content: Dict[str, Any]
    memory_type: str
    relevance_score: float
    similarity_score: float
    timestamp: datetime
    metadata: Dict[str, Any]


class RetrievalEngine:
    """
    Unified Retrieval Engine for RSI Memory Systems.
    
    Provides intelligent search and retrieval across all memory systems:
    - Semantic similarity search
    - Temporal pattern matching
    - Cross-modal retrieval
    - Relevance ranking
    - Query expansion
    """
    
    def __init__(self, vector_db_type: str = "faiss", ann_algorithm: str = "hnsw",
                 index_ef_construction: int = 200, index_m: int = 16):
        self.vector_db_type = vector_db_type
        self.ann_algorithm = ann_algorithm
        self.index_ef_construction = index_ef_construction
        self.index_m = index_m
        
        # Initialize vector indices
        self.vector_indices = {}
        self.embedding_dimension = 768
        
        # Initialize ANN index
        self._initialize_ann_index()
        
        # Search statistics
        self.stats = {
            'total_searches': 0,
            'semantic_searches': 0,
            'temporal_searches': 0,
            'procedural_searches': 0,
            'cross_modal_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_latency': 0.0
        }
        
        # Result cache
        self.result_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Retrieval Engine initialized with {vector_db_type} backend")
    
    def _initialize_ann_index(self):
        """Initialize Approximate Nearest Neighbor index."""
        try:
            if self.ann_algorithm == "hnsw" and HNSWLIB_AVAILABLE:
                self.ann_index = hnswlib.Index(space='cosine', dim=self.embedding_dimension)
                self.ann_index.init_index(
                    max_elements=100000,
                    ef_construction=self.index_ef_construction,
                    M=self.index_m
                )
                logger.info("✅ HNSW index initialized")
                
            elif self.vector_db_type == "faiss" and FAISS_AVAILABLE:
                self.ann_index = faiss.IndexFlatIP(self.embedding_dimension)
                logger.info("✅ FAISS index initialized")
                
            else:
                # Fallback to simple linear search
                self.ann_index = None
                logger.warning("Using fallback linear search (no ANN library available)")
                
        except Exception as e:
            logger.error(f"Failed to initialize ANN index: {e}")
            self.ann_index = None
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (placeholder implementation)."""
        # This would use a real embedding model in production
        import hashlib
        hash_value = hashlib.sha256(text.encode()).hexdigest()
        
        # Convert hash to embedding
        embedding = np.array([
            ord(char) / 255.0 for char in hash_value[:self.embedding_dimension]
        ], dtype=np.float32)
        
        # Pad or truncate to desired dimension
        if len(embedding) < self.embedding_dimension:
            embedding = np.pad(embedding, (0, self.embedding_dimension - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dimension]
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cache_key(self, query: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        import json
        import hashlib
        query_str = json.dumps(query, sort_keys=True, default=str)
        return hashlib.sha256(query_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        try:
            cache_time = datetime.fromisoformat(cache_entry['timestamp'])
            return (datetime.now(timezone.utc) - cache_time).total_seconds() < self.cache_ttl
        except Exception:
            return False
    
    async def search(self, query: Dict[str, Any], memory_systems: List[str] = None,
                    search_type: SearchType = SearchType.SEMANTIC,
                    max_results: int = 10) -> List[SearchResult]:
        """
        Unified search across memory systems.
        
        Args:
            query: Search query
            memory_systems: List of memory systems to search
            search_type: Type of search to perform
            max_results: Maximum number of results
            
        Returns:
            Ranked search results
        """
        try:
            with self.lock:
                search_start = datetime.now(timezone.utc)
                
                # Check cache first
                cache_key = self._cache_key({
                    'query': query,
                    'memory_systems': memory_systems,
                    'search_type': search_type.value,
                    'max_results': max_results
                })
                
                if cache_key in self.result_cache:
                    cache_entry = self.result_cache[cache_key]
                    if self._is_cache_valid(cache_entry):
                        self.stats['cache_hits'] += 1
                        return cache_entry['results']
                
                self.stats['cache_misses'] += 1
                
                # Perform search based on type
                if search_type == SearchType.SEMANTIC:
                    results = await self._semantic_search(query, memory_systems, max_results)
                elif search_type == SearchType.TEMPORAL:
                    results = await self._temporal_search(query, memory_systems, max_results)
                elif search_type == SearchType.PROCEDURAL:
                    results = await self._procedural_search(query, memory_systems, max_results)
                elif search_type == SearchType.CROSS_MODAL:
                    results = await self._cross_modal_search(query, memory_systems, max_results)
                else:
                    results = await self._exact_search(query, memory_systems, max_results)
                
                # Cache results
                self.result_cache[cache_key] = {
                    'results': results,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Update statistics
                search_duration = (datetime.now(timezone.utc) - search_start).total_seconds()
                self.stats['total_searches'] += 1
                self.stats['average_latency'] = (
                    (self.stats['average_latency'] * (self.stats['total_searches'] - 1) + search_duration) /
                    self.stats['total_searches']
                )
                
                if search_type == SearchType.SEMANTIC:
                    self.stats['semantic_searches'] += 1
                elif search_type == SearchType.TEMPORAL:
                    self.stats['temporal_searches'] += 1
                elif search_type == SearchType.PROCEDURAL:
                    self.stats['procedural_searches'] += 1
                elif search_type == SearchType.CROSS_MODAL:
                    self.stats['cross_modal_searches'] += 1
                
                return results
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _semantic_search(self, query: Dict[str, Any], memory_systems: List[str],
                             max_results: int) -> List[SearchResult]:
        """Perform semantic similarity search."""
        try:
            all_results = []
            
            # Extract search text
            search_text = query.get('text', query.get('query', ''))
            if not search_text:
                return []
            
            # Generate query embedding
            query_embedding = self._generate_embedding(search_text)
            
            # Search each memory system
            if not memory_systems:
                memory_systems = ['working', 'semantic', 'episodic', 'procedural']
            
            for memory_type in memory_systems:
                try:
                    # This would interface with actual memory systems
                    # For now, return mock results
                    mock_results = await self._mock_memory_search(memory_type, query, max_results)
                    
                    for result in mock_results:
                        # Calculate similarity score
                        result_text = str(result.get('content', ''))
                        result_embedding = self._generate_embedding(result_text)
                        
                        similarity = np.dot(query_embedding, result_embedding)
                        
                        search_result = SearchResult(
                            content=result,
                            memory_type=memory_type,
                            relevance_score=similarity,
                            similarity_score=similarity,
                            timestamp=datetime.now(timezone.utc),
                            metadata={'search_type': 'semantic'}
                        )
                        
                        all_results.append(search_result)
                        
                except Exception as e:
                    logger.error(f"Failed to search {memory_type} memory: {e}")
            
            # Rank and return top results
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _temporal_search(self, query: Dict[str, Any], memory_systems: List[str],
                             max_results: int) -> List[SearchResult]:
        """Perform temporal pattern search."""
        try:
            all_results = []
            
            # Extract temporal query parameters
            time_range = query.get('time_range', {})
            temporal_pattern = query.get('temporal_pattern', '')
            
            # Focus on episodic memory for temporal searches
            if not memory_systems:
                memory_systems = ['episodic', 'working']
            
            for memory_type in memory_systems:
                try:
                    # This would interface with actual memory systems
                    mock_results = await self._mock_memory_search(memory_type, query, max_results)
                    
                    for result in mock_results:
                        # Calculate temporal relevance
                        temporal_score = self._calculate_temporal_relevance(result, time_range, temporal_pattern)
                        
                        search_result = SearchResult(
                            content=result,
                            memory_type=memory_type,
                            relevance_score=temporal_score,
                            similarity_score=temporal_score,
                            timestamp=datetime.now(timezone.utc),
                            metadata={'search_type': 'temporal'}
                        )
                        
                        all_results.append(search_result)
                        
                except Exception as e:
                    logger.error(f"Failed temporal search in {memory_type} memory: {e}")
            
            # Rank and return top results
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            return []
    
    async def _procedural_search(self, query: Dict[str, Any], memory_systems: List[str],
                               max_results: int) -> List[SearchResult]:
        """Perform procedural/skill-based search."""
        try:
            all_results = []
            
            # Extract procedural query parameters
            skill_type = query.get('skill_type', '')
            success_threshold = query.get('success_threshold', 0.0)
            
            # Focus on procedural memory
            if not memory_systems:
                memory_systems = ['procedural', 'working']
            
            for memory_type in memory_systems:
                try:
                    mock_results = await self._mock_memory_search(memory_type, query, max_results)
                    
                    for result in mock_results:
                        # Calculate procedural relevance
                        procedural_score = self._calculate_procedural_relevance(
                            result, skill_type, success_threshold
                        )
                        
                        search_result = SearchResult(
                            content=result,
                            memory_type=memory_type,
                            relevance_score=procedural_score,
                            similarity_score=procedural_score,
                            timestamp=datetime.now(timezone.utc),
                            metadata={'search_type': 'procedural'}
                        )
                        
                        all_results.append(search_result)
                        
                except Exception as e:
                    logger.error(f"Failed procedural search in {memory_type} memory: {e}")
            
            # Rank and return top results
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Procedural search failed: {e}")
            return []
    
    async def _cross_modal_search(self, query: Dict[str, Any], memory_systems: List[str],
                                max_results: int) -> List[SearchResult]:
        """Perform cross-modal search across different memory types."""
        try:
            all_results = []
            
            # Search all memory systems
            if not memory_systems:
                memory_systems = ['working', 'semantic', 'episodic', 'procedural']
            
            # Perform semantic search first
            semantic_results = await self._semantic_search(query, memory_systems, max_results)
            
            # Expand search based on semantic results
            for result in semantic_results:
                # Extract concepts from result
                concepts = self._extract_concepts(result.content)
                
                # Search for related information in other memory systems
                for concept in concepts:
                    expanded_query = {'text': concept, 'limit': 3}
                    
                    for memory_type in memory_systems:
                        if memory_type != result.memory_type:
                            try:
                                related_results = await self._mock_memory_search(
                                    memory_type, expanded_query, 3
                                )
                                
                                for related_result in related_results:
                                    cross_modal_score = result.relevance_score * 0.8  # Decay factor
                                    
                                    search_result = SearchResult(
                                        content=related_result,
                                        memory_type=memory_type,
                                        relevance_score=cross_modal_score,
                                        similarity_score=cross_modal_score,
                                        timestamp=datetime.now(timezone.utc),
                                        metadata={
                                            'search_type': 'cross_modal',
                                            'source_memory': result.memory_type,
                                            'bridge_concept': concept
                                        }
                                    )
                                    
                                    all_results.append(search_result)
                                    
                            except Exception as e:
                                logger.error(f"Failed cross-modal search in {memory_type}: {e}")
            
            # Add original results
            all_results.extend(semantic_results)
            
            # Remove duplicates and rank
            seen = set()
            unique_results = []
            for result in all_results:
                result_id = str(result.content)
                if result_id not in seen:
                    seen.add(result_id)
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return unique_results[:max_results]
            
        except Exception as e:
            logger.error(f"Cross-modal search failed: {e}")
            return []
    
    async def _exact_search(self, query: Dict[str, Any], memory_systems: List[str],
                          max_results: int) -> List[SearchResult]:
        """Perform exact match search."""
        try:
            all_results = []
            
            if not memory_systems:
                memory_systems = ['working', 'semantic', 'episodic', 'procedural']
            
            for memory_type in memory_systems:
                try:
                    mock_results = await self._mock_memory_search(memory_type, query, max_results)
                    
                    for result in mock_results:
                        # Exact match scoring
                        exact_score = 1.0 if self._is_exact_match(result, query) else 0.0
                        
                        if exact_score > 0:
                            search_result = SearchResult(
                                content=result,
                                memory_type=memory_type,
                                relevance_score=exact_score,
                                similarity_score=exact_score,
                                timestamp=datetime.now(timezone.utc),
                                metadata={'search_type': 'exact'}
                            )
                            
                            all_results.append(search_result)
                            
                except Exception as e:
                    logger.error(f"Failed exact search in {memory_type} memory: {e}")
            
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Exact search failed: {e}")
            return []
    
    async def _mock_memory_search(self, memory_type: str, query: Dict[str, Any],
                                max_results: int) -> List[Dict[str, Any]]:
        """Mock memory search for testing purposes."""
        # This would be replaced with actual memory system interfaces
        mock_results = []
        
        for i in range(min(max_results, 5)):  # Mock up to 5 results
            mock_result = {
                'id': f"{memory_type}_{i}",
                'content': f"Mock {memory_type} content {i}",
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'memory_type': memory_type,
                'relevance': 0.8 - (i * 0.1)
            }
            mock_results.append(mock_result)
        
        return mock_results
    
    def _calculate_temporal_relevance(self, result: Dict[str, Any], time_range: Dict[str, Any],
                                    temporal_pattern: str) -> float:
        """Calculate temporal relevance score."""
        try:
            # Simple temporal scoring
            score = 0.5  # Base score
            
            # Check if result has timestamp
            if 'timestamp' in result:
                # Add scoring based on time range match
                if time_range:
                    # Simple time range matching
                    score += 0.3
            
            # Check for temporal pattern match
            if temporal_pattern and temporal_pattern in str(result):
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Failed to calculate temporal relevance: {e}")
            return 0.0
    
    def _calculate_procedural_relevance(self, result: Dict[str, Any], skill_type: str,
                                      success_threshold: float) -> float:
        """Calculate procedural relevance score."""
        try:
            score = 0.5  # Base score
            
            # Check skill type match
            if skill_type and result.get('skill_type') == skill_type:
                score += 0.3
            
            # Check success rate
            success_rate = result.get('success_rate', 0.0)
            if success_rate >= success_threshold:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Failed to calculate procedural relevance: {e}")
            return 0.0
    
    def _extract_concepts(self, content: Dict[str, Any]) -> List[str]:
        """Extract key concepts from content."""
        try:
            concepts = []
            
            # Simple concept extraction
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, str) and len(value) > 3:
                        concepts.append(value)
            elif isinstance(content, str):
                # Extract words longer than 3 characters
                words = content.split()
                concepts = [word for word in words if len(word) > 3]
            
            return concepts[:5]  # Return top 5 concepts
            
        except Exception as e:
            logger.error(f"Failed to extract concepts: {e}")
            return []
    
    def _is_exact_match(self, result: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Check if result is an exact match for query."""
        try:
            # Simple exact matching
            query_text = query.get('text', query.get('query', ''))
            result_text = str(result.get('content', ''))
            
            return query_text.lower() in result_text.lower()
            
        except Exception as e:
            logger.error(f"Failed to check exact match: {e}")
            return False
    
    async def expand_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Expand query with related terms and concepts."""
        try:
            expanded_query = query.copy()
            
            # Simple query expansion
            query_text = query.get('text', query.get('query', ''))
            if query_text:
                # Add synonyms and related terms (simplified)
                related_terms = self._get_related_terms(query_text)
                expanded_query['expanded_terms'] = related_terms
                
                # Update query text
                all_terms = [query_text] + related_terms
                expanded_query['text'] = ' '.join(all_terms)
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"Failed to expand query: {e}")
            return query
    
    def _get_related_terms(self, text: str) -> List[str]:
        """Get related terms for query expansion."""
        try:
            # Simple related terms generation
            words = text.lower().split()
            related_terms = []
            
            # Add plurals/singulars
            for word in words:
                if word.endswith('s'):
                    related_terms.append(word[:-1])  # Remove 's'
                else:
                    related_terms.append(word + 's')  # Add 's'
            
            return related_terms[:3]  # Return top 3 related terms
            
        except Exception as e:
            logger.error(f"Failed to get related terms: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'cache_size': len(self.result_cache),
                'hit_rate': (stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses'])) * 100
            })
            return stats
    
    async def optimize(self):
        """Optimize retrieval engine performance."""
        try:
            # Clean up cache
            current_time = datetime.now(timezone.utc)
            expired_keys = []
            
            for key, entry in self.result_cache.items():
                if not self._is_cache_valid(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.result_cache[key]
            
            # Rebuild ANN index if needed
            if self.ann_index and hasattr(self.ann_index, 'rebuild'):
                self.ann_index.rebuild()
            
            logger.info("Retrieval engine optimization completed")
            
        except Exception as e:
            logger.error(f"Failed to optimize retrieval engine: {e}")
    
    async def shutdown(self):
        """Shutdown retrieval engine."""
        try:
            # Clear cache
            self.result_cache.clear()
            
            # Clean up indices
            if hasattr(self.ann_index, 'save_index'):
                self.ann_index.save_index('retrieval_index.bin')
            
            logger.info("Retrieval engine shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown retrieval engine: {e}")