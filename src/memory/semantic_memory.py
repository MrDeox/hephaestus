"""
Semantic Memory Implementation for RSI AI.
Implements structured knowledge storage with graph-based relationships.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone
import json
import hashlib
import networkx as nx
from collections import defaultdict
import numpy as np
import threading

# Optional advanced dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Semantic Memory System for RSI AI.
    
    Provides structured knowledge storage with:
    - Graph-based concept relationships
    - Vector embeddings for semantic similarity
    - Knowledge graph reasoning
    - Hierarchical concept organization
    """
    
    def __init__(self, backend: str = "networkx", graph_db_type: str = "networkx", 
                 embedding_dimension: int = 768):
        self.backend = backend
        self.graph_db_type = graph_db_type
        self.embedding_dimension = embedding_dimension
        
        # Initialize graph database
        self.graph = nx.DiGraph()
        self.concept_embeddings = {}
        self.concept_metadata = {}
        self.concept_relationships = defaultdict(set)
        
        # Initialize vector database if available
        self.vector_db = None
        if CHROMADB_AVAILABLE:
            self._init_vector_db()
        
        # Initialize knowledge graph embeddings if available
        self.kg_embeddings = None
        if PYKEEN_AVAILABLE:
            self._init_kg_embeddings()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'concepts_count': 0,
            'relationships_count': 0,
            'embeddings_count': 0,
            'queries_count': 0,
            'updates_count': 0
        }
        
        logger.info(f"Semantic Memory initialized with backend: {backend}")
    
    def _init_vector_db(self):
        """Initialize vector database for semantic similarity."""
        try:
            self.vector_db = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            self.concepts_collection = self.vector_db.get_or_create_collection(
                name="semantic_concepts",
                metadata={"description": "Semantic concepts and their embeddings"}
            )
            
            logger.info("✅ ChromaDB vector database initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize vector database: {e}")
            self.vector_db = None
    
    def _init_kg_embeddings(self):
        """Initialize knowledge graph embeddings."""
        try:
            # This will be initialized when we have enough data
            logger.info("✅ Knowledge graph embeddings ready for initialization")
            
        except Exception as e:
            logger.warning(f"Failed to initialize KG embeddings: {e}")
    
    def _generate_concept_id(self, concept: Dict[str, Any]) -> str:
        """Generate unique ID for a concept."""
        concept_str = json.dumps(concept, sort_keys=True, default=str)
        return hashlib.sha256(concept_str.encode()).hexdigest()
    
    def _extract_concept_text(self, concept: Dict[str, Any]) -> str:
        """Extract text representation of concept for embedding."""
        if 'text' in concept:
            return concept['text']
        elif 'name' in concept:
            return concept['name']
        elif 'description' in concept:
            return concept['description']
        else:
            return str(concept)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (placeholder implementation)."""
        # This would use a real embedding model in production
        # For now, use a simple hash-based approach
        hash_value = hashlib.sha256(text.encode()).hexdigest()
        
        # Convert hash to embedding
        embedding = np.array([
            ord(char) / 255.0 for char in hash_value[:self.embedding_dimension]
        ])
        
        # Pad or truncate to desired dimension
        if len(embedding) < self.embedding_dimension:
            embedding = np.pad(embedding, (0, self.embedding_dimension - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dimension]
        
        return embedding
    
    async def store_concept(self, concept: Dict[str, Any], relationships: List[Dict[str, Any]] = None) -> str:
        """
        Store a concept in semantic memory.
        
        Args:
            concept: Concept data
            relationships: Optional relationships to other concepts
            
        Returns:
            Concept ID
        """
        try:
            with self.lock:
                concept_id = self._generate_concept_id(concept)
                
                # Store concept in graph
                self.graph.add_node(concept_id, **concept)
                
                # Store concept metadata
                self.concept_metadata[concept_id] = {
                    'concept': concept,
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc),
                    'access_count': 0
                }
                
                # Generate and store embedding
                concept_text = self._extract_concept_text(concept)
                embedding = self._generate_embedding(concept_text)
                self.concept_embeddings[concept_id] = embedding
                
                # Store in vector database if available
                if self.vector_db:
                    try:
                        self.concepts_collection.add(
                            embeddings=[embedding.tolist()],
                            documents=[concept_text],
                            metadatas=[concept],
                            ids=[concept_id]
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store in vector DB: {e}")
                
                # Store relationships
                if relationships:
                    for rel in relationships:
                        await self._store_relationship(concept_id, rel)
                
                self.stats['concepts_count'] += 1
                self.stats['embeddings_count'] += 1
                self.stats['updates_count'] += 1
                
                return concept_id
                
        except Exception as e:
            logger.error(f"Failed to store concept: {e}")
            raise
    
    async def _store_relationship(self, source_id: str, relationship: Dict[str, Any]):
        """Store a relationship between concepts."""
        try:
            target_id = relationship.get('target_id')
            relation_type = relationship.get('type', 'related_to')
            relation_weight = relationship.get('weight', 1.0)
            
            if target_id:
                # Add edge to graph
                self.graph.add_edge(
                    source_id, target_id,
                    relation_type=relation_type,
                    weight=relation_weight,
                    created_at=datetime.now(timezone.utc)
                )
                
                # Update relationship tracking
                self.concept_relationships[source_id].add(target_id)
                self.concept_relationships[target_id].add(source_id)
                
                self.stats['relationships_count'] += 1
                
        except Exception as e:
            logger.error(f"Failed to store relationship: {e}")
    
    async def retrieve_concepts(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve concepts from semantic memory.
        
        Args:
            query: Search query
            
        Returns:
            List of matching concepts
        """
        try:
            with self.lock:
                results = []
                
                if 'id' in query:
                    # Direct ID lookup
                    concept_id = query['id']
                    if concept_id in self.concept_metadata:
                        concept_data = self.concept_metadata[concept_id]
                        concept_data['access_count'] += 1
                        results.append(self._format_concept_result(concept_id, concept_data))
                
                elif 'text' in query:
                    # Semantic similarity search
                    query_text = query['text']
                    limit = query.get('limit', 10)
                    
                    if self.vector_db:
                        # Use vector database for similarity search
                        try:
                            query_embedding = self._generate_embedding(query_text)
                            search_results = self.concepts_collection.query(
                                query_embeddings=[query_embedding.tolist()],
                                n_results=limit
                            )
                            
                            for i, concept_id in enumerate(search_results['ids'][0]):
                                if concept_id in self.concept_metadata:
                                    concept_data = self.concept_metadata[concept_id]
                                    concept_data['access_count'] += 1
                                    result = self._format_concept_result(concept_id, concept_data)
                                    result['similarity_score'] = 1.0 - search_results['distances'][0][i]
                                    results.append(result)
                                    
                        except Exception as e:
                            logger.warning(f"Vector DB search failed: {e}")
                            # Fallback to simple text matching
                            results = self._text_search_fallback(query_text, limit)
                    else:
                        # Simple text matching fallback
                        results = self._text_search_fallback(query_text, limit)
                
                elif 'concept_type' in query:
                    # Search by concept type
                    concept_type = query['concept_type']
                    limit = query.get('limit', 10)
                    
                    for concept_id, concept_data in self.concept_metadata.items():
                        if concept_data['concept'].get('type') == concept_type:
                            concept_data['access_count'] += 1
                            results.append(self._format_concept_result(concept_id, concept_data))
                            
                            if len(results) >= limit:
                                break
                
                elif 'relationships' in query:
                    # Search by relationships
                    source_id = query['relationships'].get('source_id')
                    relation_type = query['relationships'].get('type')
                    
                    if source_id in self.concept_relationships:
                        for target_id in self.concept_relationships[source_id]:
                            if self.graph.has_edge(source_id, target_id):
                                edge_data = self.graph[source_id][target_id]
                                if not relation_type or edge_data.get('relation_type') == relation_type:
                                    if target_id in self.concept_metadata:
                                        concept_data = self.concept_metadata[target_id]
                                        concept_data['access_count'] += 1
                                        result = self._format_concept_result(target_id, concept_data)
                                        result['relationship'] = edge_data
                                        results.append(result)
                
                else:
                    # Return all concepts (limited)
                    limit = query.get('limit', 10)
                    for concept_id, concept_data in list(self.concept_metadata.items())[:limit]:
                        results.append(self._format_concept_result(concept_id, concept_data))
                
                self.stats['queries_count'] += 1
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve concepts: {e}")
            return []
    
    def _text_search_fallback(self, query_text: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback text search without vector database."""
        results = []
        query_lower = query_text.lower()
        
        for concept_id, concept_data in self.concept_metadata.items():
            concept_text = self._extract_concept_text(concept_data['concept']).lower()
            
            if query_lower in concept_text:
                concept_data['access_count'] += 1
                results.append(self._format_concept_result(concept_id, concept_data))
                
                if len(results) >= limit:
                    break
        
        return results
    
    def _format_concept_result(self, concept_id: str, concept_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format concept data for result."""
        return {
            'id': concept_id,
            'concept': concept_data['concept'],
            'created_at': concept_data['created_at'].isoformat(),
            'updated_at': concept_data['updated_at'].isoformat(),
            'access_count': concept_data['access_count'],
            'relationships_count': len(self.concept_relationships.get(concept_id, set()))
        }
    
    async def add_relationship(self, source_id: str, target_id: str, 
                             relation_type: str = "related_to", weight: float = 1.0) -> bool:
        """
        Add a relationship between two concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relation_type: Type of relationship
            weight: Relationship weight
            
        Returns:
            Success status
        """
        try:
            relationship = {
                'target_id': target_id,
                'type': relation_type,
                'weight': weight
            }
            
            await self._store_relationship(source_id, relationship)
            return True
            
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            return False
    
    async def get_related_concepts(self, concept_id: str, relation_type: str = None, 
                                 max_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept_id: Source concept ID
            relation_type: Optional relation type filter
            max_depth: Maximum traversal depth
            
        Returns:
            List of related concepts
        """
        try:
            with self.lock:
                if concept_id not in self.concept_metadata:
                    return []
                
                related_concepts = []
                visited = set()
                
                def traverse(current_id, depth):
                    if depth > max_depth or current_id in visited:
                        return
                    
                    visited.add(current_id)
                    
                    # Get direct relationships
                    if current_id in self.concept_relationships:
                        for target_id in self.concept_relationships[current_id]:
                            if self.graph.has_edge(current_id, target_id):
                                edge_data = self.graph[current_id][target_id]
                                
                                if not relation_type or edge_data.get('relation_type') == relation_type:
                                    if target_id in self.concept_metadata:
                                        concept_data = self.concept_metadata[target_id]
                                        result = self._format_concept_result(target_id, concept_data)
                                        result['relationship'] = edge_data
                                        result['depth'] = depth
                                        related_concepts.append(result)
                                        
                                        # Recursive traversal
                                        if depth < max_depth:
                                            traverse(target_id, depth + 1)
                
                traverse(concept_id, 0)
                return related_concepts
                
        except Exception as e:
            logger.error(f"Failed to get related concepts: {e}")
            return []
    
    async def get_concepts_count(self) -> int:
        """Get total number of concepts."""
        return len(self.concept_metadata)
    
    async def get_concept_paths(self, source_id: str, target_id: str, 
                              max_paths: int = 5) -> List[List[str]]:
        """
        Find paths between two concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            max_paths: Maximum number of paths to return
            
        Returns:
            List of concept paths
        """
        try:
            with self.lock:
                if source_id not in self.graph or target_id not in self.graph:
                    return []
                
                paths = []
                try:
                    # Find shortest paths
                    all_paths = nx.all_shortest_paths(self.graph, source_id, target_id)
                    
                    for path in all_paths:
                        paths.append(path)
                        if len(paths) >= max_paths:
                            break
                            
                except nx.NetworkXNoPath:
                    # No path exists
                    pass
                
                return paths
                
        except Exception as e:
            logger.error(f"Failed to find concept paths: {e}")
            return []
    
    async def cleanup(self):
        """Clean up semantic memory."""
        try:
            with self.lock:
                # Remove concepts with low access counts
                low_access_concepts = [
                    concept_id for concept_id, data in self.concept_metadata.items()
                    if data['access_count'] < 2 and 
                    (datetime.now(timezone.utc) - data['created_at']).days > 30
                ]
                
                for concept_id in low_access_concepts:
                    await self.remove_concept(concept_id)
                
                logger.info(f"Cleaned up {len(low_access_concepts)} low-access concepts")
                
        except Exception as e:
            logger.error(f"Failed to cleanup semantic memory: {e}")
    
    async def remove_concept(self, concept_id: str) -> bool:
        """Remove a concept from semantic memory."""
        try:
            with self.lock:
                if concept_id not in self.concept_metadata:
                    return False
                
                # Remove from graph
                if self.graph.has_node(concept_id):
                    self.graph.remove_node(concept_id)
                
                # Remove from metadata
                del self.concept_metadata[concept_id]
                
                # Remove embedding
                if concept_id in self.concept_embeddings:
                    del self.concept_embeddings[concept_id]
                
                # Remove from relationships
                if concept_id in self.concept_relationships:
                    del self.concept_relationships[concept_id]
                
                # Remove from vector database
                if self.vector_db:
                    try:
                        self.concepts_collection.delete(ids=[concept_id])
                    except Exception as e:
                        logger.warning(f"Failed to remove from vector DB: {e}")
                
                self.stats['concepts_count'] -= 1
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove concept: {e}")
            return False
    
    async def optimize(self):
        """Optimize semantic memory performance."""
        try:
            # Clean up orphaned relationships
            with self.lock:
                valid_concepts = set(self.concept_metadata.keys())
                
                # Clean up relationship tracking
                for concept_id in list(self.concept_relationships.keys()):
                    if concept_id not in valid_concepts:
                        del self.concept_relationships[concept_id]
                    else:
                        # Remove invalid targets
                        self.concept_relationships[concept_id] = {
                            target_id for target_id in self.concept_relationships[concept_id]
                            if target_id in valid_concepts
                        }
                
                # Update graph structure
                orphaned_nodes = [
                    node for node in self.graph.nodes()
                    if node not in valid_concepts
                ]
                
                for node in orphaned_nodes:
                    self.graph.remove_node(node)
                
                logger.info("Semantic memory optimization completed")
                
        except Exception as e:
            logger.error(f"Failed to optimize semantic memory: {e}")
    
    def get_size(self) -> int:
        """Get current size of semantic memory."""
        return len(self.concept_metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'current_concepts': len(self.concept_metadata),
                'current_relationships': len(self.graph.edges()),
                'current_embeddings': len(self.concept_embeddings),
                'graph_density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
                'connected_components': nx.number_weakly_connected_components(self.graph)
            })
            return stats
    
    async def shutdown(self):
        """Shutdown semantic memory."""
        try:
            # Final optimization
            await self.optimize()
            
            # Clear data structures
            with self.lock:
                self.graph.clear()
                self.concept_metadata.clear()
                self.concept_embeddings.clear()
                self.concept_relationships.clear()
            
            logger.info("Semantic memory shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown semantic memory: {e}")