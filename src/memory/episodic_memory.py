"""
Episodic Memory Implementation for RSI AI.
Implements temporal sequence storage and retrieval for experiences.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import deque
import json
import threading
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """An episode in episodic memory."""
    id: str
    content: Dict[str, Any]
    timestamp: datetime
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    emotions: Dict[str, float] = field(default_factory=dict)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'context': self.context,
            'importance': self.importance,
            'emotions': self.emotions,
            'outcomes': self.outcomes
        }


class EpisodicMemory:
    """
    Episodic Memory System for RSI AI.
    
    Stores temporal sequences of experiences with:
    - Time-ordered episode storage
    - Context-based retrieval
    - Importance weighting
    - Emotional tagging
    - Outcome tracking
    """
    
    def __init__(self, backend: str = "sqlite", max_episodes: int = 100000):
        self.backend = backend
        self.max_episodes = max_episodes
        self.episodes: Dict[str, Episode] = {}
        self.temporal_index: deque = deque(maxlen=max_episodes)
        self.tag_index: Dict[str, List[str]] = {}
        self.importance_index: List[Tuple[float, str]] = []
        
        # Database connection
        self.db_path = Path("./episodic_memory.db")
        self.db_connection = None
        self._init_database()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_episodes': 0,
            'queries_count': 0,
            'retrievals_count': 0,
            'consolidations_count': 0
        }
        
        logger.info(f"Episodic Memory initialized with backend: {backend}")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        try:
            self.db_connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT,
                    context TEXT,
                    importance REAL DEFAULT 1.0,
                    emotions TEXT,
                    outcomes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_importance ON episodes(importance DESC)
            ''')
            
            self.db_connection.commit()
            logger.info("✅ Episodic memory database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _generate_episode_id(self) -> str:
        """Generate unique episode ID."""
        import uuid
        return str(uuid.uuid4())
    
    async def record_episode(self, content: Dict[str, Any], tags: List[str] = None,
                           context: Dict[str, Any] = None, importance: float = 1.0,
                           emotions: Dict[str, float] = None) -> str:
        """
        Record a new episode in episodic memory.
        
        Args:
            content: Episode content
            tags: Optional tags for categorization
            context: Optional context information
            importance: Episode importance (0.0 to 1.0)
            emotions: Optional emotional annotations
            
        Returns:
            Episode ID
        """
        try:
            with self.lock:
                episode_id = self._generate_episode_id()
                
                episode = Episode(
                    id=episode_id,
                    content=content,
                    timestamp=datetime.now(timezone.utc),
                    tags=tags or [],
                    context=context or {},
                    importance=importance,
                    emotions=emotions or {}
                )
                
                # Store in memory
                self.episodes[episode_id] = episode
                self.temporal_index.append(episode_id)
                
                # Update tag index
                for tag in episode.tags:
                    if tag not in self.tag_index:
                        self.tag_index[tag] = []
                    self.tag_index[tag].append(episode_id)
                
                # Update importance index
                self.importance_index.append((importance, episode_id))
                self.importance_index.sort(reverse=True)  # High importance first
                
                # Persist to database
                if self.db_connection:
                    cursor = self.db_connection.cursor()
                    cursor.execute('''
                        INSERT INTO episodes 
                        (id, content, timestamp, tags, context, importance, emotions, outcomes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        episode_id,
                        json.dumps(content),
                        episode.timestamp.isoformat(),
                        json.dumps(tags or []),
                        json.dumps(context or {}),
                        importance,
                        json.dumps(emotions or {}),
                        json.dumps({})
                    ))
                    self.db_connection.commit()
                
                self.stats['total_episodes'] += 1
                
                # Manage memory limits
                if len(self.episodes) > self.max_episodes:
                    await self._cleanup_old_episodes()
                
                return episode_id
                
        except Exception as e:
            logger.error(f"Failed to record episode: {e}")
            raise
    
    async def retrieve_episodes(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve episodes from episodic memory.
        
        Args:
            query: Search query
            
        Returns:
            List of matching episodes
        """
        try:
            with self.lock:
                results = []
                
                if 'id' in query:
                    # Direct ID lookup
                    episode_id = query['id']
                    if episode_id in self.episodes:
                        results.append(self.episodes[episode_id].to_dict())
                
                elif 'time_range' in query:
                    # Time-based retrieval
                    start_time = datetime.fromisoformat(query['time_range']['start'])
                    end_time = datetime.fromisoformat(query['time_range']['end'])
                    limit = query.get('limit', 50)
                    
                    for episode_id in self.temporal_index:
                        if episode_id in self.episodes:
                            episode = self.episodes[episode_id]
                            if start_time <= episode.timestamp <= end_time:
                                results.append(episode.to_dict())
                                
                                if len(results) >= limit:
                                    break
                
                elif 'tags' in query:
                    # Tag-based retrieval
                    search_tags = query['tags']
                    if isinstance(search_tags, str):
                        search_tags = [search_tags]
                    
                    limit = query.get('limit', 50)
                    episode_ids = set()
                    
                    for tag in search_tags:
                        if tag in self.tag_index:
                            episode_ids.update(self.tag_index[tag])
                    
                    for episode_id in episode_ids:
                        if episode_id in self.episodes:
                            results.append(self.episodes[episode_id].to_dict())
                            
                            if len(results) >= limit:
                                break
                
                elif 'importance_threshold' in query:
                    # Importance-based retrieval
                    threshold = query['importance_threshold']
                    limit = query.get('limit', 50)
                    
                    for importance, episode_id in self.importance_index:
                        if importance >= threshold and episode_id in self.episodes:
                            results.append(self.episodes[episode_id].to_dict())
                            
                            if len(results) >= limit:
                                break
                
                elif 'recent' in query:
                    # Recent episodes
                    limit = query.get('limit', 20)
                    
                    # Get most recent episodes
                    recent_ids = list(self.temporal_index)[-limit:]
                    
                    for episode_id in reversed(recent_ids):
                        if episode_id in self.episodes:
                            results.append(self.episodes[episode_id].to_dict())
                
                elif 'emotions' in query:
                    # Emotion-based retrieval
                    target_emotions = query['emotions']
                    limit = query.get('limit', 50)
                    
                    for episode_id, episode in self.episodes.items():
                        if self._matches_emotions(episode.emotions, target_emotions):
                            results.append(episode.to_dict())
                            
                            if len(results) >= limit:
                                break
                
                elif 'context' in query:
                    # Context-based retrieval
                    target_context = query['context']
                    limit = query.get('limit', 50)
                    
                    for episode_id, episode in self.episodes.items():
                        if self._matches_context(episode.context, target_context):
                            results.append(episode.to_dict())
                            
                            if len(results) >= limit:
                                break
                
                else:
                    # Default: return recent episodes
                    limit = query.get('limit', 20)
                    recent_ids = list(self.temporal_index)[-limit:]
                    
                    for episode_id in reversed(recent_ids):
                        if episode_id in self.episodes:
                            results.append(self.episodes[episode_id].to_dict())
                
                self.stats['queries_count'] += 1
                self.stats['retrievals_count'] += len(results)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve episodes: {e}")
            return []
    
    def _matches_emotions(self, episode_emotions: Dict[str, float], 
                         target_emotions: Dict[str, float]) -> bool:
        """Check if episode emotions match target emotions."""
        try:
            for emotion, min_intensity in target_emotions.items():
                if emotion not in episode_emotions:
                    return False
                if episode_emotions[emotion] < min_intensity:
                    return False
            return True
        except Exception:
            return False
    
    def _matches_context(self, episode_context: Dict[str, Any], 
                        target_context: Dict[str, Any]) -> bool:
        """Check if episode context matches target context."""
        try:
            for key, value in target_context.items():
                if key not in episode_context:
                    return False
                if episode_context[key] != value:
                    return False
            return True
        except Exception:
            return False
    
    async def update_episode_outcome(self, episode_id: str, outcomes: Dict[str, Any]) -> bool:
        """
        Update the outcomes of an episode.
        
        Args:
            episode_id: Episode ID
            outcomes: Outcome information
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                if episode_id not in self.episodes:
                    return False
                
                # Update in memory
                self.episodes[episode_id].outcomes = outcomes
                
                # Update in database
                if self.db_connection:
                    cursor = self.db_connection.cursor()
                    cursor.execute('''
                        UPDATE episodes SET outcomes = ? WHERE id = ?
                    ''', (json.dumps(outcomes), episode_id))
                    self.db_connection.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update episode outcome: {e}")
            return False
    
    async def get_episode_sequence(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Get a sequence of episodes within a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            
        Returns:
            Chronologically ordered episodes
        """
        try:
            with self.lock:
                sequence = []
                
                for episode_id in self.temporal_index:
                    if episode_id in self.episodes:
                        episode = self.episodes[episode_id]
                        if start_time <= episode.timestamp <= end_time:
                            sequence.append(episode.to_dict())
                
                # Sort by timestamp
                sequence.sort(key=lambda x: x['timestamp'])
                
                return sequence
                
        except Exception as e:
            logger.error(f"Failed to get episode sequence: {e}")
            return []
    
    async def get_episodes_count(self) -> int:
        """Get total number of episodes."""
        return len(self.episodes)
    
    async def find_similar_episodes(self, reference_episode_id: str, 
                                  similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find episodes similar to a reference episode.
        
        Args:
            reference_episode_id: Reference episode ID
            similarity_threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar episodes
        """
        try:
            with self.lock:
                if reference_episode_id not in self.episodes:
                    return []
                
                reference_episode = self.episodes[reference_episode_id]
                similar_episodes = []
                
                for episode_id, episode in self.episodes.items():
                    if episode_id == reference_episode_id:
                        continue
                    
                    similarity = self._calculate_episode_similarity(reference_episode, episode)
                    
                    if similarity >= similarity_threshold:
                        result = episode.to_dict()
                        result['similarity_score'] = similarity
                        similar_episodes.append(result)
                
                # Sort by similarity score
                similar_episodes.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                return similar_episodes
                
        except Exception as e:
            logger.error(f"Failed to find similar episodes: {e}")
            return []
    
    def _calculate_episode_similarity(self, episode1: Episode, episode2: Episode) -> float:
        """Calculate similarity between two episodes."""
        try:
            similarity_score = 0.0
            
            # Tag similarity
            if episode1.tags and episode2.tags:
                common_tags = set(episode1.tags) & set(episode2.tags)
                total_tags = set(episode1.tags) | set(episode2.tags)
                tag_similarity = len(common_tags) / len(total_tags) if total_tags else 0
                similarity_score += tag_similarity * 0.3
            
            # Context similarity
            if episode1.context and episode2.context:
                common_context = set(episode1.context.keys()) & set(episode2.context.keys())
                total_context = set(episode1.context.keys()) | set(episode2.context.keys())
                context_similarity = len(common_context) / len(total_context) if total_context else 0
                similarity_score += context_similarity * 0.3
            
            # Emotion similarity
            if episode1.emotions and episode2.emotions:
                common_emotions = set(episode1.emotions.keys()) & set(episode2.emotions.keys())
                emotion_similarity = len(common_emotions) / max(len(episode1.emotions), len(episode2.emotions))
                similarity_score += emotion_similarity * 0.2
            
            # Temporal proximity
            time_diff = abs((episode1.timestamp - episode2.timestamp).total_seconds())
            temporal_similarity = max(0, 1 - (time_diff / (7 * 24 * 3600)))  # 1 week normalization
            similarity_score += temporal_similarity * 0.2
            
            return similarity_score
            
        except Exception as e:
            logger.error(f"Failed to calculate episode similarity: {e}")
            return 0.0
    
    async def _cleanup_old_episodes(self):
        """Clean up old episodes to maintain memory limits."""
        try:
            with self.lock:
                # Remove oldest episodes that exceed capacity
                episodes_to_remove = len(self.episodes) - self.max_episodes
                if episodes_to_remove <= 0:
                    return
                
                # Get oldest episodes (lowest importance or oldest timestamp)
                episode_scores = []
                for episode_id, episode in self.episodes.items():
                    # Score based on importance and age
                    age_days = (datetime.now(timezone.utc) - episode.timestamp).days
                    score = episode.importance - (age_days * 0.01)  # Decay by age
                    episode_scores.append((score, episode_id))
                
                # Sort by score (lowest first)
                episode_scores.sort()
                
                # Remove lowest scoring episodes
                for i in range(episodes_to_remove):
                    if i < len(episode_scores):
                        _, episode_id = episode_scores[i]
                        await self.remove_episode(episode_id)
                
                logger.info(f"Cleaned up {episodes_to_remove} old episodes")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old episodes: {e}")
    
    async def remove_episode(self, episode_id: str) -> bool:
        """Remove an episode from memory."""
        try:
            with self.lock:
                if episode_id not in self.episodes:
                    return False
                
                episode = self.episodes[episode_id]
                
                # Remove from memory structures
                del self.episodes[episode_id]
                
                # Remove from temporal index
                if episode_id in self.temporal_index:
                    temp_deque = deque()
                    for id_ in self.temporal_index:
                        if id_ != episode_id:
                            temp_deque.append(id_)
                    self.temporal_index = temp_deque
                
                # Remove from tag index
                for tag in episode.tags:
                    if tag in self.tag_index:
                        if episode_id in self.tag_index[tag]:
                            self.tag_index[tag].remove(episode_id)
                        if not self.tag_index[tag]:
                            del self.tag_index[tag]
                
                # Remove from importance index
                self.importance_index = [
                    (importance, id_) for importance, id_ in self.importance_index
                    if id_ != episode_id
                ]
                
                # Remove from database
                if self.db_connection:
                    cursor = self.db_connection.cursor()
                    cursor.execute('DELETE FROM episodes WHERE id = ?', (episode_id,))
                    self.db_connection.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove episode: {e}")
            return False
    
    async def cleanup(self):
        """Clean up episodic memory."""
        try:
            # Remove very old episodes with low importance
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
            
            with self.lock:
                episodes_to_remove = []
                
                for episode_id, episode in self.episodes.items():
                    if (episode.timestamp < cutoff_date and 
                        episode.importance < 0.3):
                        episodes_to_remove.append(episode_id)
                
                for episode_id in episodes_to_remove:
                    await self.remove_episode(episode_id)
                
                logger.info(f"Cleaned up {len(episodes_to_remove)} old episodes")
                
        except Exception as e:
            logger.error(f"Failed to cleanup episodic memory: {e}")
    
    async def optimize(self):
        """Optimize episodic memory performance."""
        try:
            # Rebuild indices
            with self.lock:
                # Rebuild tag index
                self.tag_index.clear()
                for episode_id, episode in self.episodes.items():
                    for tag in episode.tags:
                        if tag not in self.tag_index:
                            self.tag_index[tag] = []
                        self.tag_index[tag].append(episode_id)
                
                # Rebuild importance index
                self.importance_index = [
                    (episode.importance, episode_id)
                    for episode_id, episode in self.episodes.items()
                ]
                self.importance_index.sort(reverse=True)
                
                # Optimize database
                if self.db_connection:
                    cursor = self.db_connection.cursor()
                    cursor.execute('VACUUM')
                    self.db_connection.commit()
                
                logger.info("Episodic memory optimization completed")
                
        except Exception as e:
            logger.error(f"Failed to optimize episodic memory: {e}")
    
    def get_size(self) -> int:
        """Get current size of episodic memory."""
        return len(self.episodes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'current_episodes': len(self.episodes),
                'unique_tags': len(self.tag_index),
                'average_importance': sum(ep.importance for ep in self.episodes.values()) / len(self.episodes) if self.episodes else 0,
                'time_span_days': (max(ep.timestamp for ep in self.episodes.values()) - 
                                 min(ep.timestamp for ep in self.episodes.values())).days if self.episodes else 0
            })
            return stats
    
    async def shutdown(self):
        """Shutdown episodic memory."""
        try:
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
            
            # Clear memory structures
            with self.lock:
                self.episodes.clear()
                self.temporal_index.clear()
                self.tag_index.clear()
                self.importance_index.clear()
            
            logger.info("Episodic memory shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown episodic memory: {e}")