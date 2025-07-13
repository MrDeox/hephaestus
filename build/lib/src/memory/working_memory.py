"""
Working Memory Implementation for RSI AI.
Implements immediate access, temporary storage for active information.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque, OrderedDict
import hashlib
import json
import heapq
from dataclasses import dataclass, field
import threading
import weakref

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    """Item stored in working memory."""
    id: str
    content: Dict[str, Any]
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: float = 1.0
    tags: List[str] = field(default_factory=list)
    expiry: Optional[datetime] = None
    
    def __post_init__(self):
        # Calculate priority based on recency and access frequency
        self.update_priority()
    
    def update_priority(self):
        """Update priority based on recency and frequency."""
        now = datetime.now(timezone.utc)
        
        # Recency factor (0-1, higher for more recent)
        recency = 1.0 / (1.0 + (now - self.last_access).total_seconds() / 3600)
        
        # Frequency factor (0-1, higher for more frequent access)
        frequency = min(1.0, self.access_count / 10.0)
        
        # Combined priority
        self.priority = (recency * 0.7) + (frequency * 0.3)
    
    def access(self):
        """Record an access to this item."""
        self.access_count += 1
        self.last_access = datetime.now(timezone.utc)
        self.update_priority()
    
    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.expiry is None:
            return False
        return datetime.now(timezone.utc) > self.expiry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'access_count': self.access_count,
            'last_access': self.last_access.isoformat(),
            'priority': self.priority,
            'tags': self.tags,
            'expiry': self.expiry.isoformat() if self.expiry else None
        }


class WorkingMemory:
    """
    Working Memory System for RSI AI.
    
    Provides immediate access temporary storage with:
    - Fast O(1) access by ID
    - Priority-based eviction
    - Automatic cleanup of expired items
    - Thread-safe operations
    """
    
    def __init__(self, capacity: int = 10000, max_size: int = 50000):
        self.capacity = capacity
        self.max_size = max_size
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.priority_queue: List[Tuple[float, str]] = []  # (priority, item_id)
        self.access_order = OrderedDict()  # LRU tracking
        self.lock = threading.RLock()
        self.stats = {
            'total_items': 0,
            'total_accesses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'expirations': 0
        }
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info(f"Working Memory initialized with capacity {capacity}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        try:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            pass
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await self._cleanup_expired()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Working memory cleanup error: {e}")
                await asyncio.sleep(60)
    
    def _generate_id(self, content: Dict[str, Any]) -> str:
        """Generate unique ID for content."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _make_room(self, needed: int = 1):
        """Make room in working memory by evicting items."""
        with self.lock:
            if len(self.items) + needed <= self.capacity:
                return
            
            # First, remove expired items
            expired_items = [
                item_id for item_id, item in self.items.items()
                if item.is_expired()
            ]
            
            for item_id in expired_items:
                self._remove_item(item_id)
                self.stats['expirations'] += 1
            
            # If still need room, evict lowest priority items
            items_to_evict = len(self.items) + needed - self.capacity
            if items_to_evict > 0:
                # Sort by priority (lowest first)
                sorted_items = sorted(
                    self.items.items(),
                    key=lambda x: x[1].priority
                )
                
                for i in range(min(items_to_evict, len(sorted_items))):
                    item_id = sorted_items[i][0]
                    self._remove_item(item_id)
                    self.stats['evictions'] += 1
    
    def _remove_item(self, item_id: str):
        """Remove item from all data structures."""
        if item_id in self.items:
            del self.items[item_id]
        
        if item_id in self.access_order:
            del self.access_order[item_id]
        
        # Remove from priority queue (expensive, but necessary)
        self.priority_queue = [
            (priority, id_) for priority, id_ in self.priority_queue
            if id_ != item_id
        ]
        heapq.heapify(self.priority_queue)
    
    async def store(self, content: Dict[str, Any], tags: List[str] = None, 
                   ttl_seconds: Optional[int] = None) -> str:
        """
        Store content in working memory.
        
        Args:
            content: Content to store
            tags: Optional tags for categorization
            ttl_seconds: Time to live in seconds
            
        Returns:
            Item ID
        """
        try:
            with self.lock:
                item_id = self._generate_id(content)
                
                # Check if item already exists
                if item_id in self.items:
                    self.items[item_id].access()
                    self.stats['cache_hits'] += 1
                    return item_id
                
                # Make room if needed
                self._make_room(1)
                
                # Calculate expiry
                expiry = None
                if ttl_seconds:
                    expiry = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                
                # Create new item
                item = WorkingMemoryItem(
                    id=item_id,
                    content=content,
                    timestamp=datetime.now(timezone.utc),
                    tags=tags or [],
                    expiry=expiry
                )
                
                # Store item
                self.items[item_id] = item
                self.access_order[item_id] = True
                heapq.heappush(self.priority_queue, (item.priority, item_id))
                
                self.stats['total_items'] += 1
                self.stats['cache_misses'] += 1
                
                return item_id
                
        except Exception as e:
            logger.error(f"Failed to store in working memory: {e}")
            raise
    
    async def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve items from working memory.
        
        Args:
            query: Search query
            
        Returns:
            List of matching items
        """
        try:
            with self.lock:
                results = []
                
                # Handle different query types
                if 'id' in query:
                    # Direct ID lookup
                    item_id = query['id']
                    if item_id in self.items:
                        item = self.items[item_id]
                        if not item.is_expired():
                            item.access()
                            results.append(item.to_dict())
                            self.stats['cache_hits'] += 1
                        else:
                            self._remove_item(item_id)
                            self.stats['expirations'] += 1
                    else:
                        self.stats['cache_misses'] += 1
                
                elif 'tags' in query:
                    # Tag-based search
                    search_tags = query['tags']
                    if isinstance(search_tags, str):
                        search_tags = [search_tags]
                    
                    for item in self.items.values():
                        if not item.is_expired():
                            if any(tag in item.tags for tag in search_tags):
                                item.access()
                                results.append(item.to_dict())
                                self.stats['cache_hits'] += 1
                        else:
                            self._remove_item(item.id)
                            self.stats['expirations'] += 1
                
                elif 'content_match' in query:
                    # Content-based search
                    search_content = query['content_match']
                    
                    for item in self.items.values():
                        if not item.is_expired():
                            # Simple content matching
                            if self._matches_content(item.content, search_content):
                                item.access()
                                results.append(item.to_dict())
                                self.stats['cache_hits'] += 1
                        else:
                            self._remove_item(item.id)
                            self.stats['expirations'] += 1
                
                else:
                    # Return all items if no specific query
                    for item in self.items.values():
                        if not item.is_expired():
                            results.append(item.to_dict())
                        else:
                            self._remove_item(item.id)
                            self.stats['expirations'] += 1
                
                self.stats['total_accesses'] += 1
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve from working memory: {e}")
            return []
    
    def _matches_content(self, content: Dict[str, Any], search_content: Dict[str, Any]) -> bool:
        """Check if content matches search criteria."""
        try:
            for key, value in search_content.items():
                if key not in content:
                    return False
                
                if isinstance(value, str):
                    if str(content[key]).lower().find(value.lower()) == -1:
                        return False
                else:
                    if content[key] != value:
                        return False
            
            return True
            
        except Exception:
            return False
    
    async def remove(self, item_id: str) -> bool:
        """
        Remove item from working memory.
        
        Args:
            item_id: ID of item to remove
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                if item_id in self.items:
                    self._remove_item(item_id)
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove from working memory: {e}")
            return False
    
    async def clear(self):
        """Clear all items from working memory."""
        try:
            with self.lock:
                self.items.clear()
                self.access_order.clear()
                self.priority_queue.clear()
                
                # Reset stats
                self.stats.update({
                    'total_items': 0,
                    'total_accesses': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'evictions': 0,
                    'expirations': 0
                })
                
        except Exception as e:
            logger.error(f"Failed to clear working memory: {e}")
    
    async def _cleanup_expired(self):
        """Clean up expired items."""
        try:
            with self.lock:
                expired_items = [
                    item_id for item_id, item in self.items.items()
                    if item.is_expired()
                ]
                
                for item_id in expired_items:
                    self._remove_item(item_id)
                    self.stats['expirations'] += 1
                
                if expired_items:
                    logger.debug(f"Cleaned up {len(expired_items)} expired items")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup expired items: {e}")
    
    async def cleanup(self):
        """Manual cleanup of working memory."""
        await self._cleanup_expired()
        
        # Optimize priority queue
        with self.lock:
            valid_items = [
                (item.priority, item.id) for item in self.items.values()
            ]
            self.priority_queue = valid_items
            heapq.heapify(self.priority_queue)
    
    def get_size(self) -> int:
        """Get current size of working memory."""
        return len(self.items)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        with self.lock:
            stats = self.stats.copy()
            stats.update({
                'current_size': len(self.items),
                'capacity': self.capacity,
                'usage_percent': (len(self.items) / self.capacity) * 100,
                'hit_rate': (stats['cache_hits'] / max(1, stats['total_accesses'])) * 100
            })
            return stats
    
    def get_top_items(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N items by priority."""
        with self.lock:
            # Sort by priority (highest first)
            sorted_items = sorted(
                self.items.values(),
                key=lambda x: x.priority,
                reverse=True
            )
            
            return [item.to_dict() for item in sorted_items[:n]]
    
    async def shutdown(self):
        """Shutdown working memory."""
        try:
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Working memory shut down")
            
        except Exception as e:
            logger.error(f"Failed to shutdown working memory: {e}")