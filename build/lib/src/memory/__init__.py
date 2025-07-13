"""
Advanced Memory Systems for RSI AI.
Implements hierarchical memory architecture with cognitive science principles.
"""

from .memory_hierarchy import RSIMemoryHierarchy, RSIMemoryConfig
from .working_memory import WorkingMemory
from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory

# Optional imports with graceful fallbacks
try:
    from .procedural_memory import ProceduralMemory
except ImportError:
    ProceduralMemory = None

try:
    from .memory_consolidation import MemoryConsolidation
except ImportError:
    MemoryConsolidation = None

try:
    from .retrieval_engine import RetrievalEngine
except ImportError:
    RetrievalEngine = None

try:
    from .memory_manager import MemoryManager
except ImportError:
    MemoryManager = None

__all__ = [
    'RSIMemoryHierarchy',
    'RSIMemoryConfig',
    'WorkingMemory',
    'SemanticMemory',
    'EpisodicMemory',
    'ProceduralMemory',
    'MemoryConsolidation',
    'RetrievalEngine',
    'MemoryManager'
]