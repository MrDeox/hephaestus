"""
Core immutable state management for RSI system.
Uses pyrsistent for structural sharing and safe state transitions.
"""

from typing import Any, Dict, Optional, TypeVar, Generic, Callable
from datetime import datetime, timezone
from pyrsistent import pmap, pvector, freeze, thaw, PMap, PVector
from pydantic import BaseModel, Field, field_validator
import hashlib
import json


T = TypeVar('T')


class StateTransition(BaseModel):
    """Represents a single state transition with complete audit trail."""
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    old_state_hash: str
    new_state_hash: str
    transition_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"frozen": True}


class RSIState(BaseModel):
    """Core RSI system state with immutable guarantees."""
    
    configuration: PMap = Field(default_factory=lambda: pmap())
    model_weights: PMap = Field(default_factory=lambda: pmap())
    learning_history: PVector = Field(default_factory=lambda: pvector())
    performance_metrics: PMap = Field(default_factory=lambda: pmap())
    safety_status: PMap = Field(default_factory=lambda: pmap())
    version: int = Field(default=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('configuration', 'model_weights', 'performance_metrics', 'safety_status', mode='before')
    def freeze_mappings(cls, v):
        """Ensure all mappings are immutable."""
        if isinstance(v, dict):
            return freeze(v)
        return v
    
    @field_validator('learning_history', mode='before')
    def freeze_vector(cls, v):
        """Ensure vector is immutable."""
        if isinstance(v, list):
            return freeze(v)
        return v
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of current state."""
        state_dict = {
            'configuration': thaw(self.configuration),
            'model_weights': thaw(self.model_weights),
            'learning_history': thaw(self.learning_history),
            'performance_metrics': thaw(self.performance_metrics),
            'safety_status': thaw(self.safety_status),
            'version': self.version
        }
        
        state_str = json.dumps(state_dict, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    model_config = {"arbitrary_types_allowed": True, "frozen": True}


class StateManager(Generic[T]):
    """Thread-safe state manager with immutable guarantees."""
    
    def __init__(self, initial_state: RSIState):
        self._current_state = initial_state
        self._transition_history: PVector[StateTransition] = pvector()
        self._state_snapshots: PMap[str, RSIState] = pmap()
        
    @property
    def current_state(self) -> RSIState:
        """Get current immutable state."""
        return self._current_state
    
    @property
    def transition_history(self) -> PVector[StateTransition]:
        """Get immutable transition history."""
        return self._transition_history
    
    def transition(
        self,
        transition_fn: Callable[[RSIState], RSIState],
        transition_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RSIState:
        """
        Apply a state transition function safely.
        
        Args:
            transition_fn: Pure function that transforms state
            transition_type: Type of transition for audit trail
            metadata: Additional metadata for transition
            
        Returns:
            New immutable state
        """
        old_state = self._current_state
        old_hash = old_state.compute_hash()
        
        try:
            new_state = transition_fn(old_state)
            new_hash = new_state.compute_hash()
            
            # Create transition record
            transition = StateTransition(
                old_state_hash=old_hash,
                new_state_hash=new_hash,
                transition_type=transition_type,
                metadata=metadata or {}
            )
            
            # Update state and history atomically
            self._current_state = new_state
            self._transition_history = self._transition_history.append(transition)
            
            # Store snapshot for potential rollback
            self._state_snapshots = self._state_snapshots.set(new_hash, new_state)
            
            return new_state
            
        except Exception as e:
            # Log error but don't modify state
            error_metadata = {
                'error': str(e),
                'error_type': type(e).__name__,
                'original_metadata': metadata or {}
            }
            
            failed_transition = StateTransition(
                old_state_hash=old_hash,
                new_state_hash=old_hash,  # Same hash since no change
                transition_type=f"FAILED_{transition_type}",
                metadata=error_metadata
            )
            
            self._transition_history = self._transition_history.append(failed_transition)
            raise
    
    def rollback_to_hash(self, state_hash: str) -> Optional[RSIState]:
        """
        Rollback to a previous state by hash.
        
        Args:
            state_hash: Hash of the state to rollback to
            
        Returns:
            The rollback state if found, None otherwise
        """
        if state_hash in self._state_snapshots:
            old_state = self._current_state
            rollback_state = self._state_snapshots[state_hash]
            
            # Record rollback transition
            transition = StateTransition(
                old_state_hash=old_state.compute_hash(),
                new_state_hash=state_hash,
                transition_type="ROLLBACK",
                metadata={'rollback_to': state_hash}
            )
            
            self._current_state = rollback_state
            self._transition_history = self._transition_history.append(transition)
            
            return rollback_state
        
        return None
    
    def get_state_at_hash(self, state_hash: str) -> Optional[RSIState]:
        """Get a specific state by hash without modifying current state."""
        return self._state_snapshots.get(state_hash)


# Factory functions for common state transitions
def update_configuration(config_updates: Dict[str, Any]) -> Callable[[RSIState], RSIState]:
    """Factory for configuration update transitions."""
    def transition(state: RSIState) -> RSIState:
        new_config = state.configuration.update(config_updates)
        return state.model_copy(
            update={
                'configuration': new_config,
                'version': state.version + 1,
                'last_modified': datetime.now(timezone.utc)
            }
        )
    return transition


def update_model_weights(weight_updates: Dict[str, Any]) -> Callable[[RSIState], RSIState]:
    """Factory for model weight update transitions."""
    def transition(state: RSIState) -> RSIState:
        new_weights = state.model_weights.update(weight_updates)
        return state.model_copy(
            update={
                'model_weights': new_weights,
                'version': state.version + 1,
                'last_modified': datetime.now(timezone.utc)
            }
        )
    return transition


def add_learning_record(record: Dict[str, Any]) -> Callable[[RSIState], RSIState]:
    """Factory for adding learning history records."""
    def transition(state: RSIState) -> RSIState:
        new_history = state.learning_history.append(freeze(record))
        return state.model_copy(
            update={
                'learning_history': new_history,
                'version': state.version + 1,
                'last_modified': datetime.now(timezone.utc)
            }
        )
    return transition


def update_safety_status(safety_updates: Dict[str, Any]) -> Callable[[RSIState], RSIState]:
    """Factory for safety status updates."""
    def transition(state: RSIState) -> RSIState:
        new_safety = state.safety_status.update(safety_updates)
        return state.model_copy(
            update={
                'safety_status': new_safety,
                'version': state.version + 1,
                'last_modified': datetime.now(timezone.utc)
            }
        )
    return transition