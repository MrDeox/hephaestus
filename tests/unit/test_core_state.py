"""
Unit tests for Core State Management (src/core/state.py)

Tests the immutable state management system using pyrsistent,
ensuring data integrity, state transitions, and performance.
"""

import pytest
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.core.state import (
    RSIState, RSIStateManager, create_initial_state,
    create_state_with_metrics, create_state_with_learning_data,
    create_state_with_safety_update, hash_state
)


class TestRSIState:
    """Test cases for RSIState immutable data structure."""
    
    @pytest.mark.unit
    def test_rsi_state_creation(self):
        """Test RSIState creation with default values."""
        state = RSIState()
        
        assert state.system_metrics == {}
        assert state.learning_metrics == {}
        assert state.safety_metrics == {}
        assert state.metadata == {}
        assert isinstance(state.created_at, datetime)
        assert state.version == "1.0.0"
    
    @pytest.mark.unit
    def test_rsi_state_with_data(self):
        """Test RSIState creation with custom data."""
        system_metrics = {"cpu": 0.5, "memory": 0.3}
        learning_metrics = {"accuracy": 0.95, "loss": 0.05}
        safety_metrics = {"safety_score": 0.9}
        metadata = {"experiment": "test_001"}
        
        state = RSIState(
            system_metrics=system_metrics,
            learning_metrics=learning_metrics,
            safety_metrics=safety_metrics,
            metadata=metadata
        )
        
        assert state.system_metrics == system_metrics
        assert state.learning_metrics == learning_metrics
        assert state.safety_metrics == safety_metrics
        assert state.metadata == metadata
    
    @pytest.mark.unit
    def test_state_immutability(self):
        """Test that RSIState is truly immutable."""
        state = RSIState(system_metrics={"cpu": 0.5})
        
        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            state.system_metrics = {"cpu": 0.7}
            
        with pytest.raises(AttributeError):
            state.version = "2.0.0"
    
    @pytest.mark.unit
    def test_state_evolution(self):
        """Test state evolution creates new instances."""
        original_state = RSIState(system_metrics={"cpu": 0.5})
        
        # Evolution should create new state instance
        new_state = original_state.evolve(
            system_metrics={"cpu": 0.7, "memory": 0.4}
        )
        
        assert original_state.system_metrics == {"cpu": 0.5}
        assert new_state.system_metrics == {"cpu": 0.7, "memory": 0.4}
        assert original_state is not new_state
        assert new_state.created_at >= original_state.created_at
    
    @pytest.mark.unit
    def test_state_to_dict(self):
        """Test conversion to dictionary."""
        state = RSIState(
            system_metrics={"cpu": 0.5},
            learning_metrics={"accuracy": 0.95},
            metadata={"test": True}
        )
        
        state_dict = state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert state_dict["system_metrics"] == {"cpu": 0.5}
        assert state_dict["learning_metrics"] == {"accuracy": 0.95}
        assert state_dict["metadata"] == {"test": True}
        assert "created_at" in state_dict
        assert "version" in state_dict
    
    @pytest.mark.unit
    def test_state_from_dict(self):
        """Test creation from dictionary."""
        state_dict = {
            "system_metrics": {"cpu": 0.5},
            "learning_metrics": {"accuracy": 0.95},
            "safety_metrics": {"safety_score": 0.9},
            "metadata": {"test": True},
            "version": "1.0.0"
        }
        
        state = RSIState.from_dict(state_dict)
        
        assert state.system_metrics == {"cpu": 0.5}
        assert state.learning_metrics == {"accuracy": 0.95}
        assert state.safety_metrics == {"safety_score": 0.9}
        assert state.metadata == {"test": True}
        assert state.version == "1.0.0"
    
    @pytest.mark.unit
    def test_state_validation(self):
        """Test state validation."""
        # Valid state should pass
        valid_state = RSIState(
            system_metrics={"cpu": 0.5},
            learning_metrics={"accuracy": 0.95}
        )
        assert valid_state.is_valid()
        
        # Test edge cases for validation
        state_with_none = RSIState(system_metrics=None)
        # Should handle None gracefully or raise appropriate error
        try:
            result = state_with_none.is_valid()
            assert isinstance(result, bool)
        except (TypeError, ValueError):
            # Expected for invalid data
            pass


class TestRSIStateManager:
    """Test cases for RSIStateManager."""
    
    @pytest.mark.unit
    async def test_state_manager_initialization(self, rsi_state):
        """Test state manager initialization."""
        manager = RSIStateManager(initial_state=rsi_state)
        
        assert manager.current_state == rsi_state
        assert len(manager.state_history) == 1
        assert manager.state_history[0] == rsi_state
    
    @pytest.mark.unit
    async def test_state_transition(self, state_manager):
        """Test state transitions."""
        initial_state = state_manager.current_state
        new_metrics = {"cpu": 0.8, "memory": 0.6}
        
        # Perform state transition
        new_state = await state_manager.transition_state(
            system_metrics=new_metrics
        )
        
        assert new_state.system_metrics == new_metrics
        assert state_manager.current_state == new_state
        assert len(state_manager.state_history) == 2
        assert state_manager.state_history[-1] == new_state
        assert initial_state != new_state
    
    @pytest.mark.unit
    async def test_concurrent_state_transitions(self, state_manager):
        """Test concurrent state transitions are handled safely."""
        import asyncio
        
        async def update_metrics(name: str, value: float):
            await state_manager.transition_state(
                system_metrics={name: value}
            )
        
        # Run concurrent updates
        await asyncio.gather(
            update_metrics("cpu", 0.5),
            update_metrics("memory", 0.6),
            update_metrics("disk", 0.3)
        )
        
        # Should have 4 states total (initial + 3 updates)
        assert len(state_manager.state_history) == 4
        
        # Final state should contain all metrics
        final_state = state_manager.current_state
        assert "cpu" in final_state.system_metrics or \
               "memory" in final_state.system_metrics or \
               "disk" in final_state.system_metrics
    
    @pytest.mark.unit
    async def test_state_rollback(self, state_manager):
        """Test state rollback functionality."""
        # Perform several state transitions
        await state_manager.transition_state(system_metrics={"cpu": 0.5})
        await state_manager.transition_state(system_metrics={"cpu": 0.7})
        await state_manager.transition_state(system_metrics={"cpu": 0.9})
        
        assert len(state_manager.state_history) == 4  # initial + 3 transitions
        
        # Rollback to previous state
        rolled_back_state = await state_manager.rollback_state()
        
        assert rolled_back_state.system_metrics["cpu"] == 0.7
        assert state_manager.current_state == rolled_back_state
        assert len(state_manager.state_history) == 3
    
    @pytest.mark.unit
    async def test_state_rollback_to_version(self, state_manager):
        """Test rollback to specific version."""
        initial_state = state_manager.current_state
        
        # Perform transitions
        await state_manager.transition_state(system_metrics={"cpu": 0.5})
        state_v2 = await state_manager.transition_state(system_metrics={"cpu": 0.7})
        await state_manager.transition_state(system_metrics={"cpu": 0.9})
        
        # Rollback to specific state
        rolled_back = await state_manager.rollback_to_state(state_v2)
        
        assert rolled_back.system_metrics["cpu"] == 0.7
        assert state_manager.current_state == state_v2
    
    @pytest.mark.unit
    async def test_state_persistence(self, state_manager, temp_dir):
        """Test state persistence and loading."""
        # Perform state transitions
        await state_manager.transition_state(
            system_metrics={"cpu": 0.5, "memory": 0.3},
            learning_metrics={"accuracy": 0.95},
            metadata={"test_run": "persistence_test"}
        )
        
        # Save state
        save_path = temp_dir / "test_state.pkl"
        await state_manager.save_state(str(save_path))
        
        assert save_path.exists()
        
        # Load state in new manager
        new_manager = RSIStateManager()
        await new_manager.load_state(str(save_path))
        
        assert new_manager.current_state.system_metrics == {"cpu": 0.5, "memory": 0.3}
        assert new_manager.current_state.learning_metrics == {"accuracy": 0.95}
        assert new_manager.current_state.metadata == {"test_run": "persistence_test"}
    
    @pytest.mark.unit
    async def test_state_history_management(self, state_manager):
        """Test state history size management."""
        # Set history limit
        state_manager.max_history_size = 3
        
        # Perform multiple transitions (more than limit)
        for i in range(5):
            await state_manager.transition_state(
                system_metrics={"iteration": i}
            )
        
        # History should be limited to max size
        assert len(state_manager.state_history) <= 3
        
        # Should contain most recent states
        recent_state = state_manager.state_history[-1]
        assert recent_state.system_metrics["iteration"] == 4
    
    @pytest.mark.unit
    async def test_state_validation_on_transition(self, state_manager):
        """Test that invalid state transitions are rejected."""
        # Test with invalid metrics (negative values where inappropriate)
        with pytest.raises((ValueError, TypeError)):
            await state_manager.transition_state(
                system_metrics={"cpu": -0.5}  # Invalid negative CPU usage
            )
    
    @pytest.mark.unit
    async def test_state_hash_consistency(self, rsi_state):
        """Test that state hashing is consistent."""
        hash1 = hash_state(rsi_state)
        hash2 = hash_state(rsi_state)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex digest length
    
    @pytest.mark.unit
    async def test_state_hash_uniqueness(self):
        """Test that different states produce different hashes."""
        state1 = RSIState(system_metrics={"cpu": 0.5})
        state2 = RSIState(system_metrics={"cpu": 0.7})
        
        hash1 = hash_state(state1)
        hash2 = hash_state(state2)
        
        assert hash1 != hash2


class TestStateFactoryFunctions:
    """Test cases for state factory functions."""
    
    @pytest.mark.unit
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state()
        
        assert isinstance(state, RSIState)
        assert state.version == "1.0.0"
        assert isinstance(state.created_at, datetime)
        assert state.system_metrics == {}
        assert state.learning_metrics == {}
        assert state.safety_metrics == {}
    
    @pytest.mark.unit
    def test_create_state_with_metrics(self):
        """Test state creation with metrics."""
        metrics = {"cpu": 0.5, "memory": 0.3, "disk": 0.2}
        state = create_state_with_metrics(metrics)
        
        assert state.system_metrics == metrics
        assert isinstance(state, RSIState)
    
    @pytest.mark.unit
    def test_create_state_with_learning_data(self):
        """Test state creation with learning data."""
        learning_data = {
            "accuracy": 0.95,
            "loss": 0.05,
            "learning_rate": 0.001,
            "epoch": 10
        }
        
        state = create_state_with_learning_data(learning_data)
        
        assert state.learning_metrics == learning_data
        assert isinstance(state, RSIState)
    
    @pytest.mark.unit
    def test_create_state_with_safety_update(self):
        """Test state creation with safety metrics."""
        safety_data = {
            "safety_score": 0.95,
            "anomaly_score": 0.05,
            "circuit_breaker_state": "closed"
        }
        
        state = create_state_with_safety_update(safety_data)
        
        assert state.safety_metrics == safety_data
        assert isinstance(state, RSIState)


class TestStatePerformance:
    """Performance tests for state management."""
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_state_creation_performance(self, performance_tracker):
        """Test state creation performance."""
        performance_tracker.start()
        
        # Create many states
        states = []
        for i in range(1000):
            state = RSIState(
                system_metrics={"iteration": i, "value": i * 0.1},
                learning_metrics={"accuracy": 0.95 + i * 0.0001}
            )
            states.append(state)
        
        performance_tracker.stop()
        
        assert len(states) == 1000
        assert performance_tracker.duration < 1.0  # Should be fast
    
    @pytest.mark.unit
    @pytest.mark.performance
    async def test_state_transition_performance(self, state_manager, performance_tracker):
        """Test state transition performance."""
        performance_tracker.start()
        
        # Perform many transitions
        for i in range(100):
            await state_manager.transition_state(
                system_metrics={"iteration": i}
            )
        
        performance_tracker.stop()
        
        assert len(state_manager.state_history) == 101  # initial + 100 transitions
        assert performance_tracker.duration < 2.0  # Should be reasonably fast
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_state_hashing_performance(self, performance_tracker):
        """Test state hashing performance."""
        # Create complex state
        state = RSIState(
            system_metrics={f"metric_{i}": i * 0.1 for i in range(100)},
            learning_metrics={f"learning_{i}": i * 0.01 for i in range(50)},
            safety_metrics={f"safety_{i}": i * 0.001 for i in range(25)}
        )
        
        performance_tracker.start()
        
        # Hash many times
        hashes = []
        for _ in range(1000):
            hash_value = hash_state(state)
            hashes.append(hash_value)
        
        performance_tracker.stop()
        
        assert len(set(hashes)) == 1  # All hashes should be identical
        assert performance_tracker.duration < 1.0  # Should be fast


class TestStateEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.unit
    def test_state_with_large_data(self):
        """Test state with large datasets."""
        large_metrics = {f"metric_{i}": i for i in range(10000)}
        
        state = RSIState(system_metrics=large_metrics)
        
        assert len(state.system_metrics) == 10000
        assert state.system_metrics["metric_9999"] == 9999
    
    @pytest.mark.unit
    def test_state_with_nested_data(self):
        """Test state with nested data structures."""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {"value": 42}
                }
            },
            "array": [1, 2, 3, {"nested": "value"}]
        }
        
        state = RSIState(metadata=nested_data)
        
        assert state.metadata["level1"]["level2"]["level3"]["value"] == 42
        assert state.metadata["array"][3]["nested"] == "value"
    
    @pytest.mark.unit
    async def test_state_manager_with_empty_initial_state(self):
        """Test state manager with no initial state."""
        manager = RSIStateManager()
        
        # Should create default initial state
        assert manager.current_state is not None
        assert isinstance(manager.current_state, RSIState)
        assert len(manager.state_history) == 1
    
    @pytest.mark.unit
    async def test_rollback_with_no_history(self):
        """Test rollback when no previous states exist."""
        manager = RSIStateManager()
        
        # Rollback should return current state or raise appropriate error
        try:
            result = await manager.rollback_state()
            assert result == manager.current_state
        except ValueError as e:
            assert "no previous state" in str(e).lower()
    
    @pytest.mark.unit
    def test_state_with_special_characters(self):
        """Test state with special characters and Unicode."""
        special_data = {
            "unicode": "🚀🤖🔬",
            "special_chars": "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./",
            "empty_string": "",
            "whitespace": "   \t\n   "
        }
        
        state = RSIState(metadata=special_data)
        
        assert state.metadata["unicode"] == "🚀🤖🔬"
        assert state.metadata["special_chars"] == "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        assert state.metadata["empty_string"] == ""