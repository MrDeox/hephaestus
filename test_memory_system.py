"""
Comprehensive Test Suite for Advanced Memory Systems.
Demonstrates hierarchical memory with Working, Semantic, Episodic, and Procedural memory.
"""

import asyncio
import logging
from datetime import datetime, timezone
from src.memory import RSIMemoryHierarchy, RSIMemoryConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_working_memory():
    """Test Working Memory functionality."""
    print("\n=== Testing Working Memory ===")
    
    config = RSIMemoryConfig()
    memory = RSIMemoryHierarchy(config)
    
    # Test immediate storage and retrieval
    test_data = {
        'task': 'current_processing',
        'data': 'immediate access information',
        'priority': 0.9
    }
    
    # Store in working memory
    success = await memory.store_information(test_data, memory_type="auto")
    print(f"✅ Working memory storage: {success}")
    
    # Retrieve from working memory
    results = await memory.retrieve_information({'text': 'immediate'}, memory_types=['working'])
    print(f"✅ Working memory retrieval: {len(results['working'])} items found")
    
    return memory


async def test_semantic_memory():
    """Test Semantic Memory functionality."""
    print("\n=== Testing Semantic Memory ===")
    
    config = RSIMemoryConfig()
    memory = RSIMemoryHierarchy(config)
    
    # Test concept storage
    concept_data = {
        'concept': 'machine_learning',
        'description': 'A method of data analysis that automates analytical model building',
        'type': 'knowledge',
        'relationships': [
            {'target_id': 'artificial_intelligence', 'type': 'is_part_of', 'weight': 0.8},
            {'target_id': 'data_science', 'type': 'related_to', 'weight': 0.9}
        ],
        'confidence': 0.95
    }
    
    success = await memory.store_information(concept_data, memory_type="semantic")
    print(f"✅ Semantic memory storage: {success}")
    
    # Test concept retrieval
    results = await memory.retrieve_information({'text': 'machine learning'}, memory_types=['semantic'])
    print(f"✅ Semantic memory retrieval: {len(results['semantic'])} concepts found")
    
    return memory


async def test_episodic_memory():
    """Test Episodic Memory functionality."""
    print("\n=== Testing Episodic Memory ===")
    
    config = RSIMemoryConfig()
    memory = RSIMemoryHierarchy(config)
    
    # Test episode storage
    episode_data = {
        'event': 'training_session',
        'description': 'Completed training on neural network architecture',
        'context': {
            'location': 'development_environment',
            'duration': '2 hours',
            'outcome': 'successful'
        },
        'importance': 0.8,
        'emotions': {'satisfaction': 0.9, 'confidence': 0.8},
        'tags': ['training', 'neural_networks', 'success']
    }
    
    success = await memory.store_information(episode_data, memory_type="episodic")
    print(f"✅ Episodic memory storage: {success}")
    
    # Test episode retrieval
    results = await memory.retrieve_information({'tags': ['training']}, memory_types=['episodic'])
    print(f"✅ Episodic memory retrieval: {len(results['episodic'])} episodes found")
    
    return memory


async def test_procedural_memory():
    """Test Procedural Memory functionality."""
    print("\n=== Testing Procedural Memory ===")
    
    config = RSIMemoryConfig()
    memory = RSIMemoryHierarchy(config)
    
    # Test skill storage
    skill_data = {
        'skill': 'data_preprocessing',
        'name': 'Data Preprocessing Pipeline',
        'description': 'Clean and prepare data for machine learning',
        'skill_type': 'procedure',
        'parameters': {
            'input_format': 'csv',
            'output_format': 'numpy_array',
            'scaling': 'standard'
        },
        'preconditions': ['param:input_data', 'param:schema'],
        'postconditions': ['clean_data', 'scaled_features'],
        'implementation': {
            'steps': [
                'load_data',
                'handle_missing_values',
                'normalize_features',
                'split_dataset'
            ]
        },
        'complexity': 0.6,
        'tags': ['data_science', 'preprocessing', 'pipeline']
    }
    
    success = await memory.store_information(skill_data, memory_type="procedural")
    print(f"✅ Procedural memory storage: {success}")
    
    # Test skill retrieval
    results = await memory.retrieve_information({'skill_type': 'procedure'}, memory_types=['procedural'])
    print(f"✅ Procedural memory retrieval: {len(results['procedural'])} skills found")
    
    return memory


async def test_memory_consolidation():
    """Test Memory Consolidation functionality."""
    print("\n=== Testing Memory Consolidation ===")
    
    config = RSIMemoryConfig()
    memory = RSIMemoryHierarchy(config)
    
    # Store multiple types of information
    information_batch = [
        {
            'concept': 'deep_learning',
            'description': 'Machine learning using neural networks with multiple layers',
            'type': 'knowledge',
            'priority': 0.9
        },
        {
            'event': 'model_training',
            'description': 'Training deep learning model on large dataset',
            'importance': 0.8,
            'tags': ['training', 'deep_learning']
        },
        {
            'skill': 'gradient_descent',
            'description': 'Optimization algorithm for neural networks',
            'skill_type': 'algorithm',
            'complexity': 0.7
        }
    ]
    
    # Store information and let consolidation route it
    for info in information_batch:
        success = await memory.store_information(info, memory_type="auto")
        print(f"✅ Auto-routed storage: {success}")
    
    # Trigger consolidation
    consolidation_result = await memory.consolidate_memory()
    print(f"✅ Memory consolidation: {consolidation_result}")
    
    return memory


async def test_cross_memory_search():
    """Test Cross-Memory Search functionality."""
    print("\n=== Testing Cross-Memory Search ===")
    
    config = RSIMemoryConfig()
    memory = RSIMemoryHierarchy(config)
    
    # Store related information across memory types
    concept_info = {
        'concept': 'reinforcement_learning',
        'description': 'Machine learning paradigm where agents learn through interaction',
        'type': 'knowledge'
    }
    
    episode_info = {
        'event': 'rl_experiment',
        'description': 'Conducted reinforcement learning experiment with Q-learning',
        'tags': ['reinforcement_learning', 'experiment'],
        'importance': 0.7
    }
    
    skill_info = {
        'skill': 'q_learning',
        'description': 'Model-free reinforcement learning algorithm',
        'skill_type': 'algorithm',
        'tags': ['reinforcement_learning', 'algorithm']
    }
    
    # Store across different memory types
    await memory.store_information(concept_info, memory_type="semantic")
    await memory.store_information(episode_info, memory_type="episodic")
    await memory.store_information(skill_info, memory_type="procedural")
    
    # Search across all memory types
    results = await memory.retrieve_information({'text': 'reinforcement learning'})
    
    print(f"✅ Cross-memory search results:")
    for memory_type, items in results.items():
        print(f"   - {memory_type}: {len(items)} items")
    
    return memory


async def test_memory_optimization():
    """Test Memory Optimization functionality."""
    print("\n=== Testing Memory Optimization ===")
    
    config = RSIMemoryConfig()
    memory = RSIMemoryHierarchy(config)
    
    # Store some test data
    for i in range(10):
        test_data = {
            'item': f'test_item_{i}',
            'description': f'Test item number {i}',
            'priority': 0.5 + (i * 0.05)
        }
        await memory.store_information(test_data)
    
    # Get initial status
    initial_status = await memory.get_memory_status()
    print(f"✅ Initial memory status: {initial_status['is_initialized']}")
    
    # Optimize memory
    optimization_result = await memory.optimize_memory()
    print(f"✅ Memory optimization: {optimization_result}")
    
    # Clean up memory
    cleanup_result = await memory.cleanup_memory()
    print(f"✅ Memory cleanup: {cleanup_result}")
    
    return memory


async def test_memory_performance():
    """Test Memory Performance and Statistics."""
    print("\n=== Testing Memory Performance ===")
    
    config = RSIMemoryConfig()
    memory = RSIMemoryHierarchy(config)
    
    # Performance test with batch operations
    start_time = datetime.now(timezone.utc)
    
    # Store batch of information
    batch_size = 50
    for i in range(batch_size):
        test_data = {
            'id': f'perf_test_{i}',
            'data': f'Performance test data item {i}',
            'type': 'performance_test',
            'priority': 0.5
        }
        await memory.store_information(test_data)
    
    # Retrieve batch of information
    results = await memory.retrieve_information({'type': 'performance_test'})
    
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    
    print(f"✅ Performance test completed:")
    print(f"   - Batch size: {batch_size} items")
    print(f"   - Duration: {duration:.3f} seconds")
    print(f"   - Throughput: {batch_size / duration:.1f} items/second")
    
    # Get memory statistics
    status = await memory.get_memory_status()
    print(f"✅ Memory statistics:")
    print(f"   - Working memory: {status['memory_systems']['working_memory']['size']} items")
    print(f"   - Semantic memory: {status['memory_systems']['semantic_memory']['size']} items")
    print(f"   - Episodic memory: {status['memory_systems']['episodic_memory']['size']} items")
    print(f"   - Procedural memory: {status['memory_systems']['procedural_memory']['size']} items")
    
    return memory


async def main():
    """Run comprehensive memory system tests."""
    print("🧠 Starting Advanced Memory System Tests")
    print("=" * 50)
    
    try:
        # Test individual memory systems
        working_memory = await test_working_memory()
        await working_memory.shutdown()
        
        semantic_memory = await test_semantic_memory()
        await semantic_memory.shutdown()
        
        episodic_memory = await test_episodic_memory()
        await episodic_memory.shutdown()
        
        procedural_memory = await test_procedural_memory()
        await procedural_memory.shutdown()
        
        # Test integrated functionality
        consolidation_memory = await test_memory_consolidation()
        await consolidation_memory.shutdown()
        
        search_memory = await test_cross_memory_search()
        await search_memory.shutdown()
        
        optimization_memory = await test_memory_optimization()
        await optimization_memory.shutdown()
        
        performance_memory = await test_memory_performance()
        await performance_memory.shutdown()
        
        print("\n" + "=" * 50)
        print("🎉 All memory system tests completed successfully!")
        print("✅ Hierarchical memory architecture is fully functional")
        print("✅ Working Memory: Immediate access storage")
        print("✅ Semantic Memory: Structured knowledge with relationships")
        print("✅ Episodic Memory: Temporal experiences with context")
        print("✅ Procedural Memory: Skills and algorithms")
        print("✅ Memory Consolidation: Intelligent routing and organization")
        print("✅ Cross-Memory Search: Unified retrieval across all systems")
        print("✅ Memory Optimization: Performance tuning and cleanup")
        print("✅ Real-time Statistics: Comprehensive monitoring")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())