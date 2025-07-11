"""
Advanced RSI Learning System Demo.
Demonstrates all the implemented learning capabilities including meta-learning,
continual learning, reinforcement learning, and distributed optimization.
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Any
from datetime import datetime
import logging
import json
from pathlib import Path

# Import RSI components
from src.main import RSIOrchestrator
from src.learning.meta_learning import (
    create_meta_learning_system, 
    MetaLearningAlgorithm, 
    create_few_shot_task
)
from src.learning.lightning_orchestrator import (
    create_lightning_orchestrator,
    TaskType
)
from src.learning.reinforcement_learning import (
    create_rl_system,
    RLAlgorithm,
    RSITaskType
)
from src.learning.continual_learning import (
    create_continual_learning_system,
    ContinualLearningStrategy,
    create_task_sequence
)
from src.optimization.optuna_optimizer import (
    create_optuna_optimizer,
    create_ml_objective,
    OptimizationObjective
)
from src.optimization.ray_tune_optimizer import (
    create_ray_tune_orchestrator,
    create_ml_ray_objective,
    SearchAlgorithm,
    SchedulerType,
    HYPERPARAMETER_SEARCH_SPACES
)
from src.core.state import StateManager, RSIState
from src.validation.validators import create_strict_validator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRSIDemo:
    """Advanced RSI learning system demonstration."""
    
    def __init__(self):
        self.orchestrator = None
        self.state_manager = None
        self.validator = None
        self.results = {}
        
    async def initialize(self):
        """Initialize the RSI system."""
        print("🚀 Initializing Advanced RSI Learning System...")
        
        # Initialize core components
        self.orchestrator = RSIOrchestrator(environment="development")
        await self.orchestrator.start()
        
        self.state_manager = self.orchestrator.state_manager
        self.validator = self.orchestrator.validator
        
        print("✅ RSI system initialized successfully!")
        
    async def demo_meta_learning(self):
        """Demonstrate meta-learning capabilities."""
        print("\n🧠 === Meta-Learning Demo ===")
        
        # Create meta-learning system
        meta_system = create_meta_learning_system(
            algorithm=MetaLearningAlgorithm.MAML,
            state_manager=self.state_manager,
            validator=self.validator,
            num_ways=3,
            num_shots=5,
            adaptation_steps=5
        )
        
        # Generate synthetic few-shot tasks
        tasks = []
        for task_id in range(10):
            # Support set
            support_data = []
            for class_id in range(3):
                for shot in range(5):
                    features = np.random.randn(10) + class_id * 2  # Class-specific distribution
                    support_data.append({
                        'features': features.tolist(),
                        'label': class_id
                    })
            
            # Query set
            query_data = []
            for class_id in range(3):
                for query in range(15):
                    features = np.random.randn(10) + class_id * 2
                    query_data.append({
                        'features': features.tolist(),
                        'label': class_id
                    })
            
            task = create_few_shot_task(support_data, query_data, 3)
            tasks.append(task)
        
        # Meta-training
        print("🔄 Running meta-training...")
        for episode in range(5):
            batch_tasks = np.random.choice(tasks, 4, replace=False).tolist()
            metrics = await meta_system.meta_train_step(batch_tasks)
            print(f"Episode {episode}: Meta-loss = {metrics['meta_loss']:.4f}, "
                  f"Query accuracy = {metrics['query_accuracy']:.3f}")
        
        # Few-shot learning on new task
        print("🎯 Testing few-shot learning...")
        test_task = tasks[-1]
        results = await meta_system.few_shot_learn(
            test_task['support']['features'],
            test_task['query']['features']
        )
        
        print(f"Few-shot accuracy: {results['query_accuracy']:.3f}")
        print(f"Learning efficiency: {results['learning_efficiency']:.4f}")
        
        self.results['meta_learning'] = results
        
    async def demo_continual_learning(self):
        """Demonstrate continual learning capabilities."""
        print("\n🔄 === Continual Learning Demo ===")
        
        # Create continual learning system
        cl_system = create_continual_learning_system(
            strategy=ContinualLearningStrategy.EWC,
            input_size=20,
            output_size=10,
            state_manager=self.state_manager,
            validator=self.validator,
            memory_size=500,
            ewc_lambda=0.4
        )
        
        # Generate synthetic task sequence
        print("📊 Generating synthetic task sequence...")
        all_data = []
        for i in range(5000):
            features = np.random.randn(20)
            label = np.random.randint(0, 10)
            all_data.append({
                'features': features.tolist(),
                'label': label
            })
        
        tasks = create_task_sequence(all_data, num_tasks=5, samples_per_task=1000)
        
        # Train on task sequence
        task_results = []
        for task_id, task_data in enumerate(tasks):
            print(f"🎯 Learning task {task_id + 1}/5...")
            
            # Split into train/validation
            train_data = task_data[:800]
            val_data = task_data[800:]
            
            # Learn task
            result = await cl_system.learn_task(train_data, task_id, val_data)
            task_results.append(result)
            
            print(f"Task {task_id + 1} accuracy: {result['val_accuracy']:.2f}%")
            
            # Show memory efficiency
            memory_stats = cl_system.get_memory_efficiency()
            print(f"Memory usage: {memory_stats['memory_usage_percent']:.1f}%")
        
        # Evaluate forgetting
        print("📈 Evaluating backward transfer...")
        backward_results = await cl_system._evaluate_backward_transfer()
        
        if backward_results:
            avg_accuracy = backward_results.get('average_backward_accuracy', 0)
            print(f"Average backward accuracy: {avg_accuracy:.2f}%")
        
        self.results['continual_learning'] = {
            'task_results': task_results,
            'backward_transfer': backward_results,
            'memory_efficiency': cl_system.get_memory_efficiency()
        }
        
    async def demo_reinforcement_learning(self):
        """Demonstrate reinforcement learning capabilities."""
        print("\n🎮 === Reinforcement Learning Demo ===")
        
        # Create RL system for hyperparameter optimization
        rl_system = create_rl_system(
            task_type=RSITaskType.HYPERPARAMETER_OPTIMIZATION,
            algorithm=RLAlgorithm.PPO,
            state_manager=self.state_manager,
            validator=self.validator,
            total_timesteps=10000,
            n_envs=2
        )
        
        # Train RL agent
        print("🤖 Training RL agent...")
        training_results = await rl_system.train()
        
        print(f"Training completed:")
        print(f"  Mean reward: {training_results['mean_reward']:.2f}")
        print(f"  Std reward: {training_results['std_reward']:.2f}")
        
        # Use trained agent for optimization
        print("🔧 Using RL agent for hyperparameter optimization...")
        optimization_results = await rl_system.optimize_task({
            'model_type': 'neural_network',
            'target_accuracy': 0.9
        })
        
        print(f"Optimization results:")
        print(f"  Best reward: {optimization_results['best_reward']:.2f}")
        print(f"  Best config: {optimization_results['best_config']}")
        
        self.results['reinforcement_learning'] = {
            'training_results': training_results,
            'optimization_results': optimization_results
        }
        
    async def demo_hyperparameter_optimization(self):
        """Demonstrate hyperparameter optimization."""
        print("\n🔍 === Hyperparameter Optimization Demo ===")
        
        # Optuna optimization
        print("🔬 Running Optuna optimization...")
        optuna_optimizer = create_optuna_optimizer(
            study_name="rsi_neural_network",
            objective_type=OptimizationObjective.MAXIMIZE,
            n_trials=20,
            state_manager=self.state_manager,
            validator=self.validator
        )
        
        # Create dummy training data
        train_data = [{'features': np.random.randn(10).tolist(), 'label': np.random.randint(0, 2)} 
                     for _ in range(1000)]
        val_data = [{'features': np.random.randn(10).tolist(), 'label': np.random.randint(0, 2)} 
                   for _ in range(200)]
        
        # Create objective function
        objective = create_ml_objective(
            train_data=train_data,
            val_data=val_data,
            model_class=torch.nn.Linear,  # Dummy model class
            state_manager=self.state_manager,
            validator=self.validator
        )
        
        # Run optimization
        optuna_results = await optuna_optimizer.optimize(objective)
        
        print(f"Optuna optimization completed:")
        print(f"  Best value: {optuna_results['best_value']:.4f}")
        print(f"  Best params: {optuna_results['best_params']}")
        print(f"  Total trials: {optuna_results['n_trials']}")
        
        self.results['hyperparameter_optimization'] = {
            'optuna_results': optuna_results
        }
        
    async def demo_distributed_optimization(self):
        """Demonstrate distributed optimization with Ray Tune."""
        print("\n☁️ === Distributed Optimization Demo ===")
        
        try:
            # Ray Tune optimization
            print("⚡ Running Ray Tune distributed optimization...")
            ray_optimizer = create_ray_tune_orchestrator(
                experiment_name="rsi_distributed_optimization",
                search_algorithm=SearchAlgorithm.BAYESOPT,
                scheduler_type=SchedulerType.ASHA,
                num_samples=10,
                state_manager=self.state_manager,
                validator=self.validator,
                max_concurrent_trials=2
            )
            
            # Create dummy training data
            train_data = [{'features': np.random.randn(10).tolist(), 'label': np.random.randint(0, 2)} 
                         for _ in range(1000)]
            val_data = [{'features': np.random.randn(10).tolist(), 'label': np.random.randint(0, 2)} 
                       for _ in range(200)]
            
            # Create objective function
            objective = create_ml_ray_objective(
                train_data=train_data,
                val_data=val_data,
                model_class=torch.nn.Linear,
                state_manager=self.state_manager,
                validator=self.validator
            )
            
            # Use neural network search space
            search_space = HYPERPARAMETER_SEARCH_SPACES["neural_network"]
            
            # Run distributed optimization
            ray_results = await ray_optimizer.optimize(objective, search_space)
            
            print(f"Ray Tune optimization completed:")
            print(f"  Best accuracy: {ray_results['best_config']}")
            print(f"  Total trials: {ray_results['total_trials']}")
            print(f"  Successful trials: {ray_results['successful_trials']}")
            
            self.results['distributed_optimization'] = {
                'ray_results': ray_results
            }
            
        except Exception as e:
            print(f"⚠️ Distributed optimization skipped: {e}")
            self.results['distributed_optimization'] = {
                'error': str(e),
                'message': "Ray not available or initialization failed"
            }
        
    async def demo_multi_task_learning(self):
        """Demonstrate multi-task learning with PyTorch Lightning."""
        print("\n🎯 === Multi-Task Learning Demo ===")
        
        # Define task configurations
        task_configs = {
            'classification_task': {
                'type': TaskType.CLASSIFICATION,
                'output_dim': 3,
                'weight': 1.0
            },
            'regression_task': {
                'type': TaskType.REGRESSION,
                'output_dim': 1,
                'weight': 0.5
            },
            'input_dim': 20
        }
        
        # Create Lightning orchestrator
        lightning_orchestrator = create_lightning_orchestrator(
            task_configs=task_configs,
            state_manager=self.state_manager,
            validator=self.validator
        )
        
        # Generate synthetic multi-task data
        print("📊 Generating multi-task training data...")
        train_data = []
        for i in range(1000):
            features = np.random.randn(20).tolist()
            classification_label = np.random.randint(0, 3)
            regression_value = np.random.randn()
            
            train_data.append({
                'features': features,
                'classification_task': classification_label,
                'regression_task': regression_value
            })
        
        val_data = []
        for i in range(200):
            features = np.random.randn(20).tolist()
            classification_label = np.random.randint(0, 3)
            regression_value = np.random.randn()
            
            val_data.append({
                'features': features,
                'classification_task': classification_label,
                'regression_task': regression_value
            })
        
        # Train multi-task model
        print("🔄 Training multi-task model...")
        training_results = await lightning_orchestrator.train(train_data, val_data)
        
        print(f"Multi-task training completed:")
        print(f"  Best model score: {training_results['best_model_score']:.4f}")
        print(f"  Epochs trained: {training_results['current_epoch']}")
        
        # Make predictions
        print("🎯 Making multi-task predictions...")
        test_data = [{'features': np.random.randn(20).tolist()} for _ in range(10)]
        predictions = await lightning_orchestrator.predict(test_data)
        
        print(f"Predictions generated for {predictions['num_samples']} samples")
        
        self.results['multi_task_learning'] = {
            'training_results': training_results,
            'predictions': predictions
        }
        
    async def demo_system_integration(self):
        """Demonstrate integrated RSI system capabilities."""
        print("\n🔗 === System Integration Demo ===")
        
        # Use main orchestrator for integrated capabilities
        print("🔄 Testing integrated learning...")
        
        # Generate sample data
        features = {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.25,
            'wind_speed': 10.0
        }
        
        # Make prediction
        prediction = await self.orchestrator.predict(features)
        print(f"Prediction: {prediction['prediction']:.3f} (confidence: {prediction['confidence']:.3f})")
        
        # Learn from new data
        learning_result = await self.orchestrator.learn(features, 1.0)
        print(f"Learning accuracy: {learning_result['accuracy']:.3f}")
        
        # Execute safe code
        safe_code = """
import math
import numpy as np

# Generate synthetic data
data = np.random.randn(100)
mean = np.mean(data)
std = np.std(data)

# Calculate confidence interval
confidence_interval = [mean - 1.96 * std, mean + 1.96 * std]

print(f"Data statistics:")
print(f"  Mean: {mean:.3f}")
print(f"  Std: {std:.3f}")
print(f"  95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
"""
        
        code_result = await self.orchestrator.execute_code(safe_code)
        print(f"Code execution status: {code_result['status']}")
        
        # Check system health
        health = await self.orchestrator.get_system_health()
        print(f"System health: {health['overall_status']}")
        
        # Analyze performance
        performance = await self.orchestrator.analyze_performance()
        print(f"Learning accuracy: {performance['metrics']['learning']['accuracy']:.3f}")
        
        self.results['system_integration'] = {
            'prediction': prediction,
            'learning_result': learning_result,
            'code_execution': code_result,
            'system_health': health,
            'performance': performance
        }
        
    async def save_results(self):
        """Save demonstration results."""
        print("\n💾 === Saving Results ===")
        
        # Create results directory
        results_dir = Path("./demo_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"advanced_rsi_demo_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        
        # Print summary
        print("\n📊 === Demo Summary ===")
        for component, results in self.results.items():
            print(f"✅ {component.replace('_', ' ').title()}: Completed")
        
        print(f"\nTotal components demonstrated: {len(self.results)}")
        
    async def cleanup(self):
        """Cleanup resources."""
        print("\n🧹 === Cleanup ===")
        
        if self.orchestrator:
            await self.orchestrator.stop()
            print("✅ RSI orchestrator stopped")
        
        print("✅ Cleanup completed")
        
    async def run_full_demo(self):
        """Run the complete demonstration."""
        try:
            await self.initialize()
            
            # Run all demonstrations
            await self.demo_meta_learning()
            await self.demo_continual_learning()
            await self.demo_reinforcement_learning()
            await self.demo_hyperparameter_optimization()
            await self.demo_distributed_optimization()
            await self.demo_multi_task_learning()
            await self.demo_system_integration()
            
            # Save results
            await self.save_results()
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.exception("Demo failed")
        finally:
            await self.cleanup()


async def main():
    """Main demo function."""
    print("🎉 Welcome to the Advanced RSI Learning System Demo!")
    print("This demo showcases state-of-the-art recursive self-improvement capabilities.")
    print("=" * 70)
    
    demo = AdvancedRSIDemo()
    await demo.run_full_demo()
    
    print("\n🎉 Demo completed successfully!")
    print("Thank you for exploring the Advanced RSI Learning System!")


if __name__ == "__main__":
    asyncio.run(main())