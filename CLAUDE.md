# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Hephaestus is a production-ready **Recursive Self-Improvement (RSI) AI system** that implements comprehensive safety measures, monitoring, and defensive security practices. The system is designed with a **safety-first architecture** that prioritizes reliability over performance and implements multiple layers of protection.

## Development Commands

### Installation and Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install additional ML/optimization packages (many may fail - system degrades gracefully)
pip install restrictedpython pyod psutil docker
pip install torch pytorch-lightning stable-baselines3 gymnasium
pip install optuna ray[tune] transformers chromadb faiss-cpu
```

### Running the System
```bash
# Start the main RSI orchestrator and API server (port 8000)
python -m src.main

# Run comprehensive examples demonstrating all features
python example_usage.py

# Run integration tests
python test_integrated_system.py
python test_memory_system.py
python test_river_models.py

# Test individual components with quick validation
python -c "
import asyncio
from src.main import RSIOrchestrator

async def test():
    orchestrator = RSIOrchestrator(environment='development')
    await orchestrator.start()
    result = await orchestrator.predict({'x': 1.0, 'y': 2.0})
    print(f'Test prediction: {result}')
    await orchestrator.stop()

asyncio.run(test())
"
```

### Development and Testing
```bash
# Run test files (no formal test framework - use Python files directly)
python test_*.py

# Code formatting and linting
black src/
ruff src/
mypy src/

# Check system health and endpoints
curl "http://localhost:8000/health"
curl "http://localhost:8000/metrics" 
curl "http://localhost:8000/performance"
```

## Architecture Overview

### Core Design Principles
1. **Safety-First**: Multiple layers of protection prevent system corruption and security breaches
2. **Immutable State**: All state changes are managed through pyrsistent for structural sharing and corruption prevention
3. **Circuit Breakers**: Automatic failure detection and recovery prevent cascading failures
4. **Comprehensive Validation**: Every input is validated through multiple layers using Pydantic v2
5. **Secure Execution**: Multi-layer sandboxing (RestrictedPython, Docker, subprocess) for safe code execution
6. **Complete Audit Trail**: All operations are logged with cryptographic integrity verification

### System Components

The system is organized into eight main subsystems with advanced learning capabilities:

#### 1. Core System (`src/core/`)
- **State Management** (`state.py`): Immutable state with pyrsistent, providing O(log n) operations with structural sharing
- **Model Versioning** (`model_versioning.py`): MLflow-integrated model lifecycle management with comprehensive metadata tracking

#### 2. Learning System (`src/learning/`)
- **Online Learning** (`online_learning.py`): River-based incremental learning with concept drift detection
- Supports ensemble learning with multiple algorithms (LogisticRegression, ADWINBagging, HoeffdingTree)
- Real-time adaptation to concept drift with automatic model retraining

#### 3. Validation System (`src/validation/`)
- **Comprehensive Validators** (`validators.py`): Pydantic v2-based validation with custom safety constraints
- Validates model weights, learning configurations, code safety, and performance metrics
- Implements defense-in-depth with multiple validation layers

#### 4. Safety System (`src/safety/`)
- **Circuit Breakers** (`circuits.py`): PyBreaker-based failure prevention with Redis backing
- Automatic failure detection, circuit opening, and recovery with configurable thresholds
- Distributed circuit state management for scalability

#### 5. Security System (`src/security/`)
- **Execution Sandbox** (`sandbox.py`): Multi-layer sandboxing with RestrictedPython, Docker, and subprocess isolation
- AST-based code analysis, resource limits, and timeout protection
- Fallback mechanisms ensure code execution even when primary sandbox fails

#### 6. Monitoring System (`src/monitoring/`)
- **Telemetry** (`telemetry.py`): OpenTelemetry integration for distributed tracing and metrics
- **Anomaly Detection** (`anomaly_detection.py`): PyOD-based behavioral monitoring with 50+ algorithms
- **Audit Logging** (`audit_logger.py`): Cryptographically-verified audit trail with Loguru integration

#### 7. Advanced Learning Systems (`src/learning/`)
- **Meta-Learning** (`meta_learning.py`): Learn2Learn integration with MAML, ProtoNets, and Meta-SGD
- **Lightning Orchestrator** (`lightning_orchestrator.py`): PyTorch Lightning multi-task learning with distributed training
- **Reinforcement Learning** (`reinforcement_learning.py`): Stable-Baselines3 integration for continuous policy improvement
- **Continual Learning** (`continual_learning.py`): Avalanche framework with catastrophic forgetting prevention

#### 8. Optimization Systems (`src/optimization/`)
- **Optuna Optimizer** (`optuna_optimizer.py`): Bayesian optimization with 2-3x performance improvements
- **Ray Tune Orchestrator** (`ray_tune_optimizer.py`): Distributed optimization with linear scaling across 100+ nodes

### Main Orchestrator (`src/main.py`)
The `RSIOrchestrator` class coordinates all components and provides:
- FastAPI-based REST API for external interaction
- Async lifecycle management with proper startup/shutdown
- Background monitoring and self-improvement loops
- Comprehensive health checking and performance analysis

## Key Implementation Details

### Immutable State Management
The system uses pyrsistent for all state management, ensuring that:
- State corruption is impossible through structural sharing
- All state transitions are atomic and traceable
- Historical states are preserved for rollback capabilities
- Performance is optimized through O(log n) operations

### Safety Architecture
The system implements a multi-layer safety approach:
```
Input → Validation → Circuit Breaker → Execution → Monitoring → Audit
  ↓         ↓            ↓              ↓           ↓         ↓
Schema   Business    Failure         Sandboxed   Anomaly   Crypto
Check    Logic      Prevention      Execution   Detection  Verify
```

### Online Learning Pipeline
```
Features → Validation → Prediction → Learning → State Update → Drift Detection
    ↓          ↓            ↓          ↓            ↓             ↓
  Safety    Performance   Ensemble   Concept     Immutable    Adaptation
  Check     Metrics       Voting     Drift       State        Strategy
```

## Development Guidelines

### Critical Library Compatibility Patterns

**Pydantic v2 Requirements:**
- Use `field_validator` instead of `@validator`
- Use `model_config = {"frozen": True}` instead of `Config` class
- Use `pattern` instead of `regex` in Field definitions
- Use `mode='before'` for pre-validation transforms

**River ML Library (v0.21.0+):**
- Use `optimizer=optim.SGD(lr=learning_rate)` instead of `learning_rate` parameter
- Use `ADWINBaggingClassifier` instead of `AdaptiveRandomForestClassifier`
- Use `LogLoss()` instead of `Log()` for loss metrics
- Use `dummy` drift detector instead of `ddm` (not available)
- Import structure: `from river import linear_model, tree, ensemble, drift, metrics`

**FastAPI + AsyncIO Patterns:**
- All main operations are async/await based
- Use `RSIOrchestrator(environment='development')` for initialization
- Always call `await orchestrator.start()` before operations
- Always call `await orchestrator.stop()` for cleanup

**Memory System Integration:**
- Core components: episodic, semantic, procedural, working memory
- Memory hierarchy automatically consolidates and retrieves information  
- Access through `memory_manager` component in orchestrator

### Security Considerations
- All code execution must go through the sandbox system
- Never bypass validation layers
- All user inputs must be validated through the RSIValidator
- State transitions must use the immutable state manager
- All operations must be logged through the audit system

### Error Handling
The system implements comprehensive error handling:
- Circuit breakers prevent cascading failures
- Validation errors are caught and logged
- Sandbox failures trigger fallback mechanisms
- All errors are recorded in the audit trail

## Testing Strategy

### Component Testing
Each major component should be tested independently:
- State management with various transition scenarios
- Validation with both valid and invalid inputs
- Sandbox execution with safe and dangerous code
- Circuit breakers under various failure conditions
- Anomaly detection with normal and anomalous patterns

### Integration Testing
The system should be tested as a whole:
- Full prediction/learning cycles
- System health under various load conditions
- Failure recovery scenarios
- Security penetration testing

### Performance Testing
Key performance metrics to monitor:
- Request throughput (target: 10,000+ requests/second)
- Learning latency (target: sub-millisecond)
- Memory overhead from immutable structures (~10%)
- Safety validation overhead (<5%)

## Common Issues and Solutions

### Import/Dependency Errors
```bash
# If OpenTelemetry packages fail, disable telemetry:
export RSI_ENABLE_TELEMETRY=false

# If codejail/Docker unavailable, system falls back to RestrictedPython
# If advanced ML libraries fail, core River functionality still works

# Check what packages are missing:
python -c "
try:
    import torch, optuna, ray
    print('✓ Advanced ML libraries available')
except ImportError as e:
    print(f'⚠ Advanced features limited: {e}')
"
```

### Performance and State Issues
```bash
# Check circuit breaker states
curl "http://localhost:8000/metrics" | grep circuit

# Monitor memory usage patterns
curl "http://localhost:8000/performance" | grep memory

# Reset state if corrupted (DANGER: loses all learning)
rm -f rsi_continuous_state.json episodic_memory.db procedural_memory.pkl
```

### Testing Specific Components
```python
# Test individual systems without full orchestrator
from src.learning.online_learning import create_classification_learner
from src.memory.memory_manager import RSIMemoryManager
from src.safety.circuits import CircuitBreakerManager

# Each component can be tested independently
learner = create_classification_learner()
memory = RSIMemoryManager()
```

### Database and Storage Issues
- SQLite databases: `rsi_system.db`, `episodic_memory.db`, `model_registry.db`
- Memory files: `procedural_memory.pkl`, `rsi_continuous_state.json`
- Logs in: `logs/development/` and `logs/production/`
- Safety checkpoints in: `safety_checkpoints/`

## System Monitoring

### Health Endpoints
- `/health` - Overall system health
- `/metrics` - Detailed system metrics
- `/alerts` - Active anomaly alerts
- `/performance` - Performance analysis

### Key Metrics to Monitor
- Circuit breaker states (open/closed/half-open)
- Anomaly detection alerts
- Learning accuracy and concept drift
- Resource usage and limits
- Audit log integrity

The system is designed to be self-monitoring and self-healing, with comprehensive logging and alerting to ensure safe operation in production environments.

## Key Implementation Patterns

### State Management Flow
```python
# All state changes go through immutable pyrsistent structures
from src.core.state import RSIStateManager, add_learning_record

state_manager = RSIStateManager() 
new_state = add_learning_record(current_state, learning_data)
# Old state preserved, new state created atomically
```

### Safety-First Execution Pattern
```python
# Every operation follows: Validate → Execute → Monitor → Audit
async def safe_operation(data):
    # 1. Validate inputs
    validation = await validator.validate(data)
    if not validation.is_valid:
        raise ValidationError(validation.errors)
    
    # 2. Check circuit breakers
    async with circuit_breaker:
        # 3. Execute with monitoring
        result = await monitored_execution(data)
        
    # 4. Audit results
    await audit_logger.log_operation(result)
    return result
```

### Learning System Integration
```python
# Multiple learning systems work together through orchestrator
from src.main import RSIOrchestrator

orchestrator = RSIOrchestrator()
# Coordinates: online_learning, meta_learning, continual_learning, RL
# Memory: episodic, semantic, procedural, working
# Safety: circuits, validation, monitoring, audit
```

### Common Integration Points
- **Entry Point**: `src/main.py:RSIOrchestrator` coordinates all subsystems
- **State**: `src/core/state.py` provides immutable state management
- **Safety**: `src/safety/circuits.py` prevents failures, `src/validation/validators.py` ensures input integrity  
- **Learning**: `src/learning/online_learning.py` for core ML, others for advanced capabilities
- **Memory**: `src/memory/memory_manager.py` orchestrates hierarchical memory systems
- **Security**: `src/security/sandbox.py` provides multi-layer code execution safety