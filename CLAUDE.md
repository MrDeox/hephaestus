# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Hephaestus is a production-ready **Recursive Self-Improvement (RSI) AI system** that implements comprehensive safety measures, monitoring, and defensive security practices. The system is designed with a **safety-first architecture** that prioritizes reliability over performance and implements multiple layers of protection.

## Development Commands

### Installation and Setup
```bash
# Install dependencies (some packages may not be available and need alternatives)
pip install -r requirements.txt

# Additional packages needed for full functionality
pip install restrictedpython pyod psutil docker numpy pandas combo
pip install opentelemetry-propagator-b3

# Advanced Learning Libraries
pip install torch torchvision pytorch-lightning
pip install stable-baselines3 gymnasium learn2learn
pip install optuna ray[tune] avalanche-lib
pip install autogluon mabwiser transformers
pip install chromadb faiss-cpu hnswlib
```

### Running the System
```bash
# Start the main RSI orchestrator and API server
python -m src.main

# Run comprehensive examples demonstrating all features
python example_usage.py

# Run advanced RSI learning system demo
python examples/advanced_rsi_demo.py

# Test individual components
python -c "
import asyncio
from src.main import RSIOrchestrator

async def test():
    orchestrator = RSIOrchestrator(environment='development')
    await orchestrator.start()
    # Test predictions, learning, code execution, etc.
    await orchestrator.stop()

asyncio.run(test())
"
```

### Development and Testing
```bash
# Run tests (test framework not yet implemented)
pytest tests/

# Code formatting and linting
black src/
ruff src/
mypy src/

# Check system health
curl "http://localhost:8000/health"
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

### Pydantic v2 Compatibility
The system uses Pydantic v2 with specific patterns:
- Use `field_validator` instead of `@validator`
- Use `model_config = {"frozen": True}` instead of `Config` class
- Use `pattern` instead of `regex` in Field definitions
- Use `mode='before'` for pre-validation transforms

### River ML Library Integration
When working with River models:
- Use `optimizer=optim.SGD(lr=learning_rate)` instead of `learning_rate` parameter
- Use `ADWINBaggingClassifier` instead of `AdaptiveRandomForestClassifier`
- Use `LogLoss()` instead of `Log()` for loss metrics
- Use `dummy` drift detector instead of `ddm` (not available)

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

## Troubleshooting Common Issues

### Import Errors
- Many OpenTelemetry packages have changed APIs; disable telemetry if needed
- Some packages like `codejail` may not be available; use alternatives
- River package APIs have changed; check current documentation

### Performance Issues
- Monitor circuit breaker states for frequent trips
- Check anomaly detection for false positives
- Verify immutable state isn't causing memory issues
- Review audit logging volume

### Security Concerns
- All code execution is sandboxed by default
- Validation failures are logged and should be investigated
- Anomaly alerts may indicate security issues
- Audit trails provide complete operation history

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