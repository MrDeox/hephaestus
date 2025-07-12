# Getting Started with Hephaestus RSI

This guide will help you get the Hephaestus Recursive Self-Improvement system up and running.

## Prerequisites

### System Requirements
- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB disk space
- Docker (optional, for enhanced sandbox)
- Redis (optional, for distributed circuit breakers)

### Platform Support
- Linux (Ubuntu 20.04+, CentOS 8+)
- macOS (10.15+)
- Windows 10/11 (with WSL2 recommended)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd hephaestus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example configuration:
```bash
cp config/development.yml.example config/development.yml
```

Edit `config/development.yml` to match your environment:
```yaml
environment: development

database:
  url: "sqlite:///data/hephaestus.db"

security:
  sandbox:
    security_level: "standard"
    enable_docker: false

monitoring:
  telemetry:
    enabled: true
    endpoint: "console"
```

### 3. Initialize the System

```bash
# Create data directories
mkdir -p data/{models,cache,logs,temp,backups}

# Run initialization
python -c "
import asyncio
from src.main import RSIOrchestrator

async def init():
    orchestrator = RSIOrchestrator(environment='development')
    await orchestrator.initialize()
    print('System initialized successfully!')
    await orchestrator.stop()

asyncio.run(init())
"
```

### 4. Start the System

```bash
# Start the main orchestrator
python -m src.main
```

The system will start on `http://localhost:8000` by default.

### 5. Verify Installation

Check system health:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:00:00Z",
  "components": {
    "state_manager": "healthy",
    "learning_system": "healthy",
    "safety_system": "healthy",
    "security_system": "healthy"
  }
}
```

## Basic Usage

### Making Predictions

```python
import asyncio
import aiohttp

async def make_prediction():
    async with aiohttp.ClientSession() as session:
        data = {
            "features": [1.0, 2.0, 3.0],
            "model_type": "classification"
        }
        
        async with session.post(
            'http://localhost:8000/predict',
            json=data
        ) as response:
            result = await response.json()
            print(f"Prediction: {result}")

asyncio.run(make_prediction())
```

### Triggering Learning

```python
import asyncio
import aiohttp

async def trigger_learning():
    async with aiohttp.ClientSession() as session:
        data = {
            "training_data": [[1, 0], [2, 1], [3, 1]],
            "labels": [0, 1, 1],
            "model_type": "classification"
        }
        
        async with session.post(
            'http://localhost:8000/learn',
            json=data
        ) as response:
            result = await response.json()
            print(f"Learning result: {result}")

asyncio.run(trigger_learning())
```

### Monitoring System

View system metrics:
```bash
curl http://localhost:8000/metrics
```

Check performance statistics:
```bash
curl http://localhost:8000/admin/performance
```

## Configuration Options

### Environment Configurations

The system supports three environments:

1. **Development** (`config/development.yml`)
   - Minimal security for rapid development
   - Local file-based storage
   - Console logging

2. **Testing** (`config/testing.yml`)
   - Isolated test databases
   - Mock external services
   - Comprehensive test coverage

3. **Production** (`config/production.yml`)
   - Maximum security enforcement
   - Distributed databases
   - Full monitoring and alerting

### Key Configuration Sections

#### Security Settings
```yaml
security:
  sandbox:
    security_level: "high"  # minimal, standard, high, maximum
    enable_docker: true
    enable_threat_detection: true
  
  encryption:
    enabled: true
    key_rotation_days: 30
```

#### Learning Configuration
```yaml
learning:
  online_learning:
    enabled: true
    algorithms: ["logistic", "bagging", "tree"]
    drift_detection: true
  
  meta_learning:
    enabled: true
    algorithms: ["maml", "prototypical"]
    few_shot_support: true
```

#### Resource Limits
```yaml
resources:
  limits:
    max_memory_mb: 4096
    max_cpu_percent: 80.0
    max_file_handles: 1000
    max_threads: 50
```

## Development Setup

### IDE Configuration

For VS Code, create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.pylintEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m security
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/
```

## Common Operations

### Backup and Restore

Create backup:
```bash
python -c "
import asyncio
from src.main import RSIOrchestrator

async def backup():
    orchestrator = RSIOrchestrator()
    await orchestrator.initialize()
    backup_id = await orchestrator.create_backup()
    print(f'Backup created: {backup_id}')
    await orchestrator.stop()

asyncio.run(backup())
"
```

Restore from backup:
```bash
python -c "
import asyncio
from src.main import RSIOrchestrator

async def restore():
    orchestrator = RSIOrchestrator()
    await orchestrator.initialize()
    await orchestrator.restore_from_backup('backup_id_here')
    print('System restored')
    await orchestrator.stop()

asyncio.run(restore())
"
```

### Monitoring and Debugging

View system logs:
```bash
tail -f data/logs/hephaestus.log
```

Monitor resource usage:
```bash
python -c "
from src.common.resource_manager import get_resource_manager
manager = get_resource_manager()
print(manager.get_usage_summary())
"
```

Check circuit breaker status:
```bash
python -c "
from src.safety.circuits import CircuitBreakerManager
manager = CircuitBreakerManager()
print(manager.get_circuit_status())
"
```

## Troubleshooting

### Common Issues

#### Import Errors
- **Problem**: Missing dependencies
- **Solution**: `pip install -r requirements.txt`

#### Memory Issues
- **Problem**: System runs out of memory
- **Solution**: Reduce `max_memory_mb` in configuration or increase system RAM

#### Permission Errors
- **Problem**: Cannot write to data directory
- **Solution**: `chmod -R 755 data/` or run with appropriate permissions

#### Docker Issues
- **Problem**: Docker sandbox not working
- **Solution**: Install Docker and ensure user is in docker group

#### Performance Issues
- **Problem**: Slow response times
- **Solution**: Check resource limits and system load, consider scaling

### Debug Mode

Enable debug mode for detailed logging:
```bash
export HEPHAESTUS_DEBUG=true
python -m src.main
```

### Log Levels

Adjust log levels in configuration:
```yaml
logging:
  level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "detailed"  # simple, detailed, json
```

## Next Steps

1. **Read the Architecture Guide**: `docs/ARCHITECTURE.md`
2. **Review Security Guidelines**: `docs/SECURITY.md`
3. **Explore API Documentation**: `docs/API.md`
4. **Check Examples**: `examples/` directory
5. **Join the Community**: Links to forums, Discord, etc.

## Support

- **Documentation**: See `docs/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Security**: security@yourorg.com