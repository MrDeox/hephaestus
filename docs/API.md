# Hephaestus RSI API Documentation

## Overview

The Hephaestus RSI system provides a REST API for interacting with the system's learning, prediction, and management capabilities. All endpoints return JSON responses and follow standard HTTP status codes.

**Base URL**: `http://localhost:8000` (default)

## Authentication

Currently, the API uses simple token-based authentication for development. In production, implement proper OAuth2 or JWT tokens.

```http
Authorization: Bearer your-api-token
```

## Core Endpoints

### Health Check

Check system health and status.

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:00:00Z",
  "components": {
    "state_manager": "healthy",
    "learning_system": "healthy", 
    "safety_system": "healthy",
    "security_system": "healthy",
    "monitoring_system": "healthy"
  },
  "metrics": {
    "uptime_seconds": 3600,
    "total_requests": 1234,
    "active_connections": 5
  }
}
```

**Status Codes**:
- `200`: System healthy
- `503`: System unhealthy or degraded

### System Metrics

Get detailed system metrics and performance data.

```http
GET /metrics
```

**Response**:
```json
{
  "system": {
    "memory_usage_mb": 512.3,
    "cpu_usage_percent": 25.7,
    "disk_usage_mb": 1024.5,
    "active_threads": 12
  },
  "learning": {
    "models_trained": 45,
    "predictions_made": 8934,
    "accuracy_avg": 0.94
  },
  "safety": {
    "circuit_breakers_open": 0,
    "anomalies_detected": 2,
    "threats_blocked": 0
  }
}
```

## Learning Endpoints

### Make Prediction

Submit data for prediction using trained models.

```http
POST /predict
```

**Request Body**:
```json
{
  "features": [1.0, 2.5, 3.7, 0.8],
  "model_type": "classification",
  "model_version": "latest",
  "options": {
    "return_confidence": true,
    "return_explanation": false
  }
}
```

**Response**:
```json
{
  "prediction": 1,
  "confidence": 0.87,
  "model_used": "ensemble_v1.2.3",
  "prediction_id": "pred_abc123",
  "timestamp": "2023-12-01T10:00:00Z",
  "execution_time_ms": 12.5
}
```

**Status Codes**:
- `200`: Prediction successful
- `400`: Invalid input data
- `422`: Validation error
- `503`: Model unavailable

### Trigger Learning

Submit training data to improve models.

```http
POST /learn
```

**Request Body**:
```json
{
  "training_data": [
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0]
  ],
  "labels": [0, 1, 1],
  "model_type": "classification",
  "learning_options": {
    "incremental": true,
    "validate": true,
    "save_checkpoint": true
  }
}
```

**Response**:
```json
{
  "learning_id": "learn_xyz789",
  "status": "completed",
  "model_version": "v1.2.4",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.93,
    "recall": 0.97,
    "f1_score": 0.95
  },
  "samples_processed": 3,
  "execution_time_ms": 234.7
}
```

### Get Learning Status

Check the status of a learning operation.

```http
GET /learn/{learning_id}
```

**Response**:
```json
{
  "learning_id": "learn_xyz789",
  "status": "in_progress",
  "progress_percent": 67,
  "current_step": "validation",
  "estimated_completion": "2023-12-01T10:05:00Z",
  "partial_metrics": {
    "samples_processed": 1000,
    "current_loss": 0.23
  }
}
```

## Model Management

### List Models

Get information about available models.

```http
GET /models
```

**Query Parameters**:
- `model_type`: Filter by model type
- `status`: Filter by status (active, archived, training)
- `limit`: Maximum number of results (default: 50)

**Response**:
```json
{
  "models": [
    {
      "model_id": "model_123",
      "model_type": "classification",
      "version": "v1.2.3",
      "status": "active",
      "accuracy": 0.94,
      "created_at": "2023-12-01T09:00:00Z",
      "last_updated": "2023-12-01T09:30:00Z"
    }
  ],
  "total_count": 15,
  "page": 1,
  "has_more": true
}
```

### Get Model Details

Get detailed information about a specific model.

```http
GET /models/{model_id}
```

**Response**:
```json
{
  "model_id": "model_123",
  "model_type": "classification",
  "version": "v1.2.3",
  "status": "active",
  "metadata": {
    "algorithm": "ensemble",
    "features_count": 10,
    "classes": ["class_0", "class_1"],
    "training_samples": 50000
  },
  "metrics": {
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.96,
    "f1_score": 0.94
  },
  "performance": {
    "inference_time_ms": 15.2,
    "memory_usage_mb": 128.5
  },
  "created_at": "2023-12-01T09:00:00Z",
  "last_updated": "2023-12-01T09:30:00Z"
}
```

### Deploy Model

Deploy a model version to production.

```http
POST /models/{model_id}/deploy
```

**Request Body**:
```json
{
  "deployment_strategy": "canary",
  "traffic_percentage": 10,
  "validation_required": true
}
```

## System Administration

### System Status

Get comprehensive system status.

```http
GET /admin/status
```

**Response**:
```json
{
  "system": {
    "version": "1.0.0",
    "environment": "production",
    "uptime_seconds": 86400,
    "last_restart": "2023-11-30T10:00:00Z"
  },
  "components": {
    "database": {
      "status": "healthy",
      "connections": 8,
      "query_time_avg_ms": 2.3
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.89,
      "memory_usage_mb": 256.7
    }
  },
  "resource_usage": {
    "memory_mb": 1024.5,
    "cpu_percent": 23.4,
    "disk_mb": 2048.7
  }
}
```

### Configuration

Get or update system configuration.

```http
GET /admin/config
```

```http
PUT /admin/config
```

**Request Body** (for PUT):
```json
{
  "learning": {
    "batch_size": 128,
    "learning_rate": 0.001
  },
  "security": {
    "threat_detection_enabled": true
  }
}
```

### Circuit Breakers

Get circuit breaker status.

```http
GET /admin/circuit-breakers
```

**Response**:
```json
{
  "circuit_breakers": [
    {
      "name": "learning_system",
      "state": "closed",
      "failure_count": 0,
      "last_failure": null,
      "next_attempt": null
    },
    {
      "name": "database_operations", 
      "state": "half_open",
      "failure_count": 3,
      "last_failure": "2023-12-01T09:45:00Z",
      "next_attempt": "2023-12-01T10:00:00Z"
    }
  ]
}
```

### Performance Analysis

Get detailed performance analysis.

```http
GET /admin/performance
```

**Query Parameters**:
- `hours`: Time period for analysis (default: 24)
- `operation`: Specific operation to analyze

**Response**:
```json
{
  "time_period_hours": 24,
  "summary": {
    "total_operations": 12456,
    "avg_response_time_ms": 45.7,
    "p95_response_time_ms": 120.3,
    "error_rate": 0.002
  },
  "by_operation": {
    "predict": {
      "count": 8934,
      "avg_time_ms": 15.2,
      "error_rate": 0.001
    },
    "learn": {
      "count": 156,
      "avg_time_ms": 234.7,
      "error_rate": 0.006
    }
  }
}
```

## Security Endpoints

### Threat Detection

Get threat detection status and recent threats.

```http
GET /security/threats
```

**Query Parameters**:
- `hours`: Time period for threat history (default: 24)
- `severity`: Filter by threat severity

**Response**:
```json
{
  "threat_detection": {
    "enabled": true,
    "last_scan": "2023-12-01T10:00:00Z",
    "threats_detected": 2
  },
  "recent_threats": [
    {
      "threat_id": "threat_abc123",
      "type": "code_injection",
      "severity": "high",
      "source": "unknown",
      "detected_at": "2023-12-01T09:30:00Z",
      "status": "blocked",
      "description": "Detected eval() function call"
    }
  ]
}
```

### Audit Log

Access audit log entries.

```http
GET /security/audit
```

**Query Parameters**:
- `start_date`: Start date for log entries
- `end_date`: End date for log entries
- `event_type`: Filter by event type
- `component`: Filter by component
- `limit`: Maximum number of entries

**Response**:
```json
{
  "audit_entries": [
    {
      "entry_id": "audit_123",
      "timestamp": "2023-12-01T10:00:00Z",
      "event_type": "model_training",
      "component": "learning_system",
      "user_id": "user_456",
      "details": {
        "model_type": "classification",
        "samples_count": 1000
      },
      "checksum": "sha256:abc123..."
    }
  ],
  "total_count": 5432,
  "integrity_verified": true
}
```

## Error Handling

### Standard Error Response

All errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "features",
      "reason": "Array length must be 10"
    },
    "timestamp": "2023-12-01T10:00:00Z",
    "request_id": "req_xyz789"
  }
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Input validation failed
- `MODEL_NOT_FOUND`: Requested model doesn't exist
- `SYSTEM_OVERLOAD`: System is under high load
- `SECURITY_VIOLATION`: Security policy violation
- `CIRCUIT_BREAKER_OPEN`: Circuit breaker is open
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded

## Rate Limiting

Rate limits are enforced per API key:

- **Standard endpoints**: 1000 requests/hour
- **Learning endpoints**: 100 requests/hour  
- **Admin endpoints**: 50 requests/hour

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1701432000
```

## WebSocket API

Real-time updates are available via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};

// Subscribe to specific events
ws.send(JSON.stringify({
    type: 'subscribe',
    events: ['model_training', 'system_alerts']
}));
```

## SDK Usage Examples

### Python SDK

```python
from hephaestus_client import HephaestusClient

client = HephaestusClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Make prediction
result = await client.predict(
    features=[1.0, 2.0, 3.0],
    model_type="classification"
)

# Trigger learning
learning_result = await client.learn(
    training_data=[[1, 2], [3, 4]],
    labels=[0, 1]
)
```

### JavaScript SDK

```javascript
import { HephaestusClient } from 'hephaestus-js-client';

const client = new HephaestusClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your-api-key'
});

// Make prediction
const result = await client.predict({
    features: [1.0, 2.0, 3.0],
    modelType: 'classification'
});
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **JSON**: `http://localhost:8000/openapi.json`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`