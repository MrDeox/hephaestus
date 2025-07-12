"""
Base configuration system for Hephaestus RSI.

Centralizes all configuration management with validation,
environment-specific overrides, and feature flags.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml

from pydantic import BaseModel, Field, validator
from pydantic.config import ConfigDict


class Environment(str, Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecurityLevel(str, Enum):
    """Security configuration levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///hephaestus.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo: bool = False
    
    # MLflow specific
    mlflow_uri: str = "sqlite:///mlflow.db"
    
    # Vector database (ChromaDB)
    vector_db_path: str = "./data/vector_db"
    vector_db_host: Optional[str] = None
    vector_db_port: Optional[int] = None


@dataclass  
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 50
    health_check_interval: int = 30


@dataclass
class SecurityConfig:
    """Security configuration."""
    security_level: SecurityLevel = SecurityLevel.HIGH
    
    # Sandbox settings
    sandbox_timeout: int = 300
    sandbox_memory_limit: int = 512 * 1024 * 1024  # 512MB
    sandbox_cpu_limit: float = 1.0  # 1 CPU core
    
    # Allowed imports for sandbox
    allowed_imports: List[str] = field(default_factory=lambda: [
        "math", "statistics", "random", "datetime", "json", "re", 
        "collections", "itertools", "functools", "operator",
        "numpy", "pandas", "sklearn", "scipy"
    ])
    
    # Encryption settings
    encryption_key: Optional[str] = None
    hash_algorithm: str = "sha256"
    
    # Authentication
    enable_auth: bool = False
    jwt_secret: Optional[str] = None
    jwt_expiry_hours: int = 24


@dataclass
class LearningConfig:
    """Learning system configuration."""
    
    # Online learning
    online_learning_enabled: bool = True
    default_learning_rate: float = 0.001
    batch_size: int = 32
    buffer_size: int = 10000
    
    # Meta-learning
    meta_learning_enabled: bool = True
    meta_learning_interval: int = 300  # 5 minutes
    
    # Continual learning
    continual_learning_enabled: bool = True
    memory_consolidation_interval: int = 3600  # 1 hour
    
    # Reinforcement learning
    rl_enabled: bool = False  # Disabled by default for safety
    rl_exploration_rate: float = 0.1
    
    # Model versioning
    max_model_versions: int = 100
    auto_model_backup: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and telemetry configuration."""
    
    # Telemetry
    telemetry_enabled: bool = True
    telemetry_interval: int = 60  # seconds
    metrics_retention_days: int = 30
    
    # OpenTelemetry
    otel_enabled: bool = False
    otel_endpoint: Optional[str] = None
    otel_service_name: str = "hephaestus-rsi"
    
    # Anomaly detection
    anomaly_detection_enabled: bool = True
    anomaly_sensitivity: float = 0.8
    anomaly_alert_threshold: float = 0.9
    
    # Health checks
    health_check_interval: int = 30
    health_check_timeout: int = 10


@dataclass
class SafetyConfig:
    """Safety and circuit breaker configuration."""
    
    # Circuit breakers
    circuit_breaker_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 60
    circuit_expected_exception: List[str] = field(default_factory=lambda: [
        "TimeoutError", "ConnectionError", "ValidationError"
    ])
    
    # RSI safety
    rsi_safety_enabled: bool = True
    max_rsi_cycles_per_hour: int = 12
    rsi_improvement_threshold: float = 0.01  # 1% minimum improvement
    
    # Validation
    strict_validation: bool = True
    validation_timeout: int = 30


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    timeout: int = 60
    
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limiting_enabled: bool = True
    rate_limit_requests_per_minute: int = 60


@dataclass
class FeatureFlags:
    """Feature flags for gradual rollout."""
    
    # Core features
    enable_real_rsi: bool = True
    enable_meta_learning: bool = True
    enable_hypothesis_testing: bool = False
    
    # Advanced features
    enable_distributed_learning: bool = False
    enable_gpu_acceleration: bool = False
    enable_real_code_generation: bool = False
    
    # Experimental features
    enable_quantum_computing: bool = False
    enable_neuromorphic_computing: bool = False
    enable_swarm_intelligence: bool = False
    
    # Safety features
    enable_human_in_loop: bool = True
    enable_auto_rollback: bool = True
    enable_canary_deployment: bool = True


class HephaestusConfig(BaseModel):
    """Main configuration class with validation."""
    
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True
    )
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: LogLevel = LogLevel.INFO
    
    # Core configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Feature flags
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    # Paths
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    logs_dir: Path = Field(default=Path("./logs"), description="Logs directory")
    models_dir: Path = Field(default=Path("./models"), description="Models directory")
    temp_dir: Path = Field(default=Path("./temp"), description="Temporary directory")
    
    @validator("data_dir", "logs_dir", "models_dir", "temp_dir", pre=True)
    def ensure_path(cls, v):
        """Ensure paths are Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """Validate environment from string or Environment."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.logs_dir, self.models_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set debug mode based on environment
        if self.environment == Environment.DEVELOPMENT:
            object.__setattr__(self, 'debug', True)
            object.__setattr__(self, 'log_level', LogLevel.DEBUG)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level.value,
            "database": asdict(self.database),
            "redis": asdict(self.redis),
            "security": asdict(self.security),
            "learning": asdict(self.learning),
            "monitoring": asdict(self.monitoring),
            "safety": asdict(self.safety),
            "api": asdict(self.api),
            "features": asdict(self.features),
            "data_dir": str(self.data_dir),
            "logs_dir": str(self.logs_dir),
            "models_dir": str(self.models_dir),
            "temp_dir": str(self.temp_dir),
        }
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, indent=2, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "HephaestusConfig":
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HephaestusConfig":
        """Create configuration from dictionary."""
        # Convert nested dicts to dataclass instances
        if "database" in data and isinstance(data["database"], dict):
            data["database"] = DatabaseConfig(**data["database"])
        
        if "redis" in data and isinstance(data["redis"], dict):
            data["redis"] = RedisConfig(**data["redis"])
        
        if "security" in data and isinstance(data["security"], dict):
            data["security"] = SecurityConfig(**data["security"])
        
        if "learning" in data and isinstance(data["learning"], dict):
            data["learning"] = LearningConfig(**data["learning"])
        
        if "monitoring" in data and isinstance(data["monitoring"], dict):
            data["monitoring"] = MonitoringConfig(**data["monitoring"])
        
        if "safety" in data and isinstance(data["safety"], dict):
            data["safety"] = SafetyConfig(**data["safety"])
        
        if "api" in data and isinstance(data["api"], dict):
            data["api"] = APIConfig(**data["api"])
        
        if "features" in data and isinstance(data["features"], dict):
            data["features"] = FeatureFlags(**data["features"])
        
        return cls(**data)
    
    @classmethod
    def from_env(cls, prefix: str = "HEPHAESTUS_") -> "HephaestusConfig":
        """Create configuration from environment variables."""
        env_data = {}
        
        # Map environment variables to config structure
        env_mapping = {
            f"{prefix}ENVIRONMENT": "environment",
            f"{prefix}DEBUG": "debug",
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}DATABASE_URL": "database.url",
            f"{prefix}REDIS_HOST": "redis.host",
            f"{prefix}REDIS_PORT": "redis.port",
            f"{prefix}API_HOST": "api.host",
            f"{prefix}API_PORT": "api.port",
            f"{prefix}SECURITY_LEVEL": "security.security_level",
            f"{prefix}TELEMETRY_ENABLED": "monitoring.telemetry_enabled",
            f"{prefix}ENABLE_REAL_RSI": "features.enable_real_rsi",
        }
        
        for env_var, config_path in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Parse the value appropriately
                if env_value.lower() in ["true", "false"]:
                    env_value = env_value.lower() == "true"
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif env_value.replace(".", "").isdigit():
                    env_value = float(env_value)
                
                # Set nested value
                keys = config_path.split(".")
                current = env_data
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = env_value
        
        return cls.from_dict(env_data)


# Global configuration instance
_config: Optional[HephaestusConfig] = None


def get_config() -> HephaestusConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: HephaestusConfig) -> None:
    """Set global configuration instance."""
    global _config
    _config = config


def load_config(
    config_file: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None
) -> HephaestusConfig:
    """Load configuration with environment-specific overrides."""
    
    # Determine environment
    env = environment or os.getenv("HEPHAESTUS_ENVIRONMENT", "development")
    
    # Start with base configuration
    if config_file:
        config = HephaestusConfig.from_file(config_file)
    else:
        # Look for default config files
        config_dir = Path(__file__).parent
        default_files = [
            config_dir / "config.yml",
            config_dir / "config.yaml", 
            config_dir / "config.json",
            config_dir / f"{env}.yml",
            config_dir / f"{env}.yaml",
            config_dir / f"{env}.json",
        ]
        
        config = None
        for file_path in default_files:
            if file_path.exists():
                config = HephaestusConfig.from_file(file_path)
                break
        
        if config is None:
            config = HephaestusConfig()
    
    # Apply environment variable overrides
    env_config = HephaestusConfig.from_env()
    if env_config:
        # Merge configurations (env variables take precedence)
        config_dict = config.to_dict()
        env_dict = env_config.to_dict()
        
        def deep_merge(base: dict, override: dict) -> dict:
            """Deep merge two dictionaries."""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(config_dict, env_dict)
        config = HephaestusConfig.from_dict(merged_dict)
    
    return config


def reload_config(config_file: Optional[Union[str, Path]] = None) -> HephaestusConfig:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_file)
    return _config