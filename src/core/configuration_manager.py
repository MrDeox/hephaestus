"""
Configuration Manager - Extracted from main.py by architecture evolution system.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class SystemConfiguration:
    """System configuration settings."""
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"

class ConfigurationManager:
    """Manages all system configuration."""
    
    def __init__(self):
        self.config = self._load_configuration()
    
    def _load_configuration(self) -> SystemConfiguration:
        """Load configuration from environment variables."""
        return SystemConfiguration(
            environment=os.getenv("ENVIRONMENT", "production"),
            debug_mode=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return getattr(self.config, key, default)
