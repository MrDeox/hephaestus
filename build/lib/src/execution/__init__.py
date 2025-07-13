"""
Execution Module - Real RSI Implementation
Sistema completo de execução real de melhorias RSI.
"""

from .real_code_generator import (
    RealCodeGenerator,
    HypothesisSpec,
    CodeArtifact,
    SecureVirtualEnv,
    ProcessIsolator,
    CodeGenerationStatus,
    create_real_code_generator
)

from .canary_deployment import (
    CanaryDeploymentOrchestrator,
    DeploymentConfig,
    PerformanceMetrics,
    DeploymentState,
    DeploymentStatus,
    DeploymentDecision,
    MetricsCollector,
    GitManager,
    create_canary_deployment_orchestrator
)

from .rsi_execution_pipeline import (
    RSIExecutionPipeline,
    RSIExecutionResult,
    PipelineStatus,
    create_rsi_execution_pipeline
)

__all__ = [
    # Real Code Generator
    'RealCodeGenerator',
    'HypothesisSpec', 
    'CodeArtifact',
    'SecureVirtualEnv',
    'ProcessIsolator',
    'CodeGenerationStatus',
    'create_real_code_generator',
    
    # Canary Deployment
    'CanaryDeploymentOrchestrator',
    'DeploymentConfig',
    'PerformanceMetrics',
    'DeploymentState', 
    'DeploymentStatus',
    'DeploymentDecision',
    'MetricsCollector',
    'GitManager',
    'create_canary_deployment_orchestrator',
    
    # RSI Pipeline
    'RSIExecutionPipeline',
    'RSIExecutionResult',
    'PipelineStatus',
    'create_rsi_execution_pipeline'
]