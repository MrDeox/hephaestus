"""
Canary Deployment System - Rollout Gradual com Rollback Automático
Sistema de deploy seguro que testa mudanças em subconjunto antes de rollout completo.
"""

import asyncio
import json
import time
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import git
import tempfile
import shutil

from loguru import logger
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from ..core.state import RSIStateManager
from ..monitoring.audit_logger import audit_system_event
from ..safety.circuits import RSICircuitBreaker


class DeploymentStatus(str, Enum):
    """Status do deployment."""
    PENDING = "pending"
    CANARY_TESTING = "canary_testing"
    EXPANDING = "expanding"
    DEPLOYED = "deployed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    ABORTED = "aborted"


class DeploymentDecision(str, Enum):
    """Decisão de deployment baseada em métricas."""
    CONTINUE = "continue"
    PAUSE = "pause"
    ROLLBACK = "rollback"
    ABORT = "abort"


@dataclass
class DeploymentConfig:
    """Configuração do deployment canário."""
    
    canary_percentage: float = 0.05  # 5% inicial
    expansion_steps: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])
    monitoring_duration: int = 30 * 60  # 30 minutos por step
    rollback_threshold: float = 0.95  # Se performance < 95% da baseline, rollback
    error_rate_threshold: float = 0.05  # Máximo 5% error rate
    latency_threshold_multiplier: float = 1.5  # Máximo 1.5x latência baseline
    required_sample_size: int = 100  # Mínimo de samples para decisão


@dataclass
class PerformanceMetrics:
    """Métricas de performance para comparação."""
    
    accuracy: float = 0.0
    latency_ms: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    sample_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def is_better_than(self, baseline: 'PerformanceMetrics', config: DeploymentConfig) -> bool:
        """Verifica se métricas são melhores que baseline."""
        if self.sample_count < config.required_sample_size:
            return False
            
        # Verificar error rate
        if self.error_rate > config.error_rate_threshold:
            return False
            
        # Verificar performance relativa
        if self.accuracy < baseline.accuracy * config.rollback_threshold:
            return False
            
        # Verificar latência
        if self.latency_ms > baseline.latency_ms * config.latency_threshold_multiplier:
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'accuracy': self.accuracy,
            'latency_ms': self.latency_ms,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'sample_count': self.sample_count,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DeploymentState:
    """Estado atual do deployment."""
    
    deployment_id: str
    status: DeploymentStatus
    current_percentage: float
    start_time: datetime
    last_update: datetime
    baseline_metrics: Optional[PerformanceMetrics] = None
    canary_metrics: Optional[PerformanceMetrics] = None
    commit_hash: Optional[str] = None
    rollback_commit: Optional[str] = None
    error_messages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'deployment_id': self.deployment_id,
            'status': self.status.value,
            'current_percentage': self.current_percentage,
            'start_time': self.start_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'baseline_metrics': self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            'canary_metrics': self.canary_metrics.to_dict() if self.canary_metrics else None,
            'commit_hash': self.commit_hash,
            'rollback_commit': self.rollback_commit,
            'error_messages': self.error_messages
        }


class MetricsCollector:
    """Coletor de métricas para sistema canário."""
    
    def __init__(self):
        self.baseline_data: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.canary_data: List[Tuple[np.ndarray, np.ndarray, float]] = []
        
    async def collect_baseline_metrics(self, duration_seconds: int = 300) -> PerformanceMetrics:
        """Coleta métricas baseline do sistema atual."""
        logger.info(f"Coletando métricas baseline por {duration_seconds}s...")
        
        start_time = time.time()
        accuracies = []
        latencies = []
        
        while time.time() - start_time < duration_seconds:
            # Simular operação do sistema atual
            X, y, latency = await self._simulate_system_operation()
            
            # Fazer predição com sistema atual
            prediction = await self._current_system_predict(X)
            
            # Calcular accuracy
            accuracy = accuracy_score(y, prediction) if len(np.unique(y)) > 1 else 0.8
            accuracies.append(accuracy)
            latencies.append(latency)
            
            # Aguardar próxima iteração
            await asyncio.sleep(1)
        
        metrics = PerformanceMetrics(
            accuracy=np.mean(accuracies),
            latency_ms=np.mean(latencies),
            throughput=len(accuracies) / duration_seconds,
            error_rate=0.02,  # Simular error rate baixo
            memory_usage=50.0,  # MB
            cpu_usage=30.0,  # %
            sample_count=len(accuracies)
        )
        
        logger.info(f"Baseline coletado: accuracy={metrics.accuracy:.3f}, latency={metrics.latency_ms:.1f}ms")
        return metrics
    
    async def collect_canary_metrics(
        self, 
        canary_system, 
        percentage: float, 
        duration_seconds: int = 300
    ) -> PerformanceMetrics:
        """Coleta métricas do sistema canário."""
        logger.info(f"Coletando métricas canário ({percentage*100:.1f}%) por {duration_seconds}s...")
        
        start_time = time.time()
        accuracies = []
        latencies = []
        errors = 0
        
        while time.time() - start_time < duration_seconds:
            # Simular operação
            X, y, _ = await self._simulate_system_operation()
            
            # Decidir se usar canário baseado na percentagem
            use_canary = np.random.random() < percentage
            
            if use_canary:
                try:
                    # Usar sistema canário
                    start_pred = time.time()
                    prediction = await self._canary_system_predict(canary_system, X)
                    latency = (time.time() - start_pred) * 1000
                    
                    # Calcular accuracy
                    accuracy = accuracy_score(y, prediction) if len(np.unique(y)) > 1 else 0.85
                    accuracies.append(accuracy)
                    latencies.append(latency)
                    
                except Exception as e:
                    errors += 1
                    logger.warning(f"Erro no canário: {e}")
            
            await asyncio.sleep(1)
        
        total_requests = int(duration_seconds * percentage)
        
        metrics = PerformanceMetrics(
            accuracy=np.mean(accuracies) if accuracies else 0.0,
            latency_ms=np.mean(latencies) if latencies else 1000.0,
            throughput=len(accuracies) / duration_seconds if accuracies else 0.0,
            error_rate=errors / max(total_requests, 1),
            memory_usage=60.0,  # Simular uso um pouco maior
            cpu_usage=35.0,
            sample_count=len(accuracies)
        )
        
        logger.info(f"Canário coletado: accuracy={metrics.accuracy:.3f}, latency={metrics.latency_ms:.1f}ms, errors={errors}")
        return metrics
    
    async def _simulate_system_operation(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simula operação do sistema (dados + latência)."""
        # Gerar dados sintéticos
        X = np.random.randn(50, 5)
        y = (np.random.randn(50) > 0).astype(int)
        
        # Simular latência baseline
        latency = np.random.normal(100, 20)  # ~100ms
        
        return X, y, latency
    
    async def _current_system_predict(self, X: np.ndarray) -> np.ndarray:
        """Simula predição do sistema atual."""
        # Simular sistema atual com accuracy ~80%
        predictions = np.random.randint(0, 2, len(X))
        return predictions
    
    async def _canary_system_predict(self, canary_system, X: np.ndarray) -> np.ndarray:
        """Faz predição usando sistema canário."""
        try:
            # Usar sistema canário real se disponível
            if hasattr(canary_system, 'predict'):
                return canary_system.predict(X)
            else:
                # Simular sistema melhorado
                predictions = np.random.randint(0, 2, len(X))
                return predictions
        except Exception as e:
            logger.error(f"Erro na predição canário: {e}")
            raise


class GitManager:
    """Gerenciador de operações Git para deployment."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        
    def get_current_commit(self) -> str:
        """Retorna hash do commit atual."""
        return self.repo.head.commit.hexsha
    
    def create_deployment_branch(self, deployment_id: str) -> str:
        """Cria branch para deployment."""
        branch_name = f"deploy-{deployment_id}"
        
        # Criar branch
        self.repo.create_head(branch_name)
        
        logger.info(f"Branch criado: {branch_name}")
        return branch_name
    
    def safe_merge(self, source_branch: str, target_branch: str = "main") -> bool:
        """Executa merge seguro."""
        try:
            # Checkout target branch
            self.repo.git.checkout(target_branch)
            
            # Merge com --no-ff para preservar histórico
            self.repo.git.merge(source_branch, no_ff=True)
            
            logger.info(f"Merge realizado: {source_branch} -> {target_branch}")
            return True
            
        except Exception as e:
            logger.error(f"Erro no merge: {e}")
            return False
    
    def rollback_to_commit(self, commit_hash: str) -> bool:
        """Faz rollback para commit específico."""
        try:
            # Reset hard para commit anterior
            self.repo.git.reset('--hard', commit_hash)
            
            logger.info(f"Rollback realizado para: {commit_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Erro no rollback: {e}")
            return False
    
    def create_rollback_commit(self, message: str) -> str:
        """Cria commit de rollback."""
        try:
            # Commit das mudanças de rollback
            self.repo.git.add('.')
            commit = self.repo.index.commit(message)
            
            logger.info(f"Commit de rollback criado: {commit.hexsha}")
            return commit.hexsha
            
        except Exception as e:
            logger.error(f"Erro criando commit de rollback: {e}")
            return ""


class CanaryDeploymentOrchestrator:
    """Orquestrador principal do sistema de deployment canário."""
    
    def __init__(
        self,
        config: Optional[DeploymentConfig] = None,
        state_manager: Optional[RSIStateManager] = None,
        circuit_breaker: Optional[RSICircuitBreaker] = None
    ):
        self.config = config or DeploymentConfig()
        self.state_manager = state_manager
        self.circuit_breaker = circuit_breaker
        
        self.metrics_collector = MetricsCollector()
        self.git_manager = GitManager()
        
        # Estado atual
        self.current_deployment: Optional[DeploymentState] = None
        self.deployments_history: List[DeploymentState] = []
        
        # Diretório de estado
        self.state_dir = Path("deployment_state")
        self.state_dir.mkdir(exist_ok=True)
    
    async def deploy_artifact(
        self, 
        artifact_path: Path, 
        deployment_id: str
    ) -> bool:
        """Executa deployment canário completo de um artefato."""
        logger.info(f"Iniciando deployment canário: {deployment_id}")
        
        try:
            # 1. Preparar deployment
            deployment_state = DeploymentState(
                deployment_id=deployment_id,
                status=DeploymentStatus.PENDING,
                current_percentage=0.0,
                start_time=datetime.now(timezone.utc),
                last_update=datetime.now(timezone.utc),
                rollback_commit=self.git_manager.get_current_commit()
            )
            
            self.current_deployment = deployment_state
            await self._save_deployment_state()
            
            # 2. Coletar métricas baseline
            deployment_state.status = DeploymentStatus.CANARY_TESTING
            deployment_state.baseline_metrics = await self.metrics_collector.collect_baseline_metrics()
            await self._save_deployment_state()
            
            # 3. Carregar artefato canário
            canary_system = await self._load_canary_artifact(artifact_path)
            if not canary_system:
                await self._abort_deployment("Falha ao carregar artefato")
                return False
            
            # 4. Executar rollout gradual
            success = await self._execute_gradual_rollout(canary_system, deployment_state)
            
            if success:
                deployment_state.status = DeploymentStatus.DEPLOYED
                logger.info(f"Deployment {deployment_id} concluído com sucesso!")
            else:
                deployment_state.status = DeploymentStatus.FAILED
                logger.error(f"Deployment {deployment_id} falhou")
            
            await self._save_deployment_state()
            self.deployments_history.append(deployment_state)
            
            return success
            
        except Exception as e:
            logger.error(f"Erro no deployment: {e}")
            await self._abort_deployment(f"Erro: {e}")
            return False
    
    async def _execute_gradual_rollout(
        self, 
        canary_system, 
        deployment_state: DeploymentState
    ) -> bool:
        """Executa rollout gradual com monitoramento."""
        
        # Iniciar com canário
        percentages = [self.config.canary_percentage] + self.config.expansion_steps
        
        for percentage in percentages:
            logger.info(f"Expandindo para {percentage*100:.1f}% do tráfego...")
            
            deployment_state.current_percentage = percentage
            deployment_state.last_update = datetime.now(timezone.utc)
            await self._save_deployment_state()
            
            # Coletar métricas do canário
            canary_metrics = await self.metrics_collector.collect_canary_metrics(
                canary_system, 
                percentage, 
                self.config.monitoring_duration
            )
            
            deployment_state.canary_metrics = canary_metrics
            await self._save_deployment_state()
            
            # Tomar decisão baseada nas métricas
            decision = self._make_deployment_decision(
                deployment_state.baseline_metrics, 
                canary_metrics
            )
            
            if decision == DeploymentDecision.ROLLBACK:
                logger.warning("Métricas degradadas, executando rollback...")
                await self._execute_rollback(deployment_state)
                return False
                
            elif decision == DeploymentDecision.ABORT:
                logger.error("Problemas críticos detectados, abortando...")
                await self._abort_deployment("Problemas críticos nas métricas")
                return False
                
            elif decision == DeploymentDecision.PAUSE:
                logger.info("Pausando deployment por segurança...")
                await asyncio.sleep(self.config.monitoring_duration)
                continue
            
            # CONTINUE - prosseguir para próximo step
            logger.info(f"Métricas OK, prosseguindo... (accuracy: {canary_metrics.accuracy:.3f})")
            
            # Registrar sucesso do step
            await audit_system_event(
                "canary_deployment", 
                f"Step {percentage*100:.1f}% aprovado",
                metadata={'deployment_id': deployment_state.deployment_id, 'metrics': canary_metrics.to_dict()}
            )
        
        # Se chegou aqui, deployment foi bem-sucedido
        deployment_state.status = DeploymentStatus.DEPLOYED
        await self._finalize_deployment(deployment_state)
        return True
    
    def _make_deployment_decision(
        self, 
        baseline: PerformanceMetrics, 
        canary: PerformanceMetrics
    ) -> DeploymentDecision:
        """Toma decisão sobre o deployment baseado nas métricas."""
        
        # Verificar se há dados suficientes
        if canary.sample_count < self.config.required_sample_size:
            logger.warning(f"Amostras insuficientes: {canary.sample_count} < {self.config.required_sample_size}")
            return DeploymentDecision.PAUSE
        
        # Verificar se canário é melhor que baseline
        if canary.is_better_than(baseline, self.config):
            return DeploymentDecision.CONTINUE
        
        # Verificar se há problemas críticos
        if canary.error_rate > self.config.error_rate_threshold * 2:
            logger.error(f"Error rate crítico: {canary.error_rate:.3f}")
            return DeploymentDecision.ABORT
        
        # Caso contrário, rollback
        logger.warning(f"Performance degradada: accuracy {canary.accuracy:.3f} vs {baseline.accuracy:.3f}")
        return DeploymentDecision.ROLLBACK
    
    async def _execute_rollback(self, deployment_state: DeploymentState):
        """Executa rollback do deployment."""
        logger.info("Executando rollback automático...")
        
        deployment_state.status = DeploymentStatus.ROLLING_BACK
        await self._save_deployment_state()
        
        try:
            # Rollback git
            if deployment_state.rollback_commit:
                success = self.git_manager.rollback_to_commit(deployment_state.rollback_commit)
                if success:
                    # Criar commit de rollback
                    rollback_message = f"Automatic rollback from deployment {deployment_state.deployment_id}"
                    self.git_manager.create_rollback_commit(rollback_message)
                    
                    logger.info("Rollback Git concluído")
                else:
                    deployment_state.error_messages.append("Falha no rollback Git")
            
            # Registrar rollback
            await audit_system_event(
                "canary_deployment", 
                f"Rollback executado para {deployment_state.deployment_id}",
                metadata={'rollback_commit': deployment_state.rollback_commit}
            )
            
        except Exception as e:
            logger.error(f"Erro no rollback: {e}")
            deployment_state.error_messages.append(f"Rollback error: {e}")
    
    async def _abort_deployment(self, reason: str):
        """Aborta deployment por razão crítica."""
        logger.error(f"Abortando deployment: {reason}")
        
        if self.current_deployment:
            self.current_deployment.status = DeploymentStatus.ABORTED
            self.current_deployment.error_messages.append(reason)
            await self._save_deployment_state()
            
            await audit_system_event(
                "canary_deployment", 
                f"Deployment abortado: {reason}",
                metadata={'deployment_id': self.current_deployment.deployment_id}
            )
    
    async def _finalize_deployment(self, deployment_state: DeploymentState):
        """Finaliza deployment bem-sucedido."""
        logger.info("Finalizando deployment...")
        
        try:
            # Criar branch de deployment
            branch_name = self.git_manager.create_deployment_branch(deployment_state.deployment_id)
            
            # Fazer merge final
            if self.git_manager.safe_merge(branch_name):
                deployment_state.commit_hash = self.git_manager.get_current_commit()
                logger.info("Deployment finalizado e commitado")
            else:
                deployment_state.error_messages.append("Falha no merge final")
            
            # Registrar sucesso
            await audit_system_event(
                "canary_deployment", 
                f"Deployment {deployment_state.deployment_id} finalizado",
                metadata={
                    'commit_hash': deployment_state.commit_hash,
                    'final_metrics': deployment_state.canary_metrics.to_dict() if deployment_state.canary_metrics else None
                }
            )
            
        except Exception as e:
            logger.error(f"Erro finalizando deployment: {e}")
            deployment_state.error_messages.append(f"Finalization error: {e}")
    
    async def _load_canary_artifact(self, artifact_path: Path):
        """Carrega artefato para teste canário."""
        try:
            # Em implementação real, carregaria o módulo Python gerado
            # Por ora, simular carregamento
            logger.info(f"Carregando artefato: {artifact_path}")
            
            # Simular sistema canário
            class MockCanarySystem:
                def predict(self, X):
                    # Simular predições melhores que baseline
                    return np.random.randint(0, 2, len(X))
            
            return MockCanarySystem()
            
        except Exception as e:
            logger.error(f"Erro carregando artefato: {e}")
            return None
    
    async def _save_deployment_state(self):
        """Salva estado do deployment."""
        try:
            if self.current_deployment:
                state_file = self.state_dir / f"deployment_{self.current_deployment.deployment_id}.json"
                with open(state_file, 'w') as f:
                    json.dump(self.current_deployment.to_dict(), f, indent=2)
                    
        except Exception as e:
            logger.error(f"Erro salvando estado: {e}")
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Retorna status de um deployment."""
        try:
            state_file = self.state_dir / f"deployment_{deployment_id}.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    return json.load(f)
            return None
            
        except Exception as e:
            logger.error(f"Erro obtendo status: {e}")
            return None
    
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """Lista todos os deployments."""
        deployments = []
        
        try:
            for state_file in self.state_dir.glob("deployment_*.json"):
                with open(state_file, 'r') as f:
                    deployment_data = json.load(f)
                    deployments.append(deployment_data)
            
            # Ordenar por data
            deployments.sort(key=lambda x: x['start_time'], reverse=True)
            return deployments
            
        except Exception as e:
            logger.error(f"Erro listando deployments: {e}")
            return []


# Factory function
def create_canary_deployment_orchestrator(
    config: Optional[DeploymentConfig] = None,
    state_manager: Optional[RSIStateManager] = None,
    circuit_breaker: Optional[RSICircuitBreaker] = None
) -> CanaryDeploymentOrchestrator:
    """Cria orquestrador de deployment canário configurado."""
    return CanaryDeploymentOrchestrator(
        config=config,
        state_manager=state_manager,
        circuit_breaker=circuit_breaker
    )