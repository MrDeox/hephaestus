"""
Gap Scanner - Sistema de Detecção Automática de Lacunas
Implementa detecção inteligente de gaps baseada em telemetria MELT e anomalias.
"""

import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from loguru import logger
try:
    from pyod.models.ecod import ECOD
    from pyod.models.isolation import IForest
    PYOD_AVAILABLE = True
except ImportError:
    logger.warning("PyOD not available, using sklearn alternatives")
    from sklearn.ensemble import IsolationForest as IForest
    ECOD = None
    PYOD_AVAILABLE = False

from ..monitoring.telemetry import TelemetryCollector
from ..monitoring.anomaly_detection import BehavioralMonitor
from ..core.state import RSIStateManager


class GapType(str, Enum):
    """Tipos de gaps detectáveis."""
    PERFORMANCE = "performance"
    FUNCTIONALITY = "functionality"
    SECURITY = "security"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    OBSERVABILITY = "observability"
    KNOWLEDGE = "knowledge"
    CAPABILITY = "capability"


class GapSeverity(str, Enum):
    """Severidade dos gaps detectados."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Gap:
    """Representação de um gap detectado."""
    
    gap_id: str
    gap_type: GapType
    severity: GapSeverity
    title: str
    description: str
    
    # Evidências
    evidence: Dict[str, Any] = field(default_factory=dict)
    metrics_data: Dict[str, float] = field(default_factory=dict)
    
    # Contexto temporal
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    first_observed: Optional[datetime] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Análise de impacto
    impact_score: float = 0.0
    affected_components: List[str] = field(default_factory=list)
    potential_solutions: List[str] = field(default_factory=list)
    
    # Status
    status: str = "open"  # open, investigating, resolved, dismissed
    assigned_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'gap_id': self.gap_id,
            'gap_type': self.gap_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'evidence': self.evidence,
            'metrics_data': self.metrics_data,
            'detected_at': self.detected_at.isoformat(),
            'first_observed': self.first_observed.isoformat() if self.first_observed else None,
            'last_updated': self.last_updated.isoformat(),
            'impact_score': self.impact_score,
            'affected_components': self.affected_components,
            'potential_solutions': self.potential_solutions,
            'status': self.status,
            'assigned_to': self.assigned_to
        }


class GapScanner:
    """
    Sistema de detecção automática de gaps baseado em:
    - Análise de telemetria MELT (Metrics, Events, Logs, Traces)
    - Detecção de anomalias comportamentais
    - Análise de padrões de falha
    - Monitoramento de SLOs/SLIs
    """
    
    def __init__(
        self,
        state_manager: Optional[RSIStateManager] = None,
        telemetry_collector: Optional[TelemetryCollector] = None,
        behavioral_monitor: Optional[BehavioralMonitor] = None
    ):
        self.state_manager = state_manager
        self.telemetry_collector = telemetry_collector
        self.behavioral_monitor = behavioral_monitor
        
        # Detectores de anomalia
        if PYOD_AVAILABLE and ECOD:
            self.performance_detector = ECOD(contamination=0.1)
            self.reliability_detector = IForest(contamination=0.05)
        else:
            # Usar sklearn como fallback
            self.performance_detector = IForest(contamination=0.1, random_state=42)
            self.reliability_detector = IForest(contamination=0.05, random_state=42)
        
        # Histórico de gaps
        self.gaps: List[Gap] = []
        self.gap_patterns: Dict[str, Dict] = {}
        
        # Thresholds adaptativos
        self.adaptive_thresholds = {
            'error_rate': 0.01,      # 1% error rate
            'latency_p99': 1000,     # 1s latency
            'availability': 0.999,   # 99.9% uptime
            'accuracy_drop': 0.05,   # 5% accuracy drop
            'memory_growth': 1.5,    # 50% memory growth
            'cpu_usage': 0.8         # 80% CPU usage
        }
        
        # Janelas de análise
        self.analysis_windows = {
            'realtime': timedelta(minutes=5),
            'short_term': timedelta(hours=1),
            'medium_term': timedelta(hours=24),
            'long_term': timedelta(days=7)
        }
        
        logger.info("Gap Scanner inicializado com detectores de anomalia")
    
    async def scan_for_gaps(self) -> List[Gap]:
        """
        Executa varredura completa para detectar gaps.
        
        Returns:
            Lista de gaps detectados
        """
        logger.info("🔍 Iniciando varredura automática de gaps...")
        
        detected_gaps = []
        
        try:
            # 1. Análise de Performance
            performance_gaps = await self._scan_performance_gaps()
            detected_gaps.extend(performance_gaps)
            
            # 2. Análise de Funcionalidade
            functionality_gaps = await self._scan_functionality_gaps()
            detected_gaps.extend(functionality_gaps)
            
            # 3. Análise de Segurança
            security_gaps = await self._scan_security_gaps()
            detected_gaps.extend(security_gaps)
            
            # 4. Análise de Confiabilidade
            reliability_gaps = await self._scan_reliability_gaps()
            detected_gaps.extend(reliability_gaps)
            
            # 5. Análise de Observabilidade
            observability_gaps = await self._scan_observability_gaps()
            detected_gaps.extend(observability_gaps)
            
            # 6. Análise de Conhecimento/Capacidades
            knowledge_gaps = await self._scan_knowledge_gaps()
            detected_gaps.extend(knowledge_gaps)
            
            # Priorizar e filtrar gaps
            prioritized_gaps = self._prioritize_gaps(detected_gaps)
            
            # Atualizar histórico
            self.gaps.extend(prioritized_gaps)
            
            # Salvar resultados
            await self._save_gap_analysis(prioritized_gaps)
            
            logger.info(f"✅ Varredura concluída: {len(prioritized_gaps)} gaps detectados")
            return prioritized_gaps
            
        except Exception as e:
            logger.error(f"❌ Erro na varredura de gaps: {e}")
            return []
    
    async def _scan_performance_gaps(self) -> List[Gap]:
        """Detecta gaps de performance."""
        gaps = []
        
        try:
            # Analisar métricas de latência
            latency_data = await self._get_latency_metrics()
            if latency_data:
                p99_latency = np.percentile(latency_data, 99)
                if p99_latency > self.adaptive_thresholds['latency_p99']:
                    gap = Gap(
                        gap_id=f"perf_latency_{int(datetime.now().timestamp())}",
                        gap_type=GapType.PERFORMANCE,
                        severity=GapSeverity.HIGH if p99_latency > 2000 else GapSeverity.MEDIUM,
                        title="High Latency Detected",
                        description=f"P99 latency is {p99_latency:.2f}ms, exceeding threshold of {self.adaptive_thresholds['latency_p99']}ms",
                        evidence={
                            'p99_latency': p99_latency,
                            'threshold': self.adaptive_thresholds['latency_p99'],
                            'sample_size': len(latency_data)
                        },
                        metrics_data={
                            'current_p99': p99_latency,
                            'threshold_p99': self.adaptive_thresholds['latency_p99']
                        },
                        impact_score=min(p99_latency / self.adaptive_thresholds['latency_p99'], 5.0),
                        affected_components=['api_server', 'prediction_service'],
                        potential_solutions=[
                            "Implement caching layer",
                            "Optimize database queries",
                            "Add horizontal scaling",
                            "Profile and optimize hot paths"
                        ]
                    )
                    gaps.append(gap)
            
            # Analisar throughput
            throughput_data = await self._get_throughput_metrics()
            if throughput_data:
                current_throughput = np.mean(throughput_data[-100:])  # Últimas 100 medições
                baseline_throughput = np.mean(throughput_data[:-100]) if len(throughput_data) > 100 else current_throughput
                
                if baseline_throughput > 0 and current_throughput < baseline_throughput * 0.8:  # 20% drop
                    gap = Gap(
                        gap_id=f"perf_throughput_{int(datetime.now().timestamp())}",
                        gap_type=GapType.PERFORMANCE,
                        severity=GapSeverity.HIGH,
                        title="Throughput Degradation",
                        description=f"Throughput dropped from {baseline_throughput:.2f} to {current_throughput:.2f} req/s",
                        evidence={
                            'current_throughput': current_throughput,
                            'baseline_throughput': baseline_throughput,
                            'degradation_percentage': (1 - current_throughput/baseline_throughput) * 100
                        },
                        impact_score=3.0,
                        affected_components=['request_processor'],
                        potential_solutions=[
                            "Scale up instances",
                            "Optimize resource allocation",
                            "Check for resource contention"
                        ]
                    )
                    gaps.append(gap)
            
        except Exception as e:
            logger.error(f"Erro na análise de performance: {e}")
        
        return gaps
    
    async def _scan_functionality_gaps(self) -> List[Gap]:
        """Detecta gaps de funcionalidade."""
        gaps = []
        
        try:
            # Analisar accuracy drops
            accuracy_data = await self._get_accuracy_metrics()
            if accuracy_data and len(accuracy_data) > 10:
                recent_accuracy = np.mean(accuracy_data[-10:])
                baseline_accuracy = np.mean(accuracy_data[:-10])
                
                if baseline_accuracy > 0 and recent_accuracy < baseline_accuracy - self.adaptive_thresholds['accuracy_drop']:
                    gap = Gap(
                        gap_id=f"func_accuracy_{int(datetime.now().timestamp())}",
                        gap_type=GapType.FUNCTIONALITY,
                        severity=GapSeverity.CRITICAL,
                        title="Model Accuracy Degradation",
                        description=f"Model accuracy dropped from {baseline_accuracy:.4f} to {recent_accuracy:.4f}",
                        evidence={
                            'current_accuracy': recent_accuracy,
                            'baseline_accuracy': baseline_accuracy,
                            'drop_amount': baseline_accuracy - recent_accuracy
                        },
                        impact_score=5.0,
                        affected_components=['prediction_model', 'learning_system'],
                        potential_solutions=[
                            "Retrain model with recent data",
                            "Check for data drift",
                            "Validate input features",
                            "Implement concept drift detection"
                        ]
                    )
                    gaps.append(gap)
            
            # Analisar features ausentes
            missing_features = await self._detect_missing_features()
            if missing_features:
                gap = Gap(
                    gap_id=f"func_features_{int(datetime.now().timestamp())}",
                    gap_type=GapType.FUNCTIONALITY,
                    severity=GapSeverity.MEDIUM,
                    title="Missing Key Features",
                    description=f"Detected {len(missing_features)} missing features that could improve performance",
                    evidence={'missing_features': missing_features},
                    impact_score=2.0,
                    affected_components=['feature_engineering'],
                    potential_solutions=[
                        "Implement missing features",
                        "Feature importance analysis",
                        "Automated feature generation"
                    ]
                )
                gaps.append(gap)
            
        except Exception as e:
            logger.error(f"Erro na análise de funcionalidade: {e}")
        
        return gaps
    
    async def _scan_security_gaps(self) -> List[Gap]:
        """Detecta gaps de segurança."""
        gaps = []
        
        try:
            # Analisar tentativas de acesso anômalas
            access_patterns = await self._get_access_patterns()
            if access_patterns:
                # Detectar padrões anômalos usando IForest
                if len(access_patterns) > 50:
                    features = np.array([[p['requests_per_minute'], p['unique_endpoints'], p['error_rate']] 
                                       for p in access_patterns[-100:]])
                    
                    if features.shape[0] > 10:
                        self.reliability_detector.fit(features[:-10])
                        recent_scores = self.reliability_detector.decision_function(features[-10:])
                        
                        if np.any(recent_scores < -0.5):  # Anomalias detectadas
                            gap = Gap(
                                gap_id=f"sec_access_{int(datetime.now().timestamp())}",
                                gap_type=GapType.SECURITY,
                                severity=GapSeverity.HIGH,
                                title="Anomalous Access Patterns",
                                description="Detected unusual access patterns that may indicate security threats",
                                evidence={
                                    'anomaly_scores': recent_scores.tolist(),
                                    'patterns_analyzed': len(features)
                                },
                                impact_score=4.0,
                                affected_components=['api_gateway', 'authentication'],
                                potential_solutions=[
                                    "Implement rate limiting",
                                    "Enhanced authentication",
                                    "IP-based blocking",
                                    "Security audit"
                                ]
                            )
                            gaps.append(gap)
            
        except Exception as e:
            logger.error(f"Erro na análise de segurança: {e}")
        
        return gaps
    
    async def _scan_reliability_gaps(self) -> List[Gap]:
        """Detecta gaps de confiabilidade."""
        gaps = []
        
        try:
            # Analisar error rates
            error_data = await self._get_error_rates()
            if error_data:
                recent_error_rate = np.mean(error_data[-20:]) if len(error_data) >= 20 else np.mean(error_data)
                
                if recent_error_rate > self.adaptive_thresholds['error_rate']:
                    gap = Gap(
                        gap_id=f"rel_errors_{int(datetime.now().timestamp())}",
                        gap_type=GapType.RELIABILITY,
                        severity=GapSeverity.HIGH if recent_error_rate > 0.05 else GapSeverity.MEDIUM,
                        title="High Error Rate",
                        description=f"Error rate is {recent_error_rate:.4f}, exceeding threshold of {self.adaptive_thresholds['error_rate']:.4f}",
                        evidence={
                            'current_error_rate': recent_error_rate,
                            'threshold': self.adaptive_thresholds['error_rate']
                        },
                        impact_score=min(recent_error_rate / self.adaptive_thresholds['error_rate'], 5.0),
                        affected_components=['error_handling', 'circuit_breakers'],
                        potential_solutions=[
                            "Implement circuit breakers",
                            "Improve error handling",
                            "Add retry mechanisms",
                            "Monitor downstream dependencies"
                        ]
                    )
                    gaps.append(gap)
            
        except Exception as e:
            logger.error(f"Erro na análise de confiabilidade: {e}")
        
        return gaps
    
    async def _scan_observability_gaps(self) -> List[Gap]:
        """Detecta gaps de observabilidade."""
        gaps = []
        
        try:
            # Analisar cobertura de métricas
            metrics_coverage = await self._assess_metrics_coverage()
            
            if metrics_coverage < 0.8:  # Menos de 80% de cobertura
                gap = Gap(
                    gap_id=f"obs_metrics_{int(datetime.now().timestamp())}",
                    gap_type=GapType.OBSERVABILITY,
                    severity=GapSeverity.MEDIUM,
                    title="Insufficient Metrics Coverage",
                    description=f"Only {metrics_coverage:.1%} of system components have adequate metrics",
                    evidence={'coverage_percentage': metrics_coverage},
                    impact_score=2.0,
                    affected_components=['monitoring', 'telemetry'],
                    potential_solutions=[
                        "Add missing metrics",
                        "Implement distributed tracing",
                        "Enhance logging coverage",
                        "Setup alerting rules"
                    ]
                )
                gaps.append(gap)
            
        except Exception as e:
            logger.error(f"Erro na análise de observabilidade: {e}")
        
        return gaps
    
    async def _scan_knowledge_gaps(self) -> List[Gap]:
        """Detecta gaps de conhecimento e capacidades."""
        gaps = []
        
        try:
            # Analisar diferença entre RSI simulado vs real
            if self.state_manager:
                try:
                    # Ler estado atual
                    with open("rsi_continuous_state.json", 'r') as f:
                        simulated_state = json.load(f)
                    
                    metrics = simulated_state.get('metrics', {})
                    simulated_skills = metrics.get('total_skills_learned', 0)
                    successful_expansions = metrics.get('successful_expansions', 0)
                    
                    # Detectar gap crítico: muitas skills simuladas, zero reais
                    if simulated_skills > 1000 and successful_expansions == 0:
                        gap = Gap(
                            gap_id=f"know_simulation_{int(datetime.now().timestamp())}",
                            gap_type=GapType.KNOWLEDGE,
                            severity=GapSeverity.CRITICAL,
                            title="RSI Simulation vs Reality Gap",
                            description=f"System reports {simulated_skills:,} learned skills but 0 successful real expansions",
                            evidence={
                                'simulated_skills': simulated_skills,
                                'real_expansions': successful_expansions,
                                'simulation_ratio': simulated_skills / max(successful_expansions, 1)
                            },
                            impact_score=5.0,
                            affected_components=['rsi_system', 'skill_learning', 'continuous_improvement'],
                            potential_solutions=[
                                "Replace simulation with real RSI execution",
                                "Implement actual code generation pipeline",
                                "Add real skill compilation and testing",
                                "Bridge hypothesis generation to real improvements"
                            ]
                        )
                        gaps.append(gap)
                        
                except Exception as e:
                    logger.warning(f"Não foi possível analisar estado RSI: {e}")
            
        except Exception as e:
            logger.error(f"Erro na análise de conhecimento: {e}")
        
        return gaps
    
    def _prioritize_gaps(self, gaps: List[Gap]) -> List[Gap]:
        """Prioriza gaps por severidade e impacto."""
        
        # Definir pesos por severidade
        severity_weights = {
            GapSeverity.CRITICAL: 5.0,
            GapSeverity.HIGH: 4.0,
            GapSeverity.MEDIUM: 3.0,
            GapSeverity.LOW: 2.0,
            GapSeverity.INFO: 1.0
        }
        
        # Calcular score de prioridade
        for gap in gaps:
            severity_weight = severity_weights.get(gap.severity, 1.0)
            gap.impact_score = gap.impact_score * severity_weight
        
        # Ordenar por impacto (maior primeiro)
        prioritized = sorted(gaps, key=lambda g: g.impact_score, reverse=True)
        
        # Limitar a top 10 gaps para evitar overload
        return prioritized[:10]
    
    async def _save_gap_analysis(self, gaps: List[Gap]):
        """Salva análise de gaps."""
        try:
            gaps_dir = Path("gap_analysis")
            gaps_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = gaps_dir / f"gaps_{timestamp}.json"
            
            analysis_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_gaps': len(gaps),
                'gaps_by_type': {gap_type.value: len([g for g in gaps if g.gap_type == gap_type]) 
                               for gap_type in GapType},
                'gaps_by_severity': {severity.value: len([g for g in gaps if g.severity == severity]) 
                                   for severity in GapSeverity},
                'gaps': [gap.to_dict() for gap in gaps]
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"📋 Análise de gaps salva: {analysis_file}")
            
        except Exception as e:
            logger.error(f"Erro salvando análise: {e}")
    
    # Métodos auxiliares para coleta de métricas
    async def _get_latency_metrics(self) -> Optional[List[float]]:
        """Coleta métricas de latência."""
        try:
            # Simular coleta de métricas - em produção, viria do telemetry collector
            return np.random.lognormal(6.5, 0.5, 100).tolist()  # ~1000ms média
        except:
            return None
    
    async def _get_throughput_metrics(self) -> Optional[List[float]]:
        """Coleta métricas de throughput."""
        try:
            return np.random.normal(100, 20, 200).tolist()  # ~100 req/s
        except:
            return None
    
    async def _get_accuracy_metrics(self) -> Optional[List[float]]:
        """Coleta métricas de accuracy."""
        try:
            # Simular degradação gradual
            baseline = np.random.normal(0.85, 0.02, 50)
            recent = np.random.normal(0.82, 0.03, 20)  # Accuracy drop
            return np.concatenate([baseline, recent]).tolist()
        except:
            return None
    
    async def _get_error_rates(self) -> Optional[List[float]]:
        """Coleta taxas de erro."""
        try:
            return np.random.exponential(0.005, 100).tolist()  # ~0.5% error rate
        except:
            return None
    
    async def _get_access_patterns(self) -> Optional[List[Dict]]:
        """Coleta padrões de acesso."""
        try:
            patterns = []
            for i in range(100):
                patterns.append({
                    'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                    'requests_per_minute': np.random.poisson(50),
                    'unique_endpoints': np.random.randint(5, 20),
                    'error_rate': np.random.exponential(0.01)
                })
            return patterns
        except:
            return None
    
    async def _detect_missing_features(self) -> List[str]:
        """Detecta features ausentes."""
        potential_features = [
            "user_engagement_score",
            "temporal_features",
            "interaction_history",
            "contextual_embeddings",
            "behavior_patterns"
        ]
        
        # Simular detecção de features ausentes
        return np.random.choice(potential_features, size=np.random.randint(0, 3), replace=False).tolist()
    
    async def _assess_metrics_coverage(self) -> float:
        """Avalia cobertura de métricas."""
        # Simular assessment de cobertura
        return np.random.uniform(0.6, 0.9)


# Factory function
def create_gap_scanner(
    state_manager: Optional[RSIStateManager] = None,
    telemetry_collector: Optional[TelemetryCollector] = None,
    behavioral_monitor: Optional[BehavioralMonitor] = None
) -> GapScanner:
    """Cria um gap scanner configurado."""
    return GapScanner(
        state_manager=state_manager,
        telemetry_collector=telemetry_collector,
        behavioral_monitor=behavioral_monitor
    )