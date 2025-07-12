#!/usr/bin/env python3
"""
Real RSI Runner - Sistema RSI Real Integrado
Substitui a simulação por implementação real de Recursive Self-Improvement
com Gap Scanner, MML Controller e Real Code Generation.
"""

import asyncio
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import json
from pathlib import Path

from loguru import logger

# Sistemas RSI reais
from src.meta_learning import create_gap_scanner, create_mml_controller
from src.execution import create_rsi_execution_pipeline
from src.core.state import RSIStateManager, RSIState
from src.validation.validators import RSIValidator
from src.safety.circuits import CircuitBreakerManager


class RealRSIRunner:
    """
    Sistema de Execução RSI Real - Substitui simulação por código real.
    
    Arquitetura Real:
    1. Gap Scanner: Detecta lacunas automaticamente
    2. MML Controller: Meta-aprendizado com CEV
    3. Real Code Generator: Gera código Python executável
    4. Canary Deployment: Deploy seguro com rollback
    5. Feedback Loops: Aprendizado recursivo real
    """
    
    def __init__(self):
        # Estado do sistema
        self.running = False
        self.start_time = None
        self.cycle_count = 0
        self.real_improvements_count = 0
        self.failed_improvements_count = 0
        
        # Componentes RSI reais
        self.state_manager = None
        self.validator = None
        self.circuit_manager = None
        self.gap_scanner = None
        self.mml_controller = None
        self.execution_pipeline = None
        
        # Configurações
        self.cycle_interval = 300  # 5 minutos entre ciclos
        self.gap_scan_interval = 600  # 10 minutos para gap scanning
        self.meta_learning_interval = 1800  # 30 minutos para meta-learning
        
        # Estado persistente
        self.state_file = Path("real_rsi_state.json")
        self.state_data = {
            'real_cycles': 0,
            'real_improvements': 0,
            'failed_improvements': 0,
            'gaps_detected': 0,
            'patterns_learned': 0,
            'decisions_made': 0,
            'last_improvement': None,
            'uptime_seconds': 0.0,
            'avg_cycle_duration': 0.0,
            'success_rate': 0.0
        }
        
        # Histórico detalhado
        self.improvement_history = []
        self.gap_history = []
        self.decision_history = []
        
        # Setup signal handlers para shutdown gracioso
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Real RSI Runner inicializado - substituindo simulação por RSI real")
    
    async def initialize(self):
        """Inicializa todos os componentes RSI reais."""
        try:
            logger.info("🔧 Inicializando componentes RSI reais...")
            
            # Core components
            self.state_manager = RSIStateManager(initial_state=RSIState())
            self.validator = RSIValidator()
            self.circuit_manager = CircuitBreakerManager()
            
            # Real RSI components
            self.gap_scanner = create_gap_scanner(
                state_manager=self.state_manager
            )
            
            self.execution_pipeline = create_rsi_execution_pipeline(
                state_manager=self.state_manager,
                validator=self.validator,
                circuit_breaker=self.circuit_manager
            )
            
            self.mml_controller = create_mml_controller(
                gap_scanner=self.gap_scanner,
                execution_pipeline=self.execution_pipeline,
                state_manager=self.state_manager,
                validator=self.validator
            )
            
            # Carregar estado anterior se existir
            await self._load_state()
            
            logger.info("✅ Componentes RSI reais inicializados com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return False
    
    async def run_continuous(self):
        """Executa o sistema RSI real continuamente."""
        if not await self.initialize():
            logger.error("❌ Falha na inicialização - abortando")
            return
        
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        logger.info("🚀 Iniciando sistema RSI real - substituindo simulação!")
        logger.info(f"   Intervalo de ciclos: {self.cycle_interval}s")
        logger.info(f"   Gap scanning: {self.gap_scan_interval}s")
        logger.info(f"   Meta-learning: {self.meta_learning_interval}s")
        
        last_gap_scan = 0
        last_meta_learning = 0
        
        try:
            while self.running:
                cycle_start = time.time()
                self.cycle_count += 1
                self.state_data['real_cycles'] = self.cycle_count
                
                logger.info(f"🔄 Ciclo RSI real #{self.cycle_count}")
                
                try:
                    # 1. Gap Scanning (periódico)
                    if time.time() - last_gap_scan >= self.gap_scan_interval:
                        await self._run_gap_scanning()
                        last_gap_scan = time.time()
                    
                    # 2. Meta-Learning (periódico)
                    if time.time() - last_meta_learning >= self.meta_learning_interval:
                        await self._run_meta_learning()
                        last_meta_learning = time.time()
                    
                    # 3. RSI Execution (sempre)
                    await self._run_rsi_execution()
                    
                    # 4. Atualizar estado
                    await self._update_state(cycle_start)
                    
                    # 5. Salvar progresso
                    await self._save_state()
                    
                    logger.info(f"✅ Ciclo #{self.cycle_count} concluído - "
                              f"melhorias reais: {self.real_improvements_count}")
                    
                except Exception as e:
                    logger.error(f"❌ Erro no ciclo #{self.cycle_count}: {e}")
                    self.failed_improvements_count += 1
                    self.state_data['failed_improvements'] = self.failed_improvements_count
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.cycle_interval)
                
        except Exception as e:
            logger.error(f"❌ Erro fatal no sistema RSI: {e}")
        
        finally:
            await self._shutdown()
    
    async def _run_gap_scanning(self):
        """Executa detecção de gaps."""
        logger.info("🔍 Executando Gap Scanning...")
        
        try:
            gaps = await self.gap_scanner.scan_for_gaps()
            self.state_data['gaps_detected'] += len(gaps)
            
            # Salvar gaps no histórico
            gap_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'gaps_found': len(gaps),
                'gaps': [gap.to_dict() for gap in gaps[:5]]  # Top 5
            }
            self.gap_history.append(gap_entry)
            
            # Manter apenas últimos 100 entries
            if len(self.gap_history) > 100:
                self.gap_history = self.gap_history[-100:]
            
            logger.info(f"🔍 Gap scanning concluído: {len(gaps)} gaps detectados")
            
            # Se gaps críticos foram encontrados, priorizar sua resolução
            critical_gaps = [g for g in gaps if g.severity.value == 'critical']
            if critical_gaps:
                logger.warning(f"🚨 {len(critical_gaps)} gaps críticos detectados")
                # Aqui poderíamos priorizar a geração de hipóteses para resolver gaps
            
        except Exception as e:
            logger.error(f"❌ Erro no gap scanning: {e}")
    
    async def _run_meta_learning(self):
        """Executa ciclo de meta-aprendizado."""
        logger.info("🧠 Executando Meta-Learning...")
        
        try:
            results = await self.mml_controller.execute_meta_learning_cycle()
            
            if results.get('status') == 'completed':
                patterns = len(results.get('patterns_discovered', []))
                decisions = len(results.get('decisions_made', []))
                
                self.state_data['patterns_learned'] += patterns
                self.state_data['decisions_made'] += decisions
                
                # Salvar no histórico
                decision_entry = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'cycle_results': results,
                    'patterns_discovered': patterns,
                    'decisions_made': decisions
                }
                self.decision_history.append(decision_entry)
                
                # Manter apenas últimos 50 entries
                if len(self.decision_history) > 50:
                    self.decision_history = self.decision_history[-50:]
                
                logger.info(f"🧠 Meta-learning concluído: {patterns} padrões, {decisions} decisões")
            else:
                logger.warning(f"⚠️ Meta-learning falhou: {results.get('error', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"❌ Erro no meta-learning: {e}")
    
    async def _run_rsi_execution(self):
        """Executa uma iteração de RSI real."""
        logger.info("⚙️ Executando RSI real...")
        
        try:
            # Gerar hipótese simples para teste
            hypothesis = {
                'id': f'real_rsi_{self.cycle_count}_{int(time.time())}',
                'name': f'Real Improvement Cycle {self.cycle_count}',
                'description': 'Real RSI improvement generated automatically',
                'type': 'optimization',
                'priority': 'medium',
                'improvement_targets': {
                    'accuracy': 0.02,  # 2% improvement
                    'efficiency': 0.05  # 5% efficiency gain
                },
                'constraints': {
                    'max_complexity': 0.7,
                    'safety_level': 'high',
                    'timeout_seconds': 300
                },
                'context': {
                    'source': 'real_rsi_runner',
                    'cycle': self.cycle_count,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Executar através do pipeline real
            result = await self.execution_pipeline.execute_hypothesis(hypothesis)
            
            # Processar resultado
            if result.success:
                self.real_improvements_count += 1
                self.state_data['real_improvements'] = self.real_improvements_count
                self.state_data['last_improvement'] = datetime.now(timezone.utc).isoformat()
                
                logger.info(f"✅ Melhoria real aplicada! Total: {self.real_improvements_count}")
                
                # Salvar no histórico
                improvement_entry = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'pipeline_id': result.pipeline_id,
                    'hypothesis_id': result.hypothesis_id,
                    'performance_improvement': result.performance_improvement,
                    'duration_seconds': result.duration_seconds
                }
                self.improvement_history.append(improvement_entry)
                
                # Manter apenas últimos 100 melhorias
                if len(self.improvement_history) > 100:
                    self.improvement_history = self.improvement_history[-100:]
                    
            else:
                self.failed_improvements_count += 1
                self.state_data['failed_improvements'] = self.failed_improvements_count
                logger.warning(f"⚠️ Melhoria falhou: {result.error_messages}")
            
        except Exception as e:
            logger.error(f"❌ Erro na execução RSI: {e}")
            self.failed_improvements_count += 1
            self.state_data['failed_improvements'] = self.failed_improvements_count
    
    async def _update_state(self, cycle_start: float):
        """Atualiza estado do sistema."""
        cycle_duration = time.time() - cycle_start
        total_runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        # Atualizar métricas
        self.state_data['uptime_seconds'] = total_runtime
        
        # Calcular duração média de ciclo
        if self.cycle_count > 0:
            self.state_data['avg_cycle_duration'] = total_runtime / self.cycle_count
        
        # Calcular taxa de sucesso
        total_attempts = self.real_improvements_count + self.failed_improvements_count
        if total_attempts > 0:
            self.state_data['success_rate'] = self.real_improvements_count / total_attempts
        
        # Adicionar timestamp
        self.state_data['timestamp'] = datetime.now(timezone.utc).isoformat()
    
    async def _save_state(self):
        """Salva estado atual."""
        try:
            # Estado principal
            with open(self.state_file, 'w') as f:
                json.dump(self.state_data, f, indent=2)
            
            # Históricos detalhados
            history_dir = Path("real_rsi_history")
            history_dir.mkdir(exist_ok=True)
            
            # Salvar históricos se houver dados
            if self.improvement_history:
                with open(history_dir / "improvements.json", 'w') as f:
                    json.dump(self.improvement_history, f, indent=2)
            
            if self.gap_history:
                with open(history_dir / "gaps.json", 'w') as f:
                    json.dump(self.gap_history, f, indent=2)
            
            if self.decision_history:
                with open(history_dir / "decisions.json", 'w') as f:
                    json.dump(self.decision_history, f, indent=2)
                    
        except Exception as e:
            logger.error(f"❌ Erro salvando estado: {e}")
    
    async def _load_state(self):
        """Carrega estado anterior se existir."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    loaded_state = json.load(f)
                
                # Restaurar contadores
                self.cycle_count = loaded_state.get('real_cycles', 0)
                self.real_improvements_count = loaded_state.get('real_improvements', 0)
                self.failed_improvements_count = loaded_state.get('failed_improvements', 0)
                
                # Atualizar state_data
                self.state_data.update(loaded_state)
                
                logger.info(f"📂 Estado anterior carregado: {self.real_improvements_count} melhorias reais")
            
            # Carregar históricos
            history_dir = Path("real_rsi_history")
            if history_dir.exists():
                for hist_file, hist_list in [
                    ("improvements.json", self.improvement_history),
                    ("gaps.json", self.gap_history),
                    ("decisions.json", self.decision_history)
                ]:
                    hist_path = history_dir / hist_file
                    if hist_path.exists():
                        with open(hist_path, 'r') as f:
                            hist_list.extend(json.load(f))
                            
        except Exception as e:
            logger.warning(f"⚠️ Erro carregando estado anterior: {e}")
    
    async def _shutdown(self):
        """Shutdown gracioso do sistema."""
        logger.info("🛑 Iniciando shutdown do sistema RSI real...")
        
        try:
            # Salvar estado final
            await self._save_state()
            
            # Cleanup dos componentes
            if self.circuit_manager:
                # Fechar circuit breakers graciosamente
                pass
            
            logger.info("✅ Shutdown concluído - estado salvo")
            
        except Exception as e:
            logger.error(f"❌ Erro no shutdown: {e}")
        
        finally:
            self.running = False
    
    def _signal_handler(self, signum, frame):
        """Handler para sinais de sistema."""
        logger.info(f"📡 Sinal recebido: {signum} - iniciando shutdown gracioso...")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do sistema."""
        if not self.start_time:
            return {"status": "not_started"}
        
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return {
            "status": "running" if self.running else "stopped",
            "cycle_count": self.cycle_count,
            "real_improvements": self.real_improvements_count,
            "failed_improvements": self.failed_improvements_count,
            "success_rate": self.state_data.get('success_rate', 0.0),
            "uptime_seconds": uptime,
            "gaps_detected": self.state_data.get('gaps_detected', 0),
            "patterns_learned": self.state_data.get('patterns_learned', 0),
            "decisions_made": self.state_data.get('decisions_made', 0),
            "last_improvement": self.state_data.get('last_improvement'),
            "avg_cycle_duration": self.state_data.get('avg_cycle_duration', 0.0)
        }


async def main():
    """Função principal."""
    logger.info("🎯 Real RSI Runner - Substituindo simulação por RSI real!")
    
    runner = RealRSIRunner()
    
    try:
        await runner.run_continuous()
    except KeyboardInterrupt:
        logger.info("👋 Interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
    finally:
        logger.info("🏁 Real RSI Runner finalizado")


if __name__ == "__main__":
    asyncio.run(main())