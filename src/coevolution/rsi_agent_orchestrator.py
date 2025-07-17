"""
RSI-Agent Co-Evolution Orchestrator.

Coordena a co-evolução entre o sistema RSI e o Tool Agent:
- RSI gera hipóteses inteligentes
- Agent executa usando ferramentas reais  
- Ambos aprendem e evoluem juntos
- Feedback bidirecional melhora ambos os sistemas

Arquitetura: RSI ← → Agent ← → Real World
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import uuid

from loguru import logger

from ..agents.rsi_tool_agent import RSIToolAgent, create_rsi_tool_agent
from ..hypothesis.rsi_hypothesis_orchestrator import RSIHypothesisOrchestrator
from ..objectives.revenue_generation import AutonomousRevenueGenerator

@dataclass
class CoEvolutionCycle:
    """Ciclo completo de co-evolução RSI-Agent"""
    cycle_id: str
    rsi_hypothesis: Dict[str, Any]
    agent_execution: Dict[str, Any]
    rsi_learning: Dict[str, Any]
    agent_learning: Dict[str, Any]
    combined_improvement: Dict[str, Any]
    success_score: float
    timestamp: datetime

class RSIAgentOrchestrator:
    """
    Orquestrador da co-evolução RSI-Agent.
    
    Responsabilidades:
    1. Coordenar ciclos de co-evolução
    2. Facilitar feedback bidirecional
    3. Otimizar sinergia RSI-Agent
    4. Monitorar evolução conjunta
    5. Gerar insights de melhoria
    """
    
    def __init__(self, 
                 rsi_orchestrator: Optional[RSIHypothesisOrchestrator] = None,
                 revenue_generator: Optional[AutonomousRevenueGenerator] = None,
                 base_url: str = "http://localhost:8000"):
        
        self.rsi_orchestrator = rsi_orchestrator
        self.revenue_generator = revenue_generator
        self.base_url = base_url
        
        # Criar agente de ferramentas
        self.tool_agent = create_rsi_tool_agent(base_url)
        
        # Estado de co-evolução
        self.coevolution_cycles: List[CoEvolutionCycle] = []
        self.rsi_improvements: List[Dict[str, Any]] = []
        self.agent_improvements: List[Dict[str, Any]] = []
        
        # Métricas de co-evolução
        self.combined_performance_history: List[float] = []
        self.synergy_scores: List[float] = []
        
        # Configuração
        self.max_cycles_per_session = 5
        self.learning_rate = 0.1
        
        # Diretórios
        self.data_dir = Path("coevolution")
        self.data_dir.mkdir(exist_ok=True)
        
        logger.info("🔄 RSI-Agent Co-Evolution Orchestrator inicializado")
    
    async def start_coevolution_cycle(self, initial_targets: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Inicia ciclo completo de co-evolução RSI-Agent.
        
        Fluxo:
        1. RSI gera hipóteses inteligentes
        2. Agent executa usando ferramentas reais
        3. Ambos analisam resultados e aprendem
        4. Próximo ciclo usa aprendizado acumulado
        """
        
        logger.info("🚀 Iniciando ciclo de co-evolução RSI-Agent...")
        
        # Configurar targets iniciais se não fornecidos
        if initial_targets is None:
            initial_targets = {
                "revenue_improvement": 0.15,      # 15% melhoria na receita
                "execution_efficiency": 0.80,    # 80% eficiência de execução
                "learning_acceleration": 0.20    # 20% melhoria no aprendizado
            }
        
        coevolution_results = {
            "session_id": f"coevo_{uuid.uuid4().hex[:8]}",
            "cycles_completed": 0,
            "total_improvements": 0,
            "rsi_evolution": {},
            "agent_evolution": {},
            "combined_achievements": {},
            "next_cycle_suggestions": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Executar múltiplos ciclos de co-evolução
        for cycle_num in range(1, self.max_cycles_per_session + 1):
            
            logger.info(f"🔄 Ciclo de Co-Evolução #{cycle_num}")
            
            try:
                # Executar um ciclo completo
                cycle_result = await self._execute_single_coevolution_cycle(
                    cycle_num, initial_targets, coevolution_results
                )
                
                if cycle_result["success"]:
                    coevolution_results["cycles_completed"] += 1
                    coevolution_results["total_improvements"] += cycle_result["improvements_count"]
                    
                    # Atualizar targets baseado no aprendizado
                    initial_targets = self._evolve_targets(initial_targets, cycle_result)
                    
                    logger.info(f"✅ Ciclo #{cycle_num} concluído: {cycle_result['success_score']:.2f}")
                else:
                    logger.warning(f"⚠️ Ciclo #{cycle_num} teve problemas: {cycle_result.get('error')}")
                
                # Aguardar entre ciclos para permitir consolidação
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Erro no ciclo #{cycle_num}: {e}")
                continue
        
        # Análise final da sessão de co-evolução
        final_analysis = await self._analyze_coevolution_session(coevolution_results)
        coevolution_results.update(final_analysis)
        
        # Salvar resultados
        await self._save_coevolution_session(coevolution_results)
        
        logger.info(f"🎉 Sessão de co-evolução concluída: {coevolution_results['cycles_completed']} ciclos")
        return coevolution_results
    
    async def _execute_single_coevolution_cycle(self, 
                                               cycle_num: int, 
                                               targets: Dict[str, float],
                                               session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Executa um único ciclo de co-evolução"""
        
        cycle_id = f"cycle_{session_context['session_id']}_{cycle_num}"
        
        # Fase 1: RSI gera hipóteses baseado nos targets e contexto
        logger.info("🧠 Fase 1: RSI gerando hipóteses...")
        rsi_hypotheses = await self._rsi_generate_hypotheses(targets, session_context)
        
        # Fase 2: Agent executa hipóteses usando ferramentas reais
        logger.info("🔧 Fase 2: Agent executando com ferramentas reais...")
        agent_results = await self._agent_execute_hypotheses(rsi_hypotheses)
        
        # Fase 3: RSI aprende dos resultados reais do Agent
        logger.info("📚 Fase 3: RSI aprendendo dos resultados reais...")
        rsi_learning = await self._rsi_learn_from_agent(rsi_hypotheses, agent_results)
        
        # Fase 4: Agent aprende dos padrões do RSI
        logger.info("🎯 Fase 4: Agent aprendendo dos padrões RSI...")
        agent_learning = await self._agent_learn_from_rsi(rsi_hypotheses, agent_results)
        
        # Fase 5: Síntese - combinação dos aprendizados
        logger.info("⚡ Fase 5: Sintetizando aprendizados...")
        combined_improvement = await self._synthesize_learnings(rsi_learning, agent_learning)
        
        # Calcular score de sucesso do ciclo
        success_score = self._calculate_cycle_success_score(agent_results, combined_improvement)
        
        # Criar registro do ciclo
        cycle = CoEvolutionCycle(
            cycle_id=cycle_id,
            rsi_hypothesis=rsi_hypotheses,
            agent_execution=agent_results,
            rsi_learning=rsi_learning,
            agent_learning=agent_learning,
            combined_improvement=combined_improvement,
            success_score=success_score,
            timestamp=datetime.utcnow()
        )
        
        self.coevolution_cycles.append(cycle)
        
        return {
            "success": success_score > 0.6,  # 60% threshold for success
            "success_score": success_score,
            "improvements_count": len(combined_improvement.get("improvements", [])),
            "cycle_data": cycle,
            "error": None if success_score > 0.6 else "Below success threshold"
        }
    
    async def _rsi_generate_hypotheses(self, targets: Dict[str, float], context: Dict[str, Any]) -> Dict[str, Any]:
        """RSI gera hipóteses inteligentes baseado em targets e contexto"""
        
        # Usar RSI Hypothesis Orchestrator se disponível
        if self.rsi_orchestrator:
            try:
                # Adaptar targets para formato esperado pelo RSI
                rsi_targets = {
                    "accuracy": targets.get("execution_efficiency", 0.8),
                    "efficiency": targets.get("learning_acceleration", 0.2)
                }
                
                # Gerar hipóteses usando RSI
                results = await self.rsi_orchestrator.orchestrate_hypothesis_lifecycle(rsi_targets)
                
                if results:
                    return {
                        "source": "rsi_orchestrator",
                        "hypotheses": [r.hypothesis.__dict__ for r in results[:3]],  # Top 3
                        "generation_method": "advanced_rsi",
                        "targets_used": rsi_targets
                    }
            except Exception as e:
                logger.warning(f"RSI Orchestrator falhou: {e}")
        
        # Usar Revenue Generator como fallback
        if self.revenue_generator:
            try:
                # Descobrir oportunidades como hipóteses
                await self.revenue_generator._discover_opportunities()
                
                hypotheses = []
                for opp in self.revenue_generator.identified_opportunities[:3]:
                    hypothesis = {
                        "id": f"rev_{uuid.uuid4().hex[:8]}",
                        "description": opp.description,
                        "type": "revenue_opportunity",
                        "target_improvement": opp.estimated_revenue_potential,
                        "confidence": opp.confidence_score,
                        "strategy": opp.strategy.value
                    }
                    hypotheses.append(hypothesis)
                
                return {
                    "source": "revenue_generator",
                    "hypotheses": hypotheses,
                    "generation_method": "opportunity_analysis",
                    "targets_used": targets
                }
            except Exception as e:
                logger.warning(f"Revenue Generator falhou: {e}")
        
        # Fallback: gerar hipóteses simples
        fallback_hypotheses = [
            {
                "id": f"fb_{uuid.uuid4().hex[:8]}",
                "description": "Criar novos clientes usando API de customer management",
                "type": "customer_acquisition",
                "target_improvement": 100,  # 100 novos clientes
                "confidence": 0.7
            },
            {
                "id": f"fb_{uuid.uuid4().hex[:8]}",
                "description": "Otimizar preços usando dados de mercado",
                "type": "pricing_optimization",
                "target_improvement": 0.15,  # 15% melhoria
                "confidence": 0.8
            },
            {
                "id": f"fb_{uuid.uuid4().hex[:8]}",
                "description": "Implementar campanha de email marketing",
                "type": "marketing_automation",
                "target_improvement": 200,  # 200 emails enviados
                "confidence": 0.6
            }
        ]
        
        return {
            "source": "fallback_generator",
            "hypotheses": fallback_hypotheses,
            "generation_method": "simple_rules",
            "targets_used": targets
        }
    
    async def _agent_execute_hypotheses(self, rsi_hypotheses: Dict[str, Any]) -> Dict[str, Any]:
        """Agent executa hipóteses usando ferramentas reais"""
        
        execution_results = {
            "total_hypotheses": len(rsi_hypotheses.get("hypotheses", [])),
            "executed_successfully": 0,
            "execution_details": [],
            "real_metrics_collected": {},
            "agent_feedback": [],
            "tools_effectiveness": {}
        }
        
        # Executar cada hipótese
        for hypothesis in rsi_hypotheses.get("hypotheses", []):
            try:
                logger.info(f"🎯 Executando: {hypothesis.get('description', 'Unknown')[:50]}...")
                
                # Agent executa hipótese usando ferramentas reais
                result = await self.tool_agent.execute_rsi_hypothesis(hypothesis)
                
                execution_results["execution_details"].append(result)
                
                if result.get("success", False):
                    execution_results["executed_successfully"] += 1
                
                # Coletar métricas reais
                if "real_metrics" in result:
                    execution_results["real_metrics_collected"].update(result["real_metrics"])
                
                # Coletar feedback do agent
                if "agent_feedback" in result:
                    execution_results["agent_feedback"].append(result["agent_feedback"])
                
                logger.info(f"{'✅' if result.get('success') else '❌'} Hipótese executada")
                
            except Exception as e:
                logger.error(f"❌ Erro executando hipótese: {e}")
                execution_results["execution_details"].append({
                    "success": False,
                    "error": str(e),
                    "hypothesis_id": hypothesis.get("id", "unknown")
                })
        
        # Calcular taxa de sucesso geral
        execution_results["success_rate"] = (
            execution_results["executed_successfully"] / 
            max(execution_results["total_hypotheses"], 1)
        )
        
        return execution_results
    
    async def _rsi_learn_from_agent(self, hypotheses: Dict[str, Any], agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """RSI aprende dos resultados reais obtidos pelo Agent"""
        
        learning_insights = {
            "hypothesis_quality_assessment": {},
            "strategy_effectiveness": {},
            "target_accuracy": {},
            "improved_generation_patterns": [],
            "next_hypothesis_suggestions": []
        }
        
        # Analisar qualidade das hipóteses baseado nos resultados reais
        for i, hypothesis in enumerate(hypotheses.get("hypotheses", [])):
            if i < len(agent_results["execution_details"]):
                result = agent_results["execution_details"][i]
                
                # Avaliar qualidade da hipótese
                quality_score = 0.5  # Base
                if result.get("success", False):
                    quality_score += 0.3
                
                real_metrics = result.get("real_metrics", {})
                if real_metrics.get("real_actions_taken", 0) > 0:
                    quality_score += 0.2  # Bônus por ações reais
                
                learning_insights["hypothesis_quality_assessment"][hypothesis["id"]] = {
                    "quality_score": quality_score,
                    "execution_success": result.get("success", False),
                    "real_impact": real_metrics.get("real_actions_taken", 0)
                }
        
        # Identificar estratégias mais eficazes
        strategy_success = {}
        for hypothesis in hypotheses.get("hypotheses", []):
            strategy = hypothesis.get("type", "unknown")
            if strategy not in strategy_success:
                strategy_success[strategy] = []
        
        for i, result in enumerate(agent_results["execution_details"]):
            if i < len(hypotheses.get("hypotheses", [])):
                hypothesis = hypotheses["hypotheses"][i]
                strategy = hypothesis.get("type", "unknown")
                strategy_success[strategy].append(result.get("success", False))
        
        for strategy, successes in strategy_success.items():
            if successes:
                learning_insights["strategy_effectiveness"][strategy] = sum(successes) / len(successes)
        
        # Sugerir melhorias para próximas hipóteses
        if agent_results["success_rate"] < 0.7:
            learning_insights["next_hypothesis_suggestions"].extend([
                "Simplificar hipóteses para melhor execução",
                "Focar em ferramentas com maior taxa de sucesso",
                "Adicionar mais contexto para execução"
            ])
        
        if agent_results["success_rate"] > 0.8:
            learning_insights["next_hypothesis_suggestions"].extend([
                "Aumentar complexidade das hipóteses",
                "Explorar novas estratégias",
                "Combinar múltiplas abordagens"
            ])
        
        return learning_insights
    
    async def _agent_learn_from_rsi(self, hypotheses: Dict[str, Any], agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Agent aprende dos padrões do RSI para melhorar execução"""
        
        agent_insights = {
            "tool_optimization": {},
            "execution_pattern_improvements": [],
            "rsi_pattern_analysis": {},
            "next_execution_suggestions": []
        }
        
        # Analisar padrões nas hipóteses RSI
        hypothesis_patterns = {}
        for hypothesis in hypotheses.get("hypotheses", []):
            h_type = hypothesis.get("type", "unknown")
            if h_type not in hypothesis_patterns:
                hypothesis_patterns[h_type] = []
            hypothesis_patterns[h_type].append(hypothesis)
        
        agent_insights["rsi_pattern_analysis"] = {
            pattern: len(hyps) for pattern, hyps in hypothesis_patterns.items()
        }
        
        # Otimizar uso de ferramentas baseado nos tipos de hipótese
        tool_usage_patterns = []
        for result in agent_results["execution_details"]:
            if result.get("tools_used"):
                tool_usage_patterns.extend(result["tools_used"])
        
        # Contar frequência de ferramentas
        tool_frequency = {}
        for tool in tool_usage_patterns:
            tool_frequency[tool] = tool_frequency.get(tool, 0) + 1
        
        agent_insights["tool_optimization"] = {
            "most_used_tools": sorted(tool_frequency.keys(), key=lambda t: tool_frequency[t], reverse=True)[:3],
            "tool_frequency": tool_frequency
        }
        
        # Sugerir melhorias de execução
        if agent_results["success_rate"] < 0.8:
            agent_insights["next_execution_suggestions"].extend([
                "Melhorar tratamento de erros",
                "Adicionar mais validações",
                "Otimizar sequência de ferramentas"
            ])
        
        return agent_insights
    
    async def _synthesize_learnings(self, rsi_learning: Dict[str, Any], agent_learning: Dict[str, Any]) -> Dict[str, Any]:
        """Sintetiza aprendizados de RSI e Agent para melhorias combinadas"""
        
        synthesis = {
            "combined_improvements": [],
            "synergy_opportunities": [],
            "integrated_strategies": [],
            "next_cycle_optimizations": {}
        }
        
        # Combinar insights de qualidade de hipótese + eficácia de ferramentas
        rsi_strategies = rsi_learning.get("strategy_effectiveness", {})
        agent_tools = agent_learning.get("tool_optimization", {}).get("most_used_tools", [])
        
        for strategy, effectiveness in rsi_strategies.items():
            if effectiveness > 0.7:  # Estratégias eficazes
                synthesis["combined_improvements"].append(
                    f"Estratégia '{strategy}' eficaz - combinar com ferramentas {agent_tools[:2]}"
                )
        
        # Identificar oportunidades de sinergia
        if rsi_learning.get("next_hypothesis_suggestions") and agent_learning.get("next_execution_suggestions"):
            synthesis["synergy_opportunities"].append(
                "RSI pode gerar hipóteses mais específicas enquanto Agent melhora execução"
            )
        
        # Estratégias integradas para próximo ciclo
        synthesis["integrated_strategies"] = [
            "RSI foca em estratégias comprovadamente eficazes",
            "Agent prioriza ferramentas com maior taxa de sucesso",
            "Ambos colaboram para hipóteses mais executáveis"
        ]
        
        # Otimizações para próximo ciclo
        synthesis["next_cycle_optimizations"] = {
            "rsi_focus": list(rsi_strategies.keys())[:2] if rsi_strategies else ["revenue_optimization"],
            "agent_tool_priority": agent_tools[:3],
            "combined_targets": {
                "hypothesis_executability": 0.85,
                "tool_success_rate": 0.90,
                "real_impact_ratio": 0.75
            }
        }
        
        return synthesis
    
    def _calculate_cycle_success_score(self, agent_results: Dict[str, Any], combined_improvement: Dict[str, Any]) -> float:
        """Calcula score de sucesso do ciclo de co-evolução"""
        
        factors = [
            agent_results.get("success_rate", 0) * 0.4,  # 40% execução
            len(combined_improvement.get("combined_improvements", [])) / 5 * 0.3,  # 30% melhorias
            len(combined_improvement.get("synergy_opportunities", [])) / 3 * 0.2,  # 20% sinergia
            len(combined_improvement.get("integrated_strategies", [])) / 3 * 0.1   # 10% integração
        ]
        
        return min(sum(factors), 1.0)
    
    def _evolve_targets(self, current_targets: Dict[str, float], cycle_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evolui targets baseado no resultado do ciclo"""
        
        new_targets = current_targets.copy()
        
        success_score = cycle_result["success_score"]
        
        # Se ciclo foi muito bem-sucedido, aumentar targets
        if success_score > 0.8:
            for key in new_targets:
                new_targets[key] = min(new_targets[key] * 1.1, 1.0)  # Máximo 100%
        
        # Se ciclo teve problemas, tornar targets mais conservadores
        elif success_score < 0.6:
            for key in new_targets:
                new_targets[key] = max(new_targets[key] * 0.9, 0.1)  # Mínimo 10%
        
        return new_targets
    
    async def _analyze_coevolution_session(self, session_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa sessão completa de co-evolução"""
        
        analysis = {
            "rsi_evolution": {
                "hypothesis_quality_trend": [],
                "strategy_learning": {},
                "generation_improvements": []
            },
            "agent_evolution": {
                "execution_efficiency_trend": [],
                "tool_mastery": {},
                "pattern_recognition": []
            },
            "combined_achievements": {
                "synergy_development": 0.0,
                "mutual_learning": 0.0,
                "overall_improvement": 0.0
            },
            "next_cycle_suggestions": []
        }
        
        # Analisar tendências de evolução
        if self.coevolution_cycles:
            success_scores = [cycle.success_score for cycle in self.coevolution_cycles]
            analysis["combined_achievements"]["overall_improvement"] = (
                success_scores[-1] - success_scores[0] if len(success_scores) > 1 else success_scores[0]
            )
        
        # Sugestões para próximas sessões
        analysis["next_cycle_suggestions"] = [
            "Continuar focando em estratégias bem-sucedidas",
            "Expandir conjunto de ferramentas disponíveis",
            "Implementar feedback mais granular entre RSI e Agent",
            "Adicionar métricas de impacto real no mundo"
        ]
        
        return analysis
    
    async def _save_coevolution_session(self, session_results: Dict[str, Any]) -> None:
        """Salva resultados da sessão de co-evolução"""
        
        session_file = self.data_dir / f"coevolution_session_{session_results['session_id']}.json"
        
        # Preparar dados para serialização
        serializable_results = session_results.copy()
        
        with open(session_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Sessão de co-evolução salva: {session_file}")
    
    async def get_coevolution_status(self) -> Dict[str, Any]:
        """Retorna status atual da co-evolução"""
        
        return {
            "total_cycles": len(self.coevolution_cycles),
            "average_success_score": (
                sum(c.success_score for c in self.coevolution_cycles) / len(self.coevolution_cycles)
                if self.coevolution_cycles else 0
            ),
            "rsi_improvements": len(self.rsi_improvements),
            "agent_improvements": len(self.agent_improvements),
            "agent_status": await self.tool_agent.get_agent_status(),
            "evolution_trajectory": [c.success_score for c in self.coevolution_cycles[-10:]],  # Últimos 10
            "last_cycle": self.coevolution_cycles[-1].timestamp.isoformat() if self.coevolution_cycles else None
        }

# Factory function
def create_rsi_agent_orchestrator(rsi_orchestrator=None, revenue_generator=None, base_url="http://localhost:8000"):
    """Cria orquestrador de co-evolução RSI-Agent"""
    return RSIAgentOrchestrator(rsi_orchestrator, revenue_generator, base_url)