from typing import Any, Dict, List, Tuple

import pandas as pd

import governance
from task_handler.task_definitions import DataChallenge
from learning_tools.supervised_tool import ClassifierTool
from learning_tools.unsupervised_tool import ClusteringTool
from learning_tools.reinforcement_tool import OptimizationTool


class SeedAI_v2:
    """Seed AI v0.2 with simple meta-learning capabilities."""

    def __init__(self) -> None:
        self.directives = governance.CoreDirectives
        self.tools: Dict[str, Any] = {
            "classification": ClassifierTool(),
            "clustering": ClusteringTool(),
            "optimization_search": OptimizationTool(),
        }
        # policy maps task_type to tool key
        self.policy: Dict[str, str] = {
            "classification": "classification",
            "clustering": "clustering",
            "optimization_search": "optimization_search",
        }
        self.performance_log: List[Tuple[int, str, str, float]] = []

    def process_challenge(self, challenge: DataChallenge) -> None:
        task_type = challenge.task_type
        tool_key = self.policy.get(task_type)
        if tool_key is None:
            print(f"Nenhuma ferramenta configurada para {task_type}")
            return
        tool = self.tools[tool_key]
        print(f"Desafio {challenge.challenge_id} recebido: {task_type}")
        print(f"Política seleciona: {tool.__class__.__name__}")
        result = tool.execute(challenge.data, challenge.parameters)
        print(f"Resultado obtido: {result}")
        performance_score = self._extract_score(result)
        self.performance_log.append((challenge.challenge_id, task_type, tool_key, performance_score))
        self.adapt_policy()

    def _extract_score(self, result: Dict[str, Any]) -> float:
        for value in result.values():
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def adapt_policy(self) -> None:
        if not self.performance_log:
            return
        summary: Dict[str, List[float]] = {}
        for _, ttype, _, score in self.performance_log:
            summary.setdefault(ttype, []).append(score)
        print("Análise de Meta-Aprendizado:")
        for ttype, scores in summary.items():
            avg = sum(scores) / len(scores)
            print(f" - {ttype}: desempenho médio {avg:.2f}")
        print()

__all__ = ["SeedAI_v2"]
