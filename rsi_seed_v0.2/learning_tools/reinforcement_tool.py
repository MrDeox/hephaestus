from typing import Any, Dict

import pandas as pd

from .base_tool import BaseLearningTool


class OptimizationTool(BaseLearningTool):
    """Simplified search for best benefit/cost ratio."""

    def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Any:
        if not {"cost", "benefit"}.issubset(data.columns):
            raise ValueError("Data must contain 'cost' and 'benefit' columns")
        scores = data["benefit"] / data["cost"]
        best_idx = scores.idxmax()
        best_item = data.loc[best_idx].to_dict()
        best_score = scores[best_idx]
        return {"best_item": best_item, "score": float(best_score)}

__all__ = ["OptimizationTool"]
