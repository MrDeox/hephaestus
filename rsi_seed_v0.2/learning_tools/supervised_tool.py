from typing import Any, Dict

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from .base_tool import BaseLearningTool


class ClassifierTool(BaseLearningTool):
    """Simple KNN classifier tool."""

    def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Any:
        target_col = params.get("target_column")
        if target_col is None or target_col not in data.columns:
            raise ValueError("target_column missing in params or data")

        X = data.drop(columns=[target_col])
        y = data[target_col]
        # Fit KNN on entire dataset and predict on same for simplicity
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = accuracy_score(y, preds)
        return {"accuracy": accuracy}

__all__ = ["ClassifierTool"]
