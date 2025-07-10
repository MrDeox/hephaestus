from typing import Any, Dict

import pandas as pd
from sklearn.cluster import KMeans

from .base_tool import BaseLearningTool


class ClusteringTool(BaseLearningTool):
    """Simple K-Means clustering tool."""

    def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Any:
        n_clusters = int(params.get("n_clusters", 2))
        model = KMeans(n_clusters=n_clusters, n_init=10)
        model.fit(data)
        inertia = model.inertia_
        return {"inertia": inertia}

__all__ = ["ClusteringTool"]
