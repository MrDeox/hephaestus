import random
from typing import List

import pandas as pd
import numpy as np

from .task_definitions import DataChallenge


def _generate_classification(challenge_id: int) -> DataChallenge:
    """Create a simple classification dataset."""
    # Synthetic binary classification
    data = pd.DataFrame({
        "feature1": np.random.randn(20),
        "feature2": np.random.randn(20),
    })
    data["label"] = (data["feature1"] + data["feature2"] > 0).astype(int)
    return DataChallenge(
        challenge_id=challenge_id,
        task_type="classification",
        data=data,
        parameters={"target_column": "label"},
    )


def _generate_clustering(challenge_id: int) -> DataChallenge:
    """Create a simple clustering dataset."""
    centers = [(-1, -1), (1, 1)]
    features, _ = make_blobs(n_samples=20, centers=centers, n_features=2, random_state=challenge_id)
    data = pd.DataFrame(features, columns=["x", "y"])
    return DataChallenge(
        challenge_id=challenge_id,
        task_type="clustering",
        data=data,
        parameters={"n_clusters": 2},
    )


def _generate_optimization(challenge_id: int) -> DataChallenge:
    """Create a simple optimization search dataset."""
    data = pd.DataFrame({
        "cost": np.random.randint(1, 10, size=10),
        "benefit": np.random.randint(1, 20, size=10),
    })
    return DataChallenge(
        challenge_id=challenge_id,
        task_type="optimization_search",
        data=data,
        parameters={},
    )


GENERATORS = [_generate_classification, _generate_clustering, _generate_optimization]


from sklearn.datasets import make_blobs

def generate_challenges(num_challenges: int = 5) -> List[DataChallenge]:
    """Generate a list of synthetic data challenges."""
    challenges: List[DataChallenge] = []
    for cid in range(num_challenges):
        generator = random.choice(GENERATORS)
        challenge = generator(cid)
        challenges.append(challenge)
    return challenges

__all__ = ["generate_challenges", "DataChallenge"]
