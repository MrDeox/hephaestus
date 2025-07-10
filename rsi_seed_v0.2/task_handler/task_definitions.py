from dataclasses import dataclass
from typing import Any, Dict
import pandas as pd

@dataclass
class DataChallenge:
    """Structure for a data challenge presented to the Seed AI."""

    challenge_id: int
    task_type: str  # 'classification', 'clustering', or 'optimization_search'
    data: pd.DataFrame
    parameters: Dict[str, Any]

__all__ = ["DataChallenge"]
