from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


class BaseLearningTool(ABC):
    """Abstract base class for learning tools."""

    @abstractmethod
    def execute(self, data: pd.DataFrame, params: Dict[str, Any]) -> Any:
        """Process the provided data and return a performance metric."""
        raise NotImplementedError

__all__ = ["BaseLearningTool"]
