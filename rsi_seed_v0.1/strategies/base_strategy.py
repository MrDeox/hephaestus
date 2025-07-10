from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseGuessingStrategy(ABC):
    """Abstract base class for guessing strategies."""

    @abstractmethod
    def guess(self, history: List[Tuple[int, int]], min_val: int, max_val: int) -> int:
        """Return next guess based on history and bounds."""
        raise NotImplementedError
