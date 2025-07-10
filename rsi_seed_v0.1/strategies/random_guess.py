import random
from typing import List, Tuple

from .base_strategy import BaseGuessingStrategy

class RandomStrategy(BaseGuessingStrategy):
    """Inefficient random guessing strategy."""

    def guess(self, history: List[Tuple[int, int]], min_val: int, max_val: int) -> int:
        tried = {g for g, _ in history}
        possible = [n for n in range(min_val, max_val + 1) if n not in tried]
        return random.choice(possible)
