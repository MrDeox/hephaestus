from typing import List, Tuple

from .base_strategy import BaseGuessingStrategy

class BinarySearchStrategy(BaseGuessingStrategy):
    """Optimized binary search guessing strategy."""

    def guess(self, history: List[Tuple[int, int]], min_val: int, max_val: int) -> int:
        low, high = min_val, max_val
        for g, result in history:
            if result < 0:  # guess was too low
                low = max(low, g + 1)
            elif result > 0:  # guess was too high
                high = min(high, g - 1)
        return (low + high) // 2
