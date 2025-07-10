from typing import Dict

from strategies.random_guess import RandomStrategy
from strategies.binary_search import BinarySearchStrategy


def select_new_strategy(current_strategy_name: str, performance_score: float, available_strategies: Dict[str, object]):
    """Select a new strategy based on performance."""
    if (
        performance_score < 0.1
        and current_strategy_name == "RandomStrategy"
        and "BinarySearchStrategy" in available_strategies
    ):
        return available_strategies["BinarySearchStrategy"], "BinarySearchStrategy"
    return available_strategies[current_strategy_name], current_strategy_name

__all__ = ["select_new_strategy"]
