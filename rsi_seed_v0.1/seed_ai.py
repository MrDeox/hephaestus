import random
from typing import Dict, Tuple, List

import governance
from modules.reflection_module import evaluate_performance
from modules.adaptation_module import select_new_strategy
from strategies.random_guess import RandomStrategy
from strategies.binary_search import BinarySearchStrategy


class SeedAI:
    """Seed AI capable of simple self-improvement."""

    def __init__(self):
        self.directives = governance.CoreDirectives
        self.available_strategies: Dict[str, object] = {
            "RandomStrategy": RandomStrategy(),
            "BinarySearchStrategy": BinarySearchStrategy(),
        }
        self.strategy_name = "RandomStrategy"
        self.strategy = self.available_strategies[self.strategy_name]
        self.cycle_count = 0

    def run_mission(self, secret_number: int) -> Tuple[List[Tuple[int, int]], int]:
        history: List[Tuple[int, int]] = []
        min_val, max_val = 1, 100
        num_guesses = 0
        while True:
            guess = self.strategy.guess(history, min_val, max_val)
            if guess < min_val or guess > max_val:
                raise ValueError("Guess out of bounds")
            if any(g == guess for g, _ in history):
                raise ValueError("Strategy repeated guess")
            num_guesses += 1
            if guess == secret_number:
                history.append((guess, 0))
                break
            elif guess < secret_number:
                history.append((guess, -1))
            else:
                history.append((guess, 1))
        return history, num_guesses

    def run_full_cycle(self):
        self.cycle_count += 1
        secret_number = random.randint(1, 100)
        history, attempts = self.run_mission(secret_number)
        performance = evaluate_performance(attempts)
        prev_strategy_name = self.strategy_name
        self.strategy, self.strategy_name = select_new_strategy(
            self.strategy_name, performance, self.available_strategies
        )
        print(
            f"Ciclo {self.cycle_count} | Estratégia Usada: {prev_strategy_name} | "
            f"Tentativas: {attempts} | Nova Estratégia Selecionada: {self.strategy_name}"
        )

