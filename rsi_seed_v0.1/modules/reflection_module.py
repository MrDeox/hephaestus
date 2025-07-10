def evaluate_performance(num_guesses: int) -> float:
    """Return efficiency score. Higher is better."""
    if num_guesses <= 0:
        return 0.0
    return 1.0 / float(num_guesses)

__all__ = ["evaluate_performance"]
