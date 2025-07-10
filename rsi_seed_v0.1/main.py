import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from seed_ai import SeedAI


def main():
    ai = SeedAI()
    for _ in range(5):
        ai.run_full_cycle()


if __name__ == "__main__":
    main()
