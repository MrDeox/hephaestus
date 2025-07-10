import os
import sys
import importlib.util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Dynamic import because filename contains a dot
module_path = os.path.join(CURRENT_DIR, "seed_ai_v0.2.py")
spec = importlib.util.spec_from_file_location("seed_ai_v0_2", module_path)
seed_ai_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed_ai_module)
SeedAI_v2 = seed_ai_module.SeedAI_v2

from task_handler.task_generator import generate_challenges


def main() -> None:
    ai = SeedAI_v2()
    challenges = generate_challenges(5)
    for challenge in challenges:
        ai.process_challenge(challenge)


if __name__ == "__main__":
    main()
