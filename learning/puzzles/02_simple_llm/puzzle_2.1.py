"""
PUZZLE 2.1: Predict Sampling Behavior - [Easy]

TASK:
Predict and understand how different sampling parameters affect output.

GIVEN:
- Same prompt with different sampling parameters
- Understanding of temperature, top_k, top_p

CHALLENGE:
- Predict which configuration will be most deterministic
- Predict which will be most diverse
- Understand the interaction between parameters

HINT:
- temperature=0.0 → deterministic (greedy)
- temperature>0.0 → more random
- top_k limits vocabulary size
- top_p uses cumulative probability

QUESTIONS:
1. Which will be most deterministic: temp=0.0 or temp=0.7?
2. What's the difference between top_k=1 and top_k=10?
3. How does top_p interact with temperature?

TEST:
Run: pytest learning/puzzles/02_simple_llm/test_2.1.py -v
"""

import sys
import importlib.util
from pathlib import Path

# Import puzzle_1.1 using importlib (module name has dot)
puzzle_dir = Path(__file__).parent.parent / "01_core_structures"
puzzle_1_1_file = puzzle_dir / "puzzle_1.1.py"
spec = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_1_1)
create_sampling_params = puzzle_1_1.create_sampling_params
SamplingParams = puzzle_1_1.SamplingParams


def predict_sampling_behavior(params: SamplingParams) -> dict:
    """
    Predict the behavior of sampling parameters.

    Returns a dictionary with predictions:
    - "deterministic": bool - Will output be deterministic?
    - "diversity": str - "low", "medium", "high"
    - "vocab_size": str - "full", "limited", "very_limited"

    TODO: Implement this function to analyze sampling parameters
    """
    result = {
        "deterministic": False,  # TODO: Determine if params.is_greedy
        "diversity": "medium",    # TODO: Predict based on temperature
        "vocab_size": "full",    # TODO: Predict based on top_k and top_p
    }

    result["deterministic"] = params.is_greedy
    result["diversity"] = "low" if params.is_greedy or params.temperature <= 0.3 else "high" if params.temperature >= 0.8 else "medium"
    result["vocab_size"] = "full" if (params.top_k == -1 and params.top_p == 1.0) else "very_limited" if params.top_k <= 3 or params.top_p < 0.5 else "limited"

    return result


def compare_sampling_configs() -> dict:
    """
    Compare different sampling configurations.

    Returns a dictionary mapping config name to behavior prediction.
    """
    configs = {
        "greedy": create_sampling_params(temperature=0.0, top_k=1, top_p=1.0),
        "creative": create_sampling_params(temperature=1.0, top_k=50, top_p=0.9),
        "balanced": create_sampling_params(temperature=0.7, top_k=10, top_p=0.95),
        "focused": create_sampling_params(temperature=0.3, top_k=5, top_p=0.8),
    }

    results = {}
    for name, params in configs.items():
        results[name] = predict_sampling_behavior(params)

    return results


# Test your predictions
if __name__ == "__main__":
    results = compare_sampling_configs()
    for name, behavior in results.items():
        print(f"{name}: {behavior}")
