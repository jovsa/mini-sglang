"""
PUZZLE 1.1: Complete the SamplingParams - [Easy]

TASK:
Fill in the missing properties to create a SamplingParams object.
You need to implement the create_sampling_params function and the is_greedy property.

GIVEN:
- The SamplingParams dataclass structure from core.py
- Partial implementation below

CHALLENGE:
- Understand what each parameter means
- Implement the is_greedy property correctly
- Handle edge cases

HINT:
- Read python/minisgl/core.py lines 14-25
- is_greedy should return True when temperature <= 0.0 OR (top_k == 1 AND top_p == 1.0)

QUESTIONS:
1. What happens if temperature=0.0 and top_p=0.5? Is it greedy?
2. What's the difference between top_k=-1 and top_k=1?
3. When would ignore_eos be useful?

TEST:
Run: pytest learning/puzzles/01_core_structures/test_1.1.py -v
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    # TODO: Implement the is_greedy property
    # It should return True when sampling is deterministic (greedy)
    # Hint: Check core.py:22-24 for the logic
    @property
    def is_greedy(self) -> bool:
        """
        Returns True if the sampling parameters result in greedy (deterministic) sampling.

        Greedy sampling occurs when:
        - temperature <= 0.0, OR
        - top_k == 1 AND top_p == 1.0
        """
        # YOUR CODE HERE
        return self.temperature <= 0.0 or (self.top_k == 1 and self.top_p == 1.0)


def create_sampling_params(
    temperature: float = 0.0,
    top_k: int = -1,
    top_p: float = 1.0,
    ignore_eos: bool = False,
    max_tokens: int = 1024,
) -> SamplingParams:
    """
    Create a SamplingParams object with the given parameters.

    Args:
        temperature: Sampling temperature. Lower = more deterministic.
        top_k: Sample from top k tokens. -1 means no limit.
        top_p: Nucleus sampling. Sample from tokens with cumulative prob >= top_p.
        ignore_eos: Whether to ignore end-of-sequence token.
        max_tokens: Maximum tokens to generate.

    Returns:
        SamplingParams object
    """
    # TODO: Create and return a SamplingParams object
    # YOUR CODE HERE
    res = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, ignore_eos=ignore_eos, max_tokens=max_tokens)
    return res



# Test your implementation
if __name__ == "__main__":
    # Test 1: Greedy sampling
    params1 = create_sampling_params(temperature=0.0, top_k=1, top_p=1.0)
    print(f"Test 1 - Greedy: is_greedy={params1.is_greedy}, expected=True")

    # Test 2: Non-greedy sampling
    params2 = create_sampling_params(temperature=0.7, top_k=10, top_p=0.9)
    print(f"Test 2 - Sampling: is_greedy={params2.is_greedy}, expected=False")

    # Test 3: Edge case - temperature 0 but top_p < 1
    params3 = create_sampling_params(temperature=0.0, top_p=0.5)
    print(f"Test 3 - Edge case: is_greedy={params3.is_greedy}")
