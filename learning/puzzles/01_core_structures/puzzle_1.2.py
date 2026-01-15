"""
PUZZLE 1.2: Fix the Req Creation - [Medium]

TASK:
Fix bugs in the Req creation code. There are intentional bugs that need to be fixed.

GIVEN:
- Partial Req implementation with bugs
- Understanding of what Req should do

CHALLENGE:
- Fix wrong cached_len value
- Add missing validation
- Understand the relationship between cached_len, device_len, and max_device_len

HINT:
- Read python/minisgl/core.py lines 27-66, especially __post_init__
- cached_len should be less than device_len
- device_len should be less than or equal to max_device_len
- cached_len should be >= 0

QUESTIONS:
1. What happens if cached_len >= device_len? Why is this invalid?
2. What's the difference between device_len and max_device_len?
3. Why does input_ids need to be a CPU tensor?

TEST:
Run: pytest learning/puzzles/01_core_structures/test_1.2.py -v
"""

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This is just for type hints - you don't need to implement BaseCacheHandle
    from minisgl.kvcache import BaseCacheHandle


# Mock cache handle for testing
class MockCacheHandle:
    def __init__(self, cached_len: int = 0):
        self.cached_len = cached_len


@dataclass(eq=False)
class Req:
    """Represents a single inference request."""
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: object  # SamplingParams
    cache_handle: object  # BaseCacheHandle

    def __post_init__(self) -> None:
        """
        Validate and initialize request state.

        BUGS TO FIX:
        1. Missing validation that input_ids is on CPU
        2. Wrong calculation of device_len (should be len(input_ids))
        3. Wrong calculation of max_device_len
        4. Missing validation that cached_len < device_len
        5. Missing validation that device_len <= max_device_len
        """
        # TODO: Fix bug 1 - Check that input_ids is on CPU
        # YOUR CODE HERE

        # TODO: Fix bug 2 - Set device_len correctly (should be length of input_ids)
        # YOUR CODE HERE
        self.device_len = 0  # BUG: This is wrong!

        # TODO: Fix bug 3 - Calculate max_device_len correctly
        # max_device_len = input_length + output_length
        # YOUR CODE HERE
        self.max_device_len = 0  # BUG: This is wrong!

        # TODO: Fix bug 4 - Validate that cached_len < device_len
        # YOUR CODE HERE

        # TODO: Fix bug 5 - Validate that device_len <= max_device_len
        # YOUR CODE HERE

    @property
    def remain_len(self) -> int:
        """Remaining tokens that can be generated."""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """Number of new tokens to process in this step."""
        return self.device_len - self.cached_len

    def can_decode(self) -> bool:
        """Check if request can continue decoding."""
        return self.remain_len > 0


def create_req(
    input_ids: torch.Tensor,
    output_len: int,
    uid: int = 0,
    cached_len: int = 0,
    table_idx: int = 0,
) -> Req:
    """
    Create a Req object. Fix any bugs in this function too!

    Args:
        input_ids: Token IDs (must be CPU tensor)
        output_len: Maximum output length
        uid: Unique request ID
        cached_len: Number of tokens already cached (processed)
        table_idx: Table index for this request
    """
    import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from puzzle_1_1 import create_sampling_params

    # TODO: Ensure input_ids is on CPU if it's not already
    # YOUR CODE HERE

    sampling_params = create_sampling_params(max_tokens=output_len)
    cache_handle = MockCacheHandle(cached_len=cached_len)

    # TODO: Create and return Req object
    # YOUR CODE HERE
    pass


# Test your implementation
if __name__ == "__main__":
    # Test 1: Basic request creation
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    req = create_req(input_ids, output_len=10, cached_len=0)
    print(f"Test 1 - device_len={req.device_len}, expected=5")
    print(f"Test 1 - max_device_len={req.max_device_len}, expected=15")
    print(f"Test 1 - remain_len={req.remain_len}, expected=10")

    # Test 2: Request with cached tokens
    req2 = create_req(input_ids, output_len=10, cached_len=2)
    print(f"Test 2 - extend_len={req2.extend_len}, expected=3")
