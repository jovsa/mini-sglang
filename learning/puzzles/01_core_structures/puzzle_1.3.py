"""
PUZZLE 1.3: Implement Request State Transitions - [Hard]

TASK:
Implement complete_one() and append_host() methods for Req.
These methods handle state transitions during request processing.

GIVEN:
- Req class structure
- Understanding of request lifecycle: prefill → decode → completion

CHALLENGE:
- complete_one() should update cached_len and device_len correctly
- append_host() should append a token to input_ids (on host/CPU)
- Note: append_host() does NOT update device_len - only complete_one() does
- Understand the state machine: cached_len tracks processed tokens, device_len tracks current position

HINT:
- Read python/minisgl/core.py lines 51-56
- complete_one() is called after processing one token
- append_host() is called to add a new token to the sequence

QUESTIONS:
1. What's the difference between cached_len and device_len?
2. When is complete_one() called vs append_host()?
3. How does the request transition from prefill to decode?

TEST:
Run: pytest learning/puzzles/01_core_structures/test_1.3.py -v
"""

import torch
from dataclasses import dataclass
import sys
import importlib.util
from pathlib import Path
puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle files with dots in names using importlib
puzzle_1_2_file = puzzle_dir / "puzzle_1.2.py"
spec_1_2 = importlib.util.spec_from_file_location("puzzle_1_2", puzzle_1_2_file)
puzzle_1_2 = importlib.util.module_from_spec(spec_1_2)
spec_1_2.loader.exec_module(puzzle_1_2)
Req = puzzle_1_2.Req
MockCacheHandle = puzzle_1_2.MockCacheHandle

puzzle_1_1_file = puzzle_dir / "puzzle_1.1.py"
spec_1_1 = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec_1_1)
spec_1_1.loader.exec_module(puzzle_1_1)
create_sampling_params = puzzle_1_1.create_sampling_params


class ExtendedReq(Req):
    """Extended Req with methods to implement."""

    def __post_init__(self) -> None:
        """
        Override __post_init__ to allow cached_len == device_len.
        This is needed for the decode phase when all input tokens are processed.
        """
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        # Allow cached_len == device_len for decode phase (all input tokens processed)
        assert 0 <= self.cached_len <= self.device_len <= self.max_device_len

    def complete_one(self) -> None:
        """
        Mark one token as complete (processed and cached).

        This is called after processing a token during prefill or decode.
        It should:
        1. Update cached_len to device_len (mark current position as cached)
        2. Increment device_len by 1 (move to next position)

        HINT: Look at core.py:51-53
        """
        # TODO: Implement this method
        # YOUR CODE HERE
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """
        Append a new token to the input_ids (host/CPU side only).

        This is called when a new token is generated during decode.
        It should:
        1. Append next_token to input_ids using torch.cat

        NOTE: This does NOT update device_len. The device_len is only
        updated by complete_one() when a token is actually processed.

        HINT: Look at core.py:55-56
        """
        # TODO: Implement this method
        # YOUR CODE HERE
        self.input_ids = torch.cat([self.input_ids, next_token])


def create_extended_req(
    input_ids: torch.Tensor,
    output_len: int,
    cached_len: int = 0,
) -> ExtendedReq:
    """Helper to create an ExtendedReq."""
    sampling_params = create_sampling_params(max_tokens=output_len)
    cache_handle = MockCacheHandle(cached_len=cached_len)

    return ExtendedReq(
        input_ids=input_ids,
        table_idx=0,
        cached_len=cached_len,
        output_len=output_len,
        uid=0,
        sampling_params=sampling_params,
        cache_handle=cache_handle,
    )


# Test your implementation
if __name__ == "__main__":
    # Test 1: complete_one during prefill
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    req = create_extended_req(input_ids, output_len=5, cached_len=0)

    initial_device_len = req.device_len
    initial_cached_len = req.cached_len

    req.complete_one()

    print(f"After complete_one:")
    print(f"  cached_len: {req.cached_len} (was {initial_cached_len}, expected {initial_device_len})")
    print(f"  device_len: {req.device_len} (was {initial_device_len}, expected {initial_device_len + 1})")

    # Test 2: append_host during decode
    req.append_host(torch.tensor([99], dtype=torch.int32))
    print(f"\nAfter append_host:")
    print(f"  input_ids length: {len(req.input_ids)} (expected 4)")
    print(f"  device_len: {req.device_len} (expected {initial_device_len + 1}, unchanged by append_host)")
