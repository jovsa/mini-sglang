"""
PUZZLE 1.4: Batch Creation Challenge - [Medium]

TASK:
Create batches for prefill and decode phases.
Handle requests with different sequence lengths.

GIVEN:
- List of Req objects
- Understanding of Batch structure

CHALLENGE:
- Create Batch objects for prefill and decode phases
- Understand the difference between Batch.size and Batch.padded_size
- Handle requests with varying lengths

HINT:
- Reference: `python/minisgl/core.py:88-94` for size/padded_size properties
- Batch.size returns len(reqs) - the actual number of requests
- Batch.padded_size returns len(padded_reqs) - includes padding requests
- padded_reqs is set after Batch creation by the scheduler

QUESTIONS:
1. What's the difference between Batch.size and Batch.padded_size?
2. Why do we need padding in batches?
3. When would you use prefill vs decode batch?

TEST:
Run: pytest learning/puzzles/01_core_structures/test_1.4.py -v
"""

from dataclasses import dataclass, field
from typing import List, Literal
import sys
import importlib.util
from pathlib import Path
puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle file with dot in name using importlib
puzzle_1_2_file = puzzle_dir / "puzzle_1.2.py"
spec_1_2 = importlib.util.spec_from_file_location("puzzle_1_2", puzzle_1_2_file)
puzzle_1_2 = importlib.util.module_from_spec(spec_1_2)
spec_1_2.loader.exec_module(puzzle_1_2)
Req = puzzle_1_2.Req


@dataclass
class Batch:
    """Represents a batch of requests for processing."""
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # These fields should be set by scheduler
    input_ids: object = field(init=False)  # torch.Tensor
    out_loc: object = field(init=False)  # torch.Tensor
    padded_reqs: List[Req] = field(init=False)  # May contain dummy reqs for padding
    # This field should be set by attention backend
    attn_metadata: object = field(init=False)

    def __post_init__(self) -> None:
        self.padded_reqs = self.reqs

    @property
    def is_prefill(self) -> bool:
        """Check if this is a prefill batch."""
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        """Check if this is a decode batch."""
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """Number of real requests in the batch."""
        # TODO: Implement this property
        # YOUR CODE HERE
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """Total size including padding requests."""
        # TODO: Implement this property
        # YOUR CODE HERE
        return len(self.padded_reqs)


def create_prefill_batch(reqs: List[Req]) -> Batch:
    """
    Create a prefill batch from a list of requests.

    Args:
        reqs: List of requests to batch together

    Returns:
        Batch object with phase="prefill"
    """
    # TODO: Create a Batch object for prefill phase
    # For now, padded_reqs can be the same as reqs (no padding)
    # YOUR CODE HERE
    return Batch(reqs=reqs, phase="prefill")


def create_decode_batch(reqs: List[Req]) -> Batch:
    """
    Create a decode batch from a list of requests.

    Args:
        reqs: List of requests to batch together

    Returns:
        Batch object with phase="decode"
    """
    # TODO: Create a Batch object for decode phase
    # YOUR CODE HERE
    return Batch(reqs=reqs, phase="decode")


# Test your implementation
if __name__ == "__main__":
    create_req = puzzle_1_2.create_req
    import torch

    # Create some test requests
    req1 = create_req(torch.tensor([1, 2, 3], dtype=torch.int32), output_len=5)
    req2 = create_req(torch.tensor([4, 5], dtype=torch.int32), output_len=3)
    req3 = create_req(torch.tensor([6, 7, 8, 9], dtype=torch.int32), output_len=10)

    # Test prefill batch
    prefill_batch = create_prefill_batch([req1, req2, req3])
    print(f"Prefill batch: phase={prefill_batch.phase}, size={prefill_batch.size}")

    # Test decode batch
    decode_batch = create_decode_batch([req1, req2])
    print(f"Decode batch: phase={decode_batch.phase}, size={decode_batch.size}")
