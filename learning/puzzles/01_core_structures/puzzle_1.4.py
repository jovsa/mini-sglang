"""
PUZZLE 1.4: Batch Creation Challenge - [Medium]

TASK:
Create batches for prefill and decode phases.
Handle requests with different sequence lengths.

GIVEN:
- List of Req objects
- Understanding of Batch structure

================================================================================
DATA FLOW: From Manager to Batch to Engine
================================================================================

Batches are created by managers and flow through the scheduler to the engine:

```mermaid
flowchart TD
    A[PrefillManager] -->|"schedule_next_batch"| B[Create Batch]
    C[DecodeManager] -->|"schedule_next_batch"| B
    B -->|"phase=prefill or decode"| D[Batch object]
    D -->|"Scheduler._prepare_batch"| E[Allocate cache pages]
    E -->|"Pad if needed"| F[Set padded_reqs]
    F -->|"Create input_ids tensor"| G[Concatenate with padding]
    G -->|"Create out_loc tensor"| H[Map outputs to Reqs]
    H -->|"Engine.forward"| I[Process batch on GPU]
```

Step-by-step flow:
1. **Manager.schedule_next_batch()** (`python/minisgl/scheduler/prefill.py` or `decode.py`):
   - Collects runnable Reqs
   - Creates Batch with `phase="prefill"` or `phase="decode"`

2. **Scheduler._prepare_batch()** (`python/minisgl/scheduler/scheduler.py:180-201`):
   - Allocates cache pages via `CacheManager.allocate()`
   - Pads batch if needed via `GraphRunner.pad_batch()` (for CUDA graph optimization)
   - Creates `input_ids` tensor by concatenating all Req input_ids (with padding)
   - Creates `out_loc` tensor mapping output positions to Reqs
   - Prepares attention metadata

3. **Engine.forward()** (`python/minisgl/engine/engine.py`):
   - Receives Batch with prepared tensors
   - Processes all Reqs in parallel on GPU
   - Returns next tokens for each Req

**Key Differences: Prefill vs Decode Batches**
- **Prefill**: Variable sequence lengths (different input sizes), requires padding
- **Decode**: All sequences length 1 (generating one token at a time), minimal padding

================================================================================
COMPONENT CONNECTIONS: How Batch Connects Scheduler to Engine
================================================================================

```mermaid
graph TB
    subgraph "Scheduler Layer"
        PM[PrefillManager]
        DM[DecodeManager]
        Sched[Scheduler]
        CacheMgr[CacheManager]
        GraphRunner[GraphRunner]
    end
    
    subgraph "Batch Structure"
        Batch[Batch]
        Reqs[reqs: List[Req]]
        PaddedReqs[padded_reqs: List[Req]]
        InputIds[input_ids: Tensor]
        OutLoc[out_loc: Tensor]
    end
    
    subgraph "Engine Layer"
        Engine[Engine]
        Model[Model]
    end
    
    PM -->|"Creates"| Batch
    DM -->|"Creates"| Batch
    Batch -->|"Contains"| Reqs
    Batch -->|"Passed to"| Sched
    Sched -->|"Allocates"| CacheMgr
    Sched -->|"Pads"| GraphRunner
    GraphRunner -->|"Sets"| PaddedReqs
    Sched -->|"Creates"| InputIds
    Sched -->|"Creates"| OutLoc
    Batch -->|"Passed to"| Engine
    Engine -->|"Uses"| Model
    Model -->|"Processes"| InputIds
```

Key Connections:
- **Batch.reqs**: Real requests being processed
- **Batch.padded_reqs**: May include dummy Reqs for CUDA graph padding
- **Batch.input_ids**: Concatenated tensor (padded) ready for GPU
- **Batch.out_loc**: Maps each output position to the corresponding Req

================================================================================
TECHNICAL DECISIONS: Why Separate Prefill and Decode Batches
================================================================================

**Decision**: Separate batch types for prefill and decode phases.

**Why?**
1. **Different Batching Characteristics**:
   - **Prefill**: Sequences have variable lengths (different input sizes)
     - Requires padding to longest sequence
     - Attention pattern: full causal attention over input
   - **Decode**: All sequences are length 1 (generating one token)
     - Minimal padding needed
     - Attention pattern: incremental (only attend to new token + cached KV)

2. **Optimization Opportunities**:
   - **Prefill**: Can use chunked prefill (process input in chunks) for long sequences
   - **Decode**: Can use continuous batching (add/remove requests dynamically)
   - Different CUDA graph shapes for each phase

3. **Memory Efficiency**:
   - Prefill batches are larger (longer sequences) but processed once
   - Decode batches are smaller (length 1) but processed many times
   - Separate management allows better memory allocation

**Alternative Considered**: Single batch type for both phases.
   - **Problem**: Can't optimize for different characteristics
   - **Problem**: Padding logic becomes complex (variable vs fixed length)
   - **Chosen**: Separate types for clarity and optimization

**Why Padding?**
- GPU processes batches in parallel using matrix operations
- All sequences in a batch must be the same length for matrix ops
- Padding tokens are masked in attention (don't affect computation)
- CUDA graphs require fixed batch sizes, so padding enables graph optimization

**Real Impact**: This separation enables:
- Efficient prefill for long inputs (chunked processing)
- High-throughput decode (continuous batching)
- Better GPU utilization (optimized kernels for each phase)

**Batch.size vs Batch.padded_size**:
- `size`: Number of real requests (`len(reqs)`)
- `padded_size`: Total size including padding (`len(padded_reqs)`)
- Padding is added by `GraphRunner.pad_batch()` for CUDA graph optimization
- Only `reqs` are processed; padded entries are dummy Reqs

================================================================================
CHALLENGE
================================================================================

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
4. **NEW**: Trace through the code: How does `Scheduler._prepare_batch()` create the `input_ids` tensor?
   (Hint: Read `python/minisgl/scheduler/scheduler.py:180-201`)
5. **NEW**: Why does decode phase have all sequences at length 1? What happens to the input tokens?
   (Hint: They're already processed in prefill and cached in KV cache)

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
