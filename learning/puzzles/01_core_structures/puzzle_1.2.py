"""
PUZZLE 1.2: Fix the Req Creation - [Medium]

TASK:
Fix bugs in the Req creation code. There are intentional bugs that need to be fixed.

GIVEN:
- Partial Req implementation with bugs
- Understanding of what Req should do

================================================================================
DATA FLOW: From UserMsg to Req to PrefillManager
================================================================================

Req is created when the Scheduler receives a UserMsg. Here's the complete flow:

```mermaid
flowchart TD
    A[UserMsg arrives] -->|"ZMQ message"| B[Scheduler._process_one_msg]
    B -->|"Validate input_len"| C{Input too long?}
    C -->|"Yes"| D[Drop request]
    C -->|"No"| E[Adjust max_tokens]
    E -->|"Call"| F[PrefillManager.add_one_req]
    F -->|"Allocate cache"| G[CacheManager.match_req]
    G -->|"Allocate slot"| H[TableManager.allocate]
    H -->|"Create"| I[Req.__post_init__]
    I -->|"Validate"| J[Req created]
    J -->|"Add to"| K[PrefillManager.runnable_reqs]
```

Step-by-step flow:
1. **Scheduler receives UserMsg** (`python/minisgl/scheduler/scheduler.py:161-175`):
   - Validates input length against `engine.max_seq_len`
   - Adjusts `max_tokens` if needed
   - Calls `prefill_manager.add_one_req(msg)`

2. **PrefillManager.add_one_req()** (`python/minisgl/scheduler/prefill.py:63-100`):
   - Allocates cache handle via `CacheManager.match_req()` (may find prefix match)
   - Allocates table slot via `TableManager.allocate()`
   - Creates Req with `cached_len` from cache handle (0 for new, >0 for prefix match)
   - Sets `device_len = len(input_ids)` and `max_device_len = len(input_ids) + output_len`

3. **Req.__post_init__()** (`python/minisgl/core.py:37-41`):
   - Validates `input_ids.is_cpu` (critical for ZMQ serialization)
   - Validates state invariants: `0 <= cached_len < device_len <= max_device_len`
   - Initializes `device_len` and `max_device_len` properties

**Key State Variables:**
- `cached_len`: Number of tokens already processed and cached (from KV cache)
- `device_len`: Current position in sequence (length of input_ids processed so far)
- `max_device_len`: Maximum sequence length (input + max output tokens)

================================================================================
COMPONENT CONNECTIONS: How Req Connects to System Components
================================================================================

```mermaid
graph TB
    subgraph "Message Layer"
        UMsg[UserMsg]
    end
    
    subgraph "Scheduler Components"
        Sched[Scheduler]
        PrefillMgr[PrefillManager]
        CacheMgr[CacheManager]
        TableMgr[TableManager]
    end
    
    subgraph "Request State"
        Req[Req]
    end
    
    subgraph "Storage"
        TokenPool[token_pool]
        PageTable[page_table]
        KVCache[KV Cache Memory]
    end
    
    UMsg -->|"input_ids, sampling_params"| Sched
    Sched -->|"add_one_req"| PrefillMgr
    PrefillMgr -->|"match_req"| CacheMgr
    PrefillMgr -->|"allocate"| TableMgr
    CacheMgr -->|"cache_handle"| Req
    TableMgr -->|"table_idx"| Req
    Req -->|"table_idx"| TokenPool
    Req -->|"table_idx"| PageTable
    Req -->|"cache_handle"| KVCache
    Req -->|"Stored in"| PrefillMgr
```

Key Connections:
- **TableManager**: Allocates `table_idx` which maps Req to fixed-size slots in `token_pool` and `page_table`
- **CacheManager**: Provides `cache_handle` which connects Req to KV cache memory pages
- **PrefillManager**: Stores Req in `runnable_reqs` until it can be batched

================================================================================
TECHNICAL DECISIONS: Why input_ids Must Be a CPU Tensor
================================================================================

**Decision**: Req.input_ids must be a CPU tensor, not a GPU tensor.

**Why?**
1. **ZMQ Serialization**: UserMsg crosses process boundaries via ZeroMQ. GPU tensors cannot be 
   serialized directly - they must be moved to CPU first. See `python/minisgl/message/backend.py:35`
   where `input_ids` is explicitly a CPU tensor.

2. **Process Isolation**: API Server, Tokenizer, and Scheduler run in separate processes. GPU memory
   is not shared across processes, so tensors must be on CPU to cross boundaries.

3. **Lazy GPU Transfer**: The tensor is moved to GPU only when needed (during Batch creation in
   `Scheduler._prepare_batch()`). This defers GPU memory allocation until batching, allowing better
   memory management.

4. **Validation Safety**: The assertion `assert self.input_ids.is_cpu` in `__post_init__` ensures
   this invariant is maintained. If a GPU tensor somehow reaches Req creation, it will fail fast.

**Alternative Considered**: Keep tensors on GPU throughout.
   - **Problem**: Cannot serialize GPU tensors for ZMQ messages
   - **Problem**: GPU memory allocated too early (before batching decisions)
   - **Chosen**: CPU-first approach for flexibility and correctness

**Real Impact**: This design allows the scheduler to queue many requests in CPU memory, then batch
   them efficiently before moving to GPU. This is critical for high-throughput serving.

**Why the State Invariants?**
- `cached_len < device_len`: You can't have cached more tokens than you've processed
- `device_len <= max_device_len`: You can't exceed the maximum sequence length
- These invariants ensure the request state is always valid and prevent buffer overflows

================================================================================
CHALLENGE
================================================================================

- Fix wrong cached_len value
- Add missing validation
- Understand the relationship between cached_len, device_len, and max_device_len

HINT:
- Reference: `python/minisgl/core.py:37-41` for __post_init__ validation
- The validation assertion is: `0 <= cached_len < device_len <= max_device_len`
- device_len = len(input_ids)
- max_device_len = len(input_ids) + output_len

QUESTIONS:
1. What happens if cached_len >= device_len? Why is this invalid?
2. What's the difference between device_len and max_device_len?
3. Why does input_ids need to be a CPU tensor?
4. **NEW**: Trace through the code: What happens if `input_ids` is a GPU tensor when creating Req?
   (Hint: Check `__post_init__` validation)
5. **NEW**: How does `cached_len` get set when a request matches a prefix in the radix tree?
   (Hint: Read `python/minisgl/scheduler/prefill.py:42-43`)

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
        assert self.input_ids.is_cpu

        # TODO: Fix bug 2 - Set device_len correctly (should be length of input_ids)
        # YOUR CODE HERE
        self.device_len = len(self.input_ids)

        # TODO: Fix bug 3 - Calculate max_device_len correctly
        # max_device_len = input_length + output_length
        # YOUR CODE HERE
        self.max_device_len = len(self.input_ids) + self.output_len # BUG: This is wrong!

        # TODO: Fix bug 4 - Validate that cached_len < device_len
        # YOUR CODE HERE
        # When device_len == 0 (empty input), cached_len must also be 0
        # When device_len > 0, cached_len must be < device_len
        assert 0 <= self.cached_len and (self.device_len == 0 or self.cached_len < self.device_len)

        # TODO: Fix bug 5 - Validate that device_len <= max_device_len
        # YOUR CODE HERE
        assert self.device_len <= self.max_device_len

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
    import importlib.util
    from pathlib import Path
    puzzle_dir = Path(__file__).parent
    sys.path.insert(0, str(puzzle_dir))

    # Import puzzle file with dot in name using importlib
    puzzle_1_1_file = puzzle_dir / "puzzle_1.1.py"
    spec = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
    puzzle_1_1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(puzzle_1_1)
    create_sampling_params = puzzle_1_1.create_sampling_params

    # TODO: Ensure input_ids is on CPU if it's not already
    # YOUR CODE HERE
    if not input_ids.is_cpu:
        input_ids = input_ids.cpu()

    sampling_params = create_sampling_params(max_tokens=output_len)
    cache_handle = MockCacheHandle(cached_len=cached_len)

    # TODO: Create and return Req object
    # YOUR CODE HERE
    return Req(input_ids=input_ids, output_len=output_len, uid=uid, cached_len=cached_len, table_idx=table_idx, sampling_params=sampling_params, cache_handle=cache_handle)


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
