"""
PUZZLE 5.2: DecodeManager - [Easy]

TASK:
Implement a simplified DecodeManager that tracks running requests in the decode phase.

GIVEN:
- Understanding of prefill vs decode phases
- Requests transition from prefill to decode after initial processing

================================================================================
DATA FLOW: Prefill to Decode Transition
================================================================================

Requests transition from PrefillManager to DecodeManager after prefill completes:

```mermaid
flowchart TD
    A[PrefillManager] -->|"schedule_next_batch"| B[Batch phase=prefill]
    B -->|"Engine.forward"| C[Process all input tokens]
    C -->|"cached_len == device_len"| D[Prefill complete]
    D -->|"decode_manager.add_reqs"| E[DecodeManager]
    E -->|"schedule_next_batch"| F[Batch phase=decode]
    F -->|"Engine.forward"| G[Generate next_token]
    G -->|"append_host + complete_one"| H[Update Req state]
    H -->|"can_decode()?"| I{More tokens?}
    I -->|"Yes"| E
    I -->|"No"| J[Remove from DecodeManager]
```

Step-by-step flow:
1. **Prefill Phase** (`python/minisgl/scheduler/scheduler.py:180-220`):
   - PrefillManager creates batch with variable-length sequences
   - Engine processes all input tokens
   - After processing: `cached_len == device_len == len(input_ids)` (all input processed)

2. **Transition to Decode** (`python/minisgl/scheduler/scheduler.py:222`):
   - `decode_manager.add_reqs(batch.reqs)` is called
   - Only Reqs where `can_decode()` returns True are added
   - `can_decode()` checks `remain_len > 0` (more tokens can be generated)

3. **Decode Phase** (`python/minisgl/scheduler/decode.py:23-26`):
   - DecodeManager creates batch with all sequences at length 1
   - Engine generates one token per request
   - Each request calls `append_host(next_token)` then `complete_one()`

4. **Continuous Decode**:
   - After each decode step, Req state is updated
   - If `can_decode()` still True, Req stays in DecodeManager
   - If `remain_len == 0` or EOS, Req is removed

**Key State Check**: `can_decode()` returns `remain_len > 0`, where:
- `remain_len = max_device_len - device_len`
- This tracks how many more tokens can be generated

================================================================================
COMPONENT CONNECTIONS: DecodeManager in Scheduler Flow
================================================================================

```mermaid
graph TB
    subgraph "Prefill Phase"
        PM[PrefillManager]
        Batch1[Batch phase=prefill]
        Engine1[Engine.forward]
    end
    
    subgraph "Transition"
        AddReqs[decode_manager.add_reqs]
        CanDecode[can_decode check]
    end
    
    subgraph "Decode Phase"
        DM[DecodeManager]
        Batch2[Batch phase=decode]
        Engine2[Engine.forward]
        Sample[Sampler.sample]
    end
    
    subgraph "Request State"
        Req[Req]
        State[Req state updates]
    end
    
    PM -->|"Creates"| Batch1
    Batch1 -->|"Processes"| Engine1
    Engine1 -->|"Completes prefill"| AddReqs
    AddReqs -->|"Checks"| CanDecode
    CanDecode -->|"Filters Reqs"| DM
    DM -->|"Creates"| Batch2
    Batch2 -->|"Generates"| Engine2
    Engine2 -->|"Samples"| Sample
    Sample -->|"Updates"| State
    State -->|"Checks"| CanDecode
    CanDecode -->|"If True"| DM
    CanDecode -->|"If False"| Remove[Remove Req]
```

Key Connections:
- **PrefillManager → DecodeManager**: Reqs transition when prefill completes
- **can_decode()**: Gatekeeper - only Reqs that can continue are added
- **Continuous Loop**: DecodeManager schedules batches until all Reqs complete

================================================================================
TECHNICAL DECISIONS: Why Separate PrefillManager and DecodeManager
================================================================================

**Decision**: Separate managers for prefill and decode phases.

**Why?**
1. **Different Batching Characteristics**:
   - **Prefill**: Variable sequence lengths (different input sizes)
     - Requires padding to longest sequence
     - Processes all input tokens at once
   - **Decode**: All sequences length 1 (generating one token)
     - Minimal padding needed
     - Processes one token at a time, repeatedly

2. **Different Scheduling Policies**:
   - **Prefill**: Token budget-based (limit tokens per batch)
     - Prevents long sequences from blocking others
     - Chunked prefill for very long sequences
   - **Decode**: Continuous batching (add/remove requests dynamically)
     - High throughput for many concurrent requests
     - Different optimization strategies

3. **Code Organization**:
   - Clear separation of concerns
   - Each manager optimized for its phase
   - Easier to maintain and optimize independently

**Alternative Considered**: Single manager for both phases.
   - **Problem**: Complex logic to handle both cases
   - **Problem**: Can't optimize each phase independently
   - **Problem**: Harder to implement different scheduling policies
   - **Chosen**: Separate managers for clarity and optimization

**Real Impact**: This separation enables:
- Efficient prefill (chunked processing for long inputs)
- High-throughput decode (continuous batching)
- Better resource utilization (different strategies per phase)

**Why Set Instead of List for running_reqs?**
- **Set**: O(1) lookup and removal (important for `remove_req()`)
- **List**: O(n) lookup and removal
- **Chosen**: Set for efficient request management

**What inflight_tokens Tells Us**:
- Total number of tokens that will be generated across all running requests
- Useful for:
  - Load balancing (distribute requests across GPUs)
  - Resource planning (estimate completion time)
  - Quality of service (ensure adequate resources per request)

**When can_decode() Returns False**:
- `remain_len == 0`: Maximum tokens reached (`device_len == max_device_len`)
- EOS token generated: Request naturally completes
- Request cancelled: User aborts request

================================================================================
CHALLENGE
================================================================================

- Implement add_reqs() to add requests that can continue decoding
- Implement remove_req() to remove finished requests
- Implement inflight_tokens property to track total tokens being generated
- Implement runnable property to check if there are requests to process

HINT:
- Reference: `python/minisgl/scheduler/decode.py:10-30` for DecodeManager
- Use a Set to track running requests
- Only add requests where can_decode() returns True
- inflight_tokens is the sum of remain_len for all running requests

QUESTIONS:
1. Why use a Set instead of a List for running_reqs?
2. What does inflight_tokens tell us about system load?
3. When would can_decode() return False?
4. **NEW**: Trace through the code: When does a Req transition from PrefillManager to DecodeManager?
   (Hint: Read `python/minisgl/scheduler/scheduler.py:222` - after Engine.forward)
5. **NEW**: Why does DecodeManager only add Reqs where `can_decode()` is True?
   (Hint: Prevents adding completed requests that can't generate more tokens)

TEST:
Run: pytest learning/puzzles/05_scheduler/test_5.2.py -v
"""

from dataclasses import dataclass, field
from typing import Set, Iterable


@dataclass
class MockReq:
    """Mock request for testing."""
    uid: int
    remain_len: int

    def can_decode(self) -> bool:
        """Check if request can continue decoding."""
        return self.remain_len > 0


@dataclass
class SimpleDecodeManager:
    """
    Manages requests in the decode phase.

    Tracks which requests are currently running and need to generate more tokens.
    """
    running_reqs: Set[MockReq] = field(default_factory=set)

    def add_reqs(self, reqs: Iterable[MockReq]) -> None:
        """
        Add requests that can continue decoding.

        Only adds requests where can_decode() returns True.

        Args:
            reqs: Iterable of requests to potentially add

        HINT: Use a generator expression with update()
              Only add req if req.can_decode() is True
        """
        # YOUR CODE HERE
        pass

    def remove_req(self, req: MockReq) -> None:
        """
        Remove a finished request.

        Uses discard() to safely remove (no error if not present).

        Args:
            req: The request to remove
        """
        # YOUR CODE HERE
        pass

    @property
    def inflight_tokens(self) -> int:
        """
        Calculate total tokens being generated across all running requests.

        Returns:
            Sum of remain_len for all running requests
        """
        # YOUR CODE HERE
        pass

    @property
    def runnable(self) -> bool:
        """
        Check if there are any requests to process.

        Returns:
            True if running_reqs is not empty
        """
        # YOUR CODE HERE
        pass


def create_decode_manager() -> SimpleDecodeManager:
    """Create a new SimpleDecodeManager."""
    return SimpleDecodeManager()


def create_mock_req(uid: int, remain_len: int) -> MockReq:
    """Create a mock request for testing."""
    return MockReq(uid=uid, remain_len=remain_len)


# Test your implementation
if __name__ == "__main__":
    manager = create_decode_manager()
    print(f"Initial runnable: {manager.runnable} (expected False)")
    print(f"Initial inflight_tokens: {manager.inflight_tokens} (expected 0)")

    # Add some requests
    req1 = create_mock_req(uid=1, remain_len=10)
    req2 = create_mock_req(uid=2, remain_len=5)
    req3 = create_mock_req(uid=3, remain_len=0)  # Can't decode

    manager.add_reqs([req1, req2, req3])
    print(f"\nAfter adding 3 reqs (one with remain_len=0):")
    print(f"  runnable: {manager.runnable} (expected True)")
    print(f"  inflight_tokens: {manager.inflight_tokens} (expected 15)")

    # Remove a request
    manager.remove_req(req1)
    print(f"\nAfter removing req1:")
    print(f"  inflight_tokens: {manager.inflight_tokens} (expected 5)")
