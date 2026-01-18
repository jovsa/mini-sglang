"""
PUZZLE 5.1: TableManager Basics - [Easy]

TASK:
Implement a simplified TableManager that manages request slots.
The TableManager tracks which slots are free and allocates them to requests.

GIVEN:
- Understanding of resource allocation
- Fixed number of maximum concurrent requests

================================================================================
DATA FLOW: TableManager Slot Allocation
================================================================================

TableManager allocates fixed-size slots for requests. Here's the flow:

```mermaid
flowchart TD
    A[UserMsg arrives] -->|"Scheduler"| B[PrefillManager.add_one_req]
    B -->|"Allocate resources"| C[TableManager.allocate]
    C -->|"Get free slot"| D[table_idx = 0]
    D -->|"Create Req"| E[Req.table_idx = 0]
    E -->|"Map to storage"| F[token_pool[0]]
    E -->|"Map to storage"| G[page_table[0]]
    F -->|"Store token IDs"| H[GPU Memory]
    G -->|"Store page indices"| H
    I[Request completes] -->|"Free slot"| J[TableManager.free]
    J -->|"Return slot 0"| C
```

Step-by-step flow:
1. **Scheduler receives UserMsg** (`python/minisgl/scheduler/scheduler.py:161-175`):
   - Calls `prefill_manager.add_one_req(msg)`

2. **PrefillManager.add_one_req()** (`python/minisgl/scheduler/prefill.py:38-61`):
   - Allocates cache handle via `CacheManager.match_req()`
   - Allocates table slot via `TableManager.allocate()`
   - Returns `table_idx` (0 to max_running_reqs-1)

3. **Req Creation** (`python/minisgl/scheduler/prefill.py:80-88`):
   - Req stores `table_idx`
   - This index maps to fixed-size slots in:
     - `token_pool[table_idx]`: Stores token IDs for this request
     - `page_table[table_idx]`: Maps token positions to KV cache pages

4. **Request Processing**:
   - Token IDs are written to `token_pool[table_idx]`
   - Page indices are written to `page_table[table_idx]`
   - These are GPU tensors with fixed size (max_seq_len)

5. **Request Completion**:
   - When request finishes, `TableManager.free(table_idx)` is called
   - Slot becomes available for new requests

================================================================================
COMPONENT CONNECTIONS: TableManager to Storage
================================================================================

```mermaid
graph TB
    subgraph "Scheduler"
        Sched[Scheduler]
        PrefillMgr[PrefillManager]
        TableMgr[TableManager]
    end
    
    subgraph "Request"
        Req[Req]
        TableIdx[Req.table_idx]
    end
    
    subgraph "Storage Arrays"
        TokenPool[token_pool]
        PageTable[page_table]
        Slot0[Slot 0]
        Slot1[Slot 1]
        SlotN[Slot N]
    end
    
    subgraph "GPU Memory"
        GPUMem[GPU Memory]
    end
    
    Sched -->|"Manages"| PrefillMgr
    PrefillMgr -->|"Allocates"| TableMgr
    TableMgr -->|"Returns"| TableIdx
    TableIdx -->|"Maps to"| Slot0
    TableIdx -->|"Maps to"| Slot1
    Req -->|"Contains"| TableIdx
    Slot0 -->|"Part of"| TokenPool
    Slot1 -->|"Part of"| TokenPool
    SlotN -->|"Part of"| TokenPool
    Slot0 -->|"Part of"| PageTable
    Slot1 -->|"Part of"| PageTable
    SlotN -->|"Part of"| PageTable
    TokenPool -->|"Stored in"| GPUMem
    PageTable -->|"Stored in"| GPUMem
```

Key Connections:
- **TableManager**: Manages allocation of fixed-size slots
- **token_pool**: 2D tensor `[max_running_reqs, max_seq_len]` storing token IDs
- **page_table**: 2D tensor `[max_running_reqs, max_seq_len]` storing KV cache page indices
- **Req.table_idx**: Index into both arrays (maps request to its storage slot)

================================================================================
TECHNICAL DECISIONS: Why Fixed-Size Table
================================================================================

**Decision**: Use fixed-size table (max_running_reqs) instead of dynamic allocation.

**Why?**
1. **GPU Memory Constraints**:
   - GPU memory must be allocated upfront (can't dynamically grow)
   - `token_pool` and `page_table` are pre-allocated GPU tensors
   - Fixed size enables efficient memory management

2. **Performance**:
   - Fixed-size arrays enable efficient indexing (O(1) access)
   - No memory allocation/deallocation overhead during request processing
   - Predictable memory usage

3. **CUDA Graph Optimization**:
   - CUDA graphs require fixed shapes
   - Fixed table size enables graph capture and optimization
   - Dynamic allocation would break graph optimization

**Alternative Considered**: Dynamic allocation per request.
   - **Problem**: GPU memory can't be dynamically allocated efficiently
   - **Problem**: Breaks CUDA graph optimization
   - **Problem**: Memory fragmentation issues
   - **Chosen**: Fixed-size table for performance and simplicity

**Real Impact**: This design enables:
- Predictable memory usage (know max memory upfront)
- Efficient GPU operations (fixed shapes)
- CUDA graph optimization (fixed batch sizes)

**Why List Instead of Set for Free Slots?**
- **List**: Simple, efficient for stack-like operations (pop/append)
- **Set**: More overhead, not needed since we just need LIFO behavior
- **Chosen**: List for simplicity and performance

**What Happens When No Slots Available?**
- `TableManager.allocate()` raises `IndexError`
- Scheduler must wait or reject request
- This enforces max_running_reqs limit (prevents memory exhaustion)

**Why Limit max_running_reqs?**
1. **Memory**: Each slot uses `max_seq_len * sizeof(int32)` bytes
2. **GPU Constraints**: Limited GPU memory
3. **Performance**: Too many concurrent requests degrade performance
4. **Quality of Service**: Limits ensure each request gets adequate resources

================================================================================
CHALLENGE
================================================================================

- Implement available_size property
- Implement allocate() to get a free slot
- Implement free() to return a slot

HINT:
- Reference: `python/minisgl/scheduler/table.py:4-19` for TableManager
- Use a list to track free slots
- allocate() pops from the free list
- free() appends back to the free list

QUESTIONS:
1. Why do we need to limit max_running_reqs?
2. What happens if we try to allocate when no slots are free?
3. Why use a list instead of a set for free slots?
4. **NEW**: Trace through the code: How does `table_idx` map to `token_pool` and `page_table`?
   (Hint: Read `python/minisgl/scheduler/scheduler.py:187-194` - _make_2d_indices)
5. **NEW**: Why are `token_pool` and `page_table` pre-allocated GPU tensors?
   (Hint: GPU memory management and CUDA graph requirements)

TEST:
Run: pytest learning/puzzles/05_scheduler/test_5.1.py -v
"""

from typing import List


class SimpleTableManager:
    """
    Manages request slots for the scheduler.

    Each request needs a slot (table index) to track its KV cache.
    This manager tracks which slots are available.
    """

    def __init__(self, max_running_reqs: int) -> None:
        """
        Initialize the table manager.

        Args:
            max_running_reqs: Maximum number of concurrent requests

        TODO: Initialize _free_slots as a list of slot indices [0, 1, 2, ..., max_running_reqs-1]
        """
        self._max_running_reqs = max_running_reqs
        # YOUR CODE HERE - initialize _free_slots
        self._free_slots: List[int] = []

    @property
    def available_size(self) -> int:
        """
        Return the number of available slots.

        Returns:
            Number of free slots that can be allocated
        """
        # YOUR CODE HERE
        pass

    def allocate(self) -> int:
        """
        Allocate a slot for a new request.

        Returns:
            The allocated slot index

        Raises:
            IndexError: If no slots are available

        HINT: Use list.pop() to get and remove the last element
        """
        # YOUR CODE HERE
        pass

    def free(self, slot: int) -> None:
        """
        Free a slot when a request completes.

        Args:
            slot: The slot index to free

        HINT: Use list.append() to add the slot back
        """
        # YOUR CODE HERE
        pass


def create_table_manager(max_reqs: int) -> SimpleTableManager:
    """
    Create a SimpleTableManager.

    Args:
        max_reqs: Maximum number of concurrent requests

    Returns:
        SimpleTableManager object
    """
    return SimpleTableManager(max_reqs)


# Test your implementation
if __name__ == "__main__":
    # Create a manager with 5 slots
    manager = create_table_manager(5)
    print(f"Initial available: {manager.available_size} (expected 5)")

    # Allocate some slots
    slot1 = manager.allocate()
    slot2 = manager.allocate()
    print(f"Allocated slots: {slot1}, {slot2}")
    print(f"Available after allocate: {manager.available_size} (expected 3)")

    # Free a slot
    manager.free(slot1)
    print(f"Available after free: {manager.available_size} (expected 4)")
