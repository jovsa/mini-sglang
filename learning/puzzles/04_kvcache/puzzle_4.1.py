"""
PUZZLE 4.1: Cache Handle Basics - [Easy]

TASK:
Implement SizeInfo and SimpleCacheHandle to understand KV cache management basics.

GIVEN:
- Understanding of how KV caches track memory usage
- Two size categories: evictable (can be freed) and protected (in use)

================================================================================
DATA FLOW: How Cache Handles Connect Reqs to Cache Memory
================================================================================

Cache handles are the bridge between requests and physical KV cache memory:

```mermaid
flowchart TD
    A[UserMsg arrives] -->|"Scheduler"| B[PrefillManager.add_one_req]
    B -->|"Calls"| C[CacheManager.match_req]
    C -->|"Finds prefix or allocates new"| D[CacheHandle]
    D -->|"Contains cached_len"| E[Create Req]
    E -->|"Stores cache_handle"| F[Req.cache_handle]
    F -->|"Used during forward"| G[KV Cache Memory]
    G -->|"Stores key/value"| H[Attention Computation]
    H -->|"Updates cached_len"| F
```

Step-by-step flow:
1. **CacheManager.match_req()** (`python/minisgl/kvcache/radix_manager.py` or `naive_manager.py`):
   - Tries to find matching prefix in radix tree (if using radix cache)
   - Returns `CacheHandle` with `cached_len` (0 for new, >0 for prefix match)
   - CacheHandle points to physical memory pages in KV cache

2. **Req Creation** (`python/minisgl/scheduler/prefill.py:80-88`):
   - Req stores `cache_handle` reference
   - `cached_len` from handle indicates how many tokens are already cached
   - This enables prefix sharing (multiple requests can share same cache)

3. **During Forward Pass** (`python/minisgl/engine/engine.py`):
   - Attention layers use `cache_handle` to access KV cache memory
   - Only processes tokens from `cached_len` to `device_len` (new tokens)
   - After processing, `cached_len` is updated to `device_len`

4. **Cache Eviction**:
   - When request completes, cache entry becomes evictable
   - `SizeInfo` tracks evictable vs protected sizes
   - Evictable entries can be freed when cache is full

================================================================================
COMPONENT CONNECTIONS: Cache Handle to Physical Memory
================================================================================

```mermaid
graph TB
    subgraph "Request Layer"
        Req[Req]
        CH[Req.cache_handle]
    end
    
    subgraph "Cache Management"
        CacheMgr[CacheManager]
        SizeInfo[SizeInfo]
        RadixTree[RadixTree]
    end
    
    subgraph "Physical Memory"
        KVCache[KV Cache Memory]
        Pages[Cache Pages]
        PageTable[Page Table]
    end
    
    subgraph "Engine"
        Attention[Attention Layers]
    end
    
    Req -->|"Contains"| CH
    CH -->|"Points to"| Pages
    CacheMgr -->|"Manages"| RadixTree
    RadixTree -->|"Finds prefix"| CH
    CacheMgr -->|"Tracks"| SizeInfo
    SizeInfo -->|"evictable_size"| KVCache
    SizeInfo -->|"protected_size"| KVCache
    CH -->|"Used by"| Attention
    Attention -->|"Reads/Writes"| Pages
    Pages -->|"Mapped by"| PageTable
```

Key Connections:
- **CacheHandle**: Abstract reference to cache memory (hides implementation details)
- **SizeInfo**: Tracks memory usage by category (evictable vs protected)
- **Physical Memory**: Actual GPU memory storing key/value tensors
- **Page Table**: Maps cache handles to physical memory pages

================================================================================
TECHNICAL DECISIONS: Why Evictable vs Protected Size
================================================================================

**Decision**: Track cache size in two categories: evictable and protected.

**Why?**
1. **Request Lifecycle**:
   - **Protected**: Cache entries currently in use by active requests
     - Cannot be evicted (would break active requests)
     - Tracked per request's `cache_handle`
   - **Evictable**: Cache entries from completed requests
     - Can be freed when cache is full
     - Enables cache reuse and memory management

2. **Memory Management**:
   - When cache is full, only evictable entries can be freed
   - Protected entries must remain until request completes
   - This prevents corruption of active requests

3. **Prefix Sharing**:
   - Multiple requests can share the same cache entry (radix tree)
   - Entry is protected as long as ANY request uses it
   - Entry becomes evictable only when ALL requests complete

**Alternative Considered**: Single size counter.
   - **Problem**: Can't distinguish between in-use and freeable entries
   - **Problem**: Risk of evicting active cache entries
   - **Chosen**: Two-category approach for safety and correctness

**Real Impact**: This design enables:
- Safe cache eviction (only free unused entries)
- Prefix sharing (multiple requests share cache)
- Memory efficiency (reuse cache across requests)

**What cached_len Represents**:
- `cached_len`: Number of tokens already processed and stored in KV cache
- For new requests: `cached_len = 0` (no tokens cached yet)
- For prefix matches: `cached_len > 0` (some tokens already cached)
- During processing: `cached_len` increases as tokens are processed

**When Cache Entry Moves from Protected to Evictable**:
- When the last request using that cache entry completes
- `CacheManager` decrements reference count
- When ref_count reaches 0, entry becomes evictable
- Can be freed during next cache allocation if needed

================================================================================
CHALLENGE
================================================================================

- Implement SizeInfo with evictable_size and protected_size
- Add a total_size property that returns the sum
- Implement SimpleCacheHandle with cached_len tracking

HINT:
- Reference: `python/minisgl/kvcache/base.py:46-57` for SizeInfo and BaseCacheHandle
- SizeInfo is a NamedTuple with two fields
- total_size is a computed property

QUESTIONS:
1. Why do we track evictable vs protected size separately?
2. What does cached_len represent in a cache handle?
3. When would a cache entry move from protected to evictable?
4. **NEW**: Trace through the code: How does `CacheManager.match_req()` set `cached_len`?
   (Hint: Read `python/minisgl/kvcache/radix_manager.py` - prefix matching)
5. **NEW**: What happens if we try to evict a protected cache entry?
   (Hint: This would break active requests - why the separation is critical)

TEST:
Run: pytest learning/puzzles/04_kvcache/test_4.1.py -v
"""

from typing import NamedTuple
from dataclasses import dataclass


class SizeInfo(NamedTuple):
    """
    Tracks the size of cache entries by category.

    Attributes:
        evictable_size: Number of cache slots that can be freed
        protected_size: Number of cache slots currently in use (locked)

    TODO: Implement the total_size property
    """
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        """Return the total cache size (evictable + protected)."""
        # YOUR CODE HERE
        pass


@dataclass(frozen=True)
class SimpleCacheHandle:
    """
    A simple cache handle that tracks the cached length.

    This represents a reference to cached KV entries.
    The cached_len indicates how many tokens are already cached.

    TODO: Add the cached_len field
    """
    # YOUR CODE HERE - add the cached_len field
    pass


def create_size_info(evictable: int, protected: int) -> SizeInfo:
    """
    Create a SizeInfo object.

    Args:
        evictable: Number of evictable cache slots
        protected: Number of protected cache slots

    Returns:
        SizeInfo object
    """
    # TODO: Create and return SizeInfo
    # YOUR CODE HERE
    pass


def create_cache_handle(cached_len: int) -> SimpleCacheHandle:
    """
    Create a SimpleCacheHandle.

    Args:
        cached_len: Number of tokens already cached

    Returns:
        SimpleCacheHandle object
    """
    # TODO: Create and return SimpleCacheHandle
    # YOUR CODE HERE
    pass


# Test your implementation
if __name__ == "__main__":
    # Test SizeInfo
    size_info = create_size_info(evictable=100, protected=50)
    print(f"SizeInfo: evictable={size_info.evictable_size}, protected={size_info.protected_size}")
    print(f"Total size: {size_info.total_size} (expected 150)")

    # Test SimpleCacheHandle
    handle = create_cache_handle(cached_len=10)
    print(f"CacheHandle: cached_len={handle.cached_len} (expected 10)")
