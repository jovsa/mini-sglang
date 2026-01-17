"""
PUZZLE 4.1: Cache Handle Basics - [Easy]

TASK:
Implement SizeInfo and SimpleCacheHandle to understand KV cache management basics.

GIVEN:
- Understanding of how KV caches track memory usage
- Two size categories: evictable (can be freed) and protected (in use)

CHALLENGE:
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
