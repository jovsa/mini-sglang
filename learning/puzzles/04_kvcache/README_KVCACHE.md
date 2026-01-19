# KV Cache Learning Guide

This directory contains educational materials for understanding how KV cache works in mini-sglang.

## Quick Start

### Run the Example

```bash
# From the project root
python learning/puzzles/04_kvcache/kvcache_example.py
```

This will run a complete end-to-end example demonstrating:
- Request arrival and cache matching
- KV cache allocation
- Forward pass with KV cache usage
- Decode phase incremental updates
- Cache cleanup and eviction

### Read the Breakdown

For detailed explanations, see:
- **[KVCACHE_BREAKDOWN.md](KVCACHE_BREAKDOWN.md)** - Complete system explanation with code references

## Files in This Directory

### Example Files

- **`kvcache_example.py`** - Runnable end-to-end example
  - Demonstrates all phases of KV cache usage
  - Shows actual code paths and data structures
  - Includes state printing for visibility

### Documentation

- **`KVCACHE_BREAKDOWN.md`** - Detailed breakdown
  - Step-by-step explanations for each phase
  - Code references to `python/minisgl/` implementation
  - Memory layout visualizations
  - Flow diagrams

### Puzzles

- **`puzzle_4.1.py`** - Cache Handle Basics
  - Implement `SizeInfo` and `SimpleCacheHandle`
  - Learn about evictable vs protected cache

- **`puzzle_4.2.py`** - RadixTree Node Operations
  - Implement radix tree node operations
  - Learn about prefix matching

## Learning Path

### 1. Start with the Example

Run `kvcache_example.py` to see the complete flow:

```bash
python learning/puzzles/04_kvcache/kvcache_example.py
```

Observe:
- How requests are matched against cached prefixes
- How cache pages are allocated
- How the forward pass uses cache
- How cache is cleaned up

### 2. Read the Breakdown

Read `KVCACHE_BREAKDOWN.md` to understand:
- Each phase in detail
- Code references to actual implementation
- Memory layouts and data structures
- Design decisions and trade-offs

### 3. Study the Puzzles

Work through the puzzles to implement key components:
- `puzzle_4.1.py` - Cache handle basics
- `puzzle_4.2.py` - Radix tree operations

### 4. Explore the Code

Use the code references in `KVCACHE_BREAKDOWN.md` to explore:
- `python/minisgl/kvcache/` - KV cache implementation
- `python/minisgl/scheduler/` - Request scheduling with cache
- `python/minisgl/engine/` - Forward pass execution

## Key Concepts

### Cache Matching

Requests are matched against cached prefixes using a radix tree:
- **New request**: `cached_len = 0`, no prefix match
- **Prefix match**: `cached_len > 0`, reuse cached tokens

**Code**: `python/minisgl/kvcache/radix_manager.py:116-127`

### Cache Allocation

Cache pages are allocated from:
- **Free pool**: Available pages
- **Eviction**: When cache is full, evict unused entries

**Code**: `python/minisgl/scheduler/cache.py:39-52`

### Forward Pass

Only new tokens are processed:
- `extend_len = device_len - cached_len`
- KV cache is stored using page indices
- `cached_len` tracks cached tokens

**Code**: `python/minisgl/engine/engine.py:188-203`

### Cache Cleanup

When requests complete:
- Prefix is inserted into radix tree
- Handle is unlocked (protected → evictable)
- Unused pages are freed

**Code**: `python/minisgl/scheduler/cache.py:54-62`

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Request Layer                         │
│  UserMsg → PendingReq → Req (with cache_handle)        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Cache Management                        │
│  CacheManager → RadixCacheManager → RadixTree           │
│  - Match prefixes                                        │
│  - Allocate pages                                        │
│  - Manage eviction                                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    Storage Layer                         │
│  MHAKVCache (KV storage)                                │
│  PageTable (token → page mapping)                       │
│  TokenPool (token IDs)                                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    Engine Layer                          │
│  Engine → Model → Attention → KVCache                   │
│  - Forward pass                                          │
│  - Store KV cache                                       │
│  - Read from cache                                       │
└─────────────────────────────────────────────────────────┘
```

## Code References

### Main Components

- **KV Cache**: `python/minisgl/kvcache/mha_pool.py`
- **Cache Manager**: `python/minisgl/scheduler/cache.py`
- **Radix Manager**: `python/minisgl/kvcache/radix_manager.py`
- **Prefill Manager**: `python/minisgl/scheduler/prefill.py`
- **Engine**: `python/minisgl/engine/engine.py`

### Key Functions

- **Cache Matching**: `python/minisgl/kvcache/radix_manager.py:116-127`
- **Cache Allocation**: `python/minisgl/scheduler/cache.py:39-52`
- **KV Storage**: `python/minisgl/kvcache/mha_pool.py:56-67`
- **Forward Pass**: `python/minisgl/engine/engine.py:188-203`

For complete code references, see `KVCACHE_BREAKDOWN.md`.

## Next Steps

1. **Run the example** to see the complete flow
2. **Read the breakdown** to understand each phase
3. **Work through puzzles** to implement components
4. **Explore the code** using provided references
5. **Experiment** by modifying the example

## Questions to Explore

1. How does prefix matching work in the radix tree?
2. What happens when cache is full?
3. How are cache entries protected from eviction?
4. How does the page table map tokens to cache pages?
5. What is the difference between `cached_len` and `device_len`?

Answers to these questions can be found in `KVCACHE_BREAKDOWN.md`.

## Related Files

- **Puzzles**: `puzzle_4.1.py`, `puzzle_4.2.py`
- **Tests**: `test_4.1.py`, `test_4.2.py`
- **Implementation**: `python/minisgl/kvcache/`, `python/minisgl/scheduler/`

## Summary

The KV cache system enables efficient LLM inference by:
- **Caching** computed key/value pairs
- **Sharing** prefixes across requests
- **Managing** memory through eviction
- **Protecting** active requests

Start with the example, then dive into the breakdown for detailed explanations!
