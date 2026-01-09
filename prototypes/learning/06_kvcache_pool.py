#!/usr/bin/env python3
"""
Learning Prototype 6: KV Cache Pool
====================================

Demonstrates Mini-SGLang's KV cache storage system:
- BaseKVCache: Interface for KV cache storage
- MHAKVCache: Multi-Head Attention KV cache implementation
- KVCacheLayout: Memory layout options
- Page-based storage for efficient memory management

Uses real code from minisgl.kvcache.
Requires CUDA GPU.
"""

import torch

# Import REAL Mini-SGLang components
from minisgl.kvcache.base import BaseKVCache, KVCacheLayout, BaseCacheManager, SizeInfo
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.distributed.info import set_tp_info

# Reset TP info for demo
import minisgl.distributed.info as info_module
info_module._TP_INFO = None
set_tp_info(rank=0, size=1)


def demo_kvcache_interface():
    """Demonstrate the BaseKVCache interface."""
    print("=" * 60)
    print("1. BaseKVCache Interface")
    print("=" * 60)

    print("""
BaseKVCache is the interface for KV cache storage:

class BaseKVCache(ABC):
    @abstractmethod
    def k_cache(self, layer_id: int) -> Tensor:
        '''Get K cache tensor for a specific layer'''
        ...

    @abstractmethod
    def v_cache(self, layer_id: int) -> Tensor:
        '''Get V cache tensor for a specific layer'''
        ...

    @abstractmethod
    def store_kv(self, k, v, out_loc, layer_id) -> None:
        '''Store K and V tensors at specified locations'''
        ...

    @property
    def device: torch.device
    @property
    def dtype: torch.dtype
    @property
    def num_layers: int
""")


def demo_kvcache_layout():
    """Demonstrate KVCacheLayout options."""
    print("\n" + "=" * 60)
    print("2. KVCacheLayout - Memory Organization")
    print("=" * 60)

    print("""
Two layout options for KV cache memory:

1. LayerFirst (default):
   Shape: [2, num_layers, num_pages, num_kv_heads, head_dim]
   
   Memory organization:
   ├── K cache
   │   ├── Layer 0: [num_pages, num_kv_heads, head_dim]
   │   ├── Layer 1: [num_pages, num_kv_heads, head_dim]
   │   └── ...
   └── V cache
       ├── Layer 0: [num_pages, num_kv_heads, head_dim]
       ├── Layer 1: [num_pages, num_kv_heads, head_dim]
       └── ...

   Pros: Better cache locality for layer-by-layer processing

2. PageFirst:
   Shape: [2, num_pages, num_layers, num_kv_heads, head_dim]
   
   Memory organization:
   ├── K cache
   │   ├── Page 0: [num_layers, num_kv_heads, head_dim]
   │   ├── Page 1: [num_layers, num_kv_heads, head_dim]
   │   └── ...
   └── V cache
       └── ...

   Pros: Better for page-level operations (eviction, copy)
""")

    print("\nAvailable layouts:")
    for layout in KVCacheLayout:
        print(f"  KVCacheLayout.{layout.name}")


def demo_mha_kvcache():
    """Demonstrate MHAKVCache creation and usage."""
    print("\n" + "=" * 60)
    print("3. MHAKVCache - Creating the Cache Pool")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nSkipping GPU demo - CUDA not available")
        return None

    device = torch.device("cuda:0")

    # Create a small KV cache for demo
    kvcache = MHAKVCache(
        num_kv_heads=8,      # GQA with 8 KV heads
        num_layers=4,        # 4 transformer layers
        head_dim=64,         # 64-dim per head
        num_pages=100,       # 100 pages (tokens)
        dtype=torch.float16,
        kv_layout=KVCacheLayout.LayerFirst,
        device=device,
    )

    print(f"\nMHAKVCache created:")
    print(f"  device: {kvcache.device}")
    print(f"  dtype: {kvcache.dtype}")
    print(f"  num_layers: {kvcache.num_layers}")

    # Access K and V caches for a specific layer
    k_cache_0 = kvcache.k_cache(0)
    v_cache_0 = kvcache.v_cache(0)

    print(f"\nLayer 0 cache shapes:")
    print(f"  K cache: {k_cache_0.shape}")
    print(f"  V cache: {v_cache_0.shape}")

    return kvcache


def demo_memory_calculation():
    """Show how to calculate KV cache memory requirements."""
    print("\n" + "=" * 60)
    print("4. KV Cache Memory Calculation")
    print("=" * 60)

    # Example model config
    num_layers = 32
    num_kv_heads = 8
    head_dim = 128
    dtype_bytes = 2  # float16

    # Memory per token (page)
    mem_per_token = (
        2  # K + V
        * num_layers
        * num_kv_heads
        * head_dim
        * dtype_bytes
    )

    print(f"\nExample: Llama-style model")
    print(f"  num_layers: {num_layers}")
    print(f"  num_kv_heads: {num_kv_heads} (GQA)")
    print(f"  head_dim: {head_dim}")
    print(f"  dtype: float16 ({dtype_bytes} bytes)")

    print(f"\nMemory per token:")
    print(f"  {mem_per_token:,} bytes = {mem_per_token / 1024:.2f} KB")

    # For different cache sizes
    print(f"\nTotal KV cache memory:")
    for num_tokens in [1000, 10000, 100000]:
        total_mem = num_tokens * mem_per_token
        print(f"  {num_tokens:,} tokens: {total_mem / (1024**3):.2f} GB")


def demo_page_based_storage():
    """Explain page-based KV cache storage."""
    print("\n" + "=" * 60)
    print("5. Page-Based Storage")
    print("=" * 60)

    print("""
Mini-SGLang uses page-based KV cache storage:

1. Each "page" stores K and V for ONE token across ALL layers
   - Page size = 1 token (configurable in some systems)

2. Page table maps logical positions to physical pages:

   Request 0 (5 tokens):
   Logical:  [0, 1, 2, 3, 4]
   Physical: [page_12, page_5, page_8, page_3, page_20]

   Request 1 (3 tokens):
   Logical:  [0, 1, 2]
   Physical: [page_7, page_15, page_1]

3. Benefits of paging:
   - Non-contiguous allocation (like virtual memory)
   - Efficient memory utilization
   - Easy eviction (just free pages)
   - Prefix sharing (multiple requests share pages)

Page Table Structure:
┌─────────────────────────────────────────────────────────┐
│ page_table[req_idx, position] = physical_page_idx      │
├─────────────────────────────────────────────────────────┤
│ Request 0: [12,  5,  8,  3, 20,  0,  0, ...]          │
│ Request 1: [ 7, 15,  1,  0,  0,  0,  0, ...]          │
│ Request 2: [22, 11, 19, 30,  0,  0,  0, ...]          │
└─────────────────────────────────────────────────────────┘
""")


def demo_store_kv(kvcache):
    """Demonstrate storing K and V to cache."""
    print("\n" + "=" * 60)
    print("6. Storing K and V to Cache")
    print("=" * 60)

    if kvcache is None:
        print("\nSkipping - no KV cache available")
        return

    print("""
store_kv(k, v, out_loc, layer_id):
  - k: Key tensor [num_tokens, num_kv_heads, head_dim]
  - v: Value tensor [num_tokens, num_kv_heads, head_dim]
  - out_loc: Physical page indices [num_tokens]
  - layer_id: Which layer to store to

The kernel scatters K and V to non-contiguous locations:
  for i, page_idx in enumerate(out_loc):
      k_cache[page_idx] = k[i]
      v_cache[page_idx] = v[i]
""")

    # Note: We can't easily demo store_kv without the kernel
    # But we can show the concept
    num_tokens = 5
    num_kv_heads = 8
    head_dim = 64

    print(f"\nExample store operation:")
    print(f"  Storing {num_tokens} tokens to layer 0")
    print(f"  K shape: [{num_tokens}, {num_kv_heads}, {head_dim}]")
    print(f"  V shape: [{num_tokens}, {num_kv_heads}, {head_dim}]")
    print(f"  out_loc: [page_5, page_12, page_3, page_8, page_20]")


def demo_cache_manager_interface():
    """Show BaseCacheManager interface (manages page allocation)."""
    print("\n" + "=" * 60)
    print("7. BaseCacheManager - Page Allocation")
    print("=" * 60)

    print("""
BaseCacheManager manages which pages are used/free:

class BaseCacheManager(ABC):
    def match_prefix(input_ids) -> (handle, indices):
        '''Find cached prefix, return page indices'''

    def lock_handle(handle, unlock=False):
        '''Protect pages from eviction while in use'''

    def insert_prefix(input_ids, indices) -> int:
        '''Add new prefix to cache'''

    def evict(size) -> Tensor:
        '''Free pages using LRU policy'''

    @property
    def size_info -> SizeInfo:
        '''evictable_size + protected_size'''

Two implementations:
1. NaiveCacheManager: No prefix sharing, simple allocation
2. RadixCacheManager: Radix tree for prefix sharing (demo in 02)
""")

    # Show SizeInfo
    info = SizeInfo(evictable_size=50, protected_size=30)
    print(f"\nSizeInfo example:")
    print(f"  evictable_size: {info.evictable_size} (can be freed)")
    print(f"  protected_size: {info.protected_size} (in use)")
    print(f"  total_size: {info.total_size}")


if __name__ == "__main__":
    print("Mini-SGLang KV Cache Pool Demo")
    print("Using REAL code from minisgl.kvcache\n")

    demo_kvcache_interface()
    demo_kvcache_layout()
    kvcache = demo_mha_kvcache()
    demo_memory_calculation()
    demo_page_based_storage()
    demo_store_kv(kvcache)
    demo_cache_manager_interface()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. BaseKVCache is the interface for KV storage
   - k_cache(layer_id), v_cache(layer_id): Access cache tensors
   - store_kv(): Scatter K, V to non-contiguous pages

2. MHAKVCache is the Multi-Head Attention implementation
   - Stores K and V for all layers in one large tensor
   - Shape: [2, num_layers, num_pages, num_kv_heads, head_dim]

3. KVCacheLayout controls memory organization
   - LayerFirst: Better for layer-by-layer processing
   - PageFirst: Better for page-level operations

4. Page-based storage enables:
   - Non-contiguous allocation
   - Efficient memory utilization
   - Easy eviction (LRU on pages)
   - Prefix sharing across requests

5. Memory scales with:
   - 2 * num_layers * num_kv_heads * head_dim * dtype_bytes per token
""")
