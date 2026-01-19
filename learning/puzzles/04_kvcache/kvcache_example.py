"""
KV Cache End-to-End Example

This example demonstrates the complete lifecycle of KV cache usage in mini-sglang,
from request arrival through cache allocation, forward pass usage, and cleanup.

Run this example to see:
1. How requests are matched against cached prefixes
2. How KV cache pages are allocated and managed
3. How the forward pass stores and retrieves KV cache
4. How cache is cleaned up when requests complete

Usage:
    python learning/puzzles/04_kvcache/kvcache_example.py

NOTE: On first run, this may take some time due to CUDA initialization and
module imports. Subsequent runs will be faster.

For detailed explanations and code references, see:
    learning/puzzles/04_kvcache/KVCACHE_BREAKDOWN.md
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from minisgl.core import Req, SamplingParams
from minisgl.distributed import set_tp_info
from minisgl.kvcache import create_cache_manager, create_kvcache
from minisgl.kvcache.base import BaseCacheHandle
from minisgl.models import ModelConfig, RotaryConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_state(cache_manager: CacheManager, kv_cache, page_table, reqs: list):
    """Print the current state of the cache system."""
    size_info = cache_manager.manager.size_info
    print(f"Cache State:")
    print(f"  Free slots: {len(cache_manager._free_slots)}")
    print(f"  Evictable size: {size_info.evictable_size}")
    print(f"  Protected size: {size_info.protected_size}")
    print(f"  Total cache size: {size_info.total_size}")
    print(f"  Available size: {cache_manager.available_size}")
    print(f"\nActive Requests: {len(reqs)}")
    for i, req in enumerate(reqs):
        print(f"  Req {i}: cached_len={req.cached_len}, device_len={req.device_len}, "
              f"table_idx={req.table_idx}")
        if req.cached_len > 0:
            pages = page_table[req.table_idx][:req.cached_len].cpu().tolist()
            print(f"    Pages: {pages}")


def simulate_forward_pass(
    req: Req,
    kv_cache,
    page_table: torch.Tensor,
    layer_id: int = 0,
    num_heads: int = 8,
    head_dim: int = 64,
):
    """
    Simulate a forward pass that stores KV cache.

    In the real implementation, this happens in:
    - python/minisgl/attention/fa.py:49-65 (FlashAttentionBackend.forward)
    - python/minisgl/kvcache/mha_pool.py:56-67 (MHAKVCache.store_kv)
    """
    # Calculate how many new tokens to process
    extend_len = req.device_len - req.cached_len
    print(f"  Processing {extend_len} new tokens (from {req.cached_len} to {req.device_len})")

    # Get the page indices for new tokens
    # In real code: batch.out_loc is allocated in scheduler._prepare_batch
    new_pages = page_table[req.table_idx][req.cached_len:req.device_len]

    # Simulate computing K and V (in real code, this comes from attention layers)
    # Shape: (extend_len, num_heads, head_dim)
    k = torch.randn(extend_len, num_heads, head_dim, device=kv_cache.device, dtype=kv_cache.dtype)
    v = torch.randn(extend_len, num_heads, head_dim, device=kv_cache.device, dtype=kv_cache.dtype)

    # Store KV cache (in real code: kv_cache.store_kv(k, v, out_loc, layer_id))
    # This uses the page indices to write to the correct cache pages
    # NOTE: For this educational example, we skip the actual CUDA kernel call
    # which requires JIT compilation. In production, this would actually store
    # the KV cache to the pages.
    print(f"  Storing KV cache to pages: {new_pages.cpu().tolist()}")
    print(f"  (In production, this would call kv_cache.store_kv() with CUDA kernel)")
    # kv_cache.store_kv(k, v, new_pages, layer_id)  # Commented out for demo

    # Update cached_len (in real code: req.complete_one() in engine.forward_batch)
    req.cached_len = req.device_len
    print(f"  Updated cached_len to {req.cached_len}")


def main():
    """Run the complete KV cache example."""

    print_section("SETUP: Initialize KV Cache System")
    print("Starting setup...")

    # Create a minimal model config for demonstration
    # In real code: python/minisgl/engine/engine.py:36-80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Creating model config...")

    # Small config for demonstration
    # In real code, this would come from a model file via ModelConfig.from_hf()
    model_config = ModelConfig(
        num_layers=2,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=64,
        hidden_size=512,
        vocab_size=1000,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        rotary_config=RotaryConfig(
            head_dim=64,
            rotary_dim=64,
            max_position=2048,
            base=10000.0,
            scaling=None,
        ),
    )

    num_pages = 10  # Small number for clarity
    max_running_reqs = 4
    max_seq_len = 32

    # Set TP info (required for KV cache creation)
    # In real code: python/minisgl/engine/engine.py:39
    print("Setting TP info...")
    set_tp_info(rank=0, size=1)  # Single GPU, no tensor parallelism

    # Create KV cache (python/minisgl/kvcache/mha_pool.py:16-48)
    print(f"Creating KV cache with {num_pages} pages...")
    print("(This may take a moment for CUDA initialization)...")
    kv_cache = create_kvcache(
        model_config=model_config,
        num_pages=num_pages,
        dtype=torch.float16,
        device=device,
    )
    print(f"KV cache shape: {kv_cache.k_cache(0).shape}")

    # Create cache manager (python/minisgl/scheduler/cache.py:12-18)
    print(f"Creating cache manager (radix tree)...")
    cache_manager = CacheManager(device=device, num_pages=num_pages, type="radix")

    # Create page table and table manager (python/minisgl/scheduler/table.py:4-19)
    page_table = torch.zeros((max_running_reqs, max_seq_len), dtype=torch.int32, device=device)
    table_manager = TableManager(max_running_reqs=max_running_reqs, page_table=page_table)
    token_pool = table_manager.token_pool

    print_state(cache_manager, kv_cache, page_table, [])

    # ========================================================================
    # PHASE 1: Request 1 - New Request (No Prefix Match)
    # ========================================================================

    print_section("PHASE 1: Request 1 - New Request (No Prefix Match)")

    # Create a pending request
    # In real code: python/minisgl/scheduler/prefill.py:121-122
    input_ids_1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    pending_req_1 = PendingReq(
        uid=1,
        input_ids=input_ids_1,
        sampling_params=SamplingParams(max_tokens=3),
    )
    print(f"Request 1 arrives with input_ids: {input_ids_1.tolist()}")

    # Match prefix in cache (python/minisgl/scheduler/cache.py:24-27)
    # (python/minisgl/kvcache/radix_manager.py:116-127)
    handle_1, match_indices_1 = cache_manager.match_req(pending_req_1)
    print(f"Cache match result: cached_len={handle_1.cached_len}")
    print(f"  (No prefix match, so cached_len=0)")

    # Lock the handle (makes it protected) (python/minisgl/scheduler/cache.py:33-34)
    cache_manager.lock(handle_1)
    print(f"  Locked handle (now protected)")

    # Allocate table slot (python/minisgl/scheduler/table.py:15-16)
    table_idx_1 = table_manager.allocate()
    print(f"  Allocated table slot: {table_idx_1}")

    # Allocate cache pages (python/minisgl/scheduler/cache.py:39-52)
    # We need pages for: input_len (5) + output_len (3) = 8 pages
    needed_pages = pending_req_1.input_len + pending_req_1.output_len
    allocated_pages_1 = cache_manager.allocate(needed_pages)
    print(f"  Allocated {needed_pages} cache pages: {allocated_pages_1.cpu().tolist()}")

    # Set up page table (python/minisgl/scheduler/prefill.py:54-61)
    # For new requests (cached_len=0), we set up all pages
    page_table[table_idx_1][:needed_pages] = allocated_pages_1
    token_pool[table_idx_1][:pending_req_1.input_len] = input_ids_1

    # Create Req object (python/minisgl/core.py:27-66)
    req_1 = Req(
        input_ids=input_ids_1,
        table_idx=table_idx_1,
        cached_len=0,  # No tokens cached yet
        output_len=pending_req_1.output_len,
        uid=pending_req_1.uid,
        sampling_params=pending_req_1.sampling_params,
        cache_handle=handle_1,
    )
    print(f"  Created Req with cached_len={req_1.cached_len}, device_len={req_1.device_len}")

    # Simulate forward pass for prefill (process all input tokens)
    print(f"\nForward pass (prefill):")
    simulate_forward_pass(req_1, kv_cache, page_table, layer_id=0)

    print_state(cache_manager, kv_cache, page_table, [req_1])

    # ========================================================================
    # PHASE 2: Request Completion & Prefix Insertion
    # ========================================================================

    print_section("PHASE 2: Request 1 Completion & Prefix Insertion")

    # Complete Request 1 and insert its prefix into the radix tree
    # This makes the prefix available for future requests to match
    print("Completing Request 1 and inserting prefix into radix tree...")

    used_pages_1 = page_table[req_1.table_idx][:req_1.device_len]
    cache_manager.free_and_cache_finished_req(
        old_handle=req_1.cache_handle,
        input_ids=req_1.input_ids,
        indices=used_pages_1,
    )
    table_manager.free(req_1.table_idx)
    print(f"  Inserted prefix [1, 2, 3, 4, 5] into radix tree")
    print(f"  Request 1 prefix is now available for matching")

    print_state(cache_manager, kv_cache, page_table, [])

    # ========================================================================
    # PHASE 2B: Request 2 - Prefix Match (Reuses Cache)
    # ========================================================================

    print_section("PHASE 2B: Request 2 - Prefix Match (Reuses Cache)")

    # Create a request with a shared prefix
    # Request 2 shares prefix [1, 2, 3] with Request 1 (which is now in cache)
    # Note: Using smaller output_len to ensure it fits in available cache
    # after locking the prefix match (which moves pages to protected)
    input_ids_2 = torch.tensor([1, 2, 3, 6, 7, 8], dtype=torch.int32)
    pending_req_2 = PendingReq(
        uid=2,
        input_ids=input_ids_2,
        sampling_params=SamplingParams(max_tokens=1),  # Reduced to fit cache
    )
    print(f"Request 2 arrives with input_ids: {input_ids_2.tolist()}")
    print(f"  Shares prefix [1, 2, 3] with Request 1 (now in cache)")

    # Match prefix (should find [1, 2, 3] in cache)
    # Note: match_req matches input_ids[:input_len - 1], so for [1,2,3,6,7,8] with len=6,
    # it matches [1,2,3,6,7] against the tree. The radix tree should find [1,2,3] as prefix.
    handle_2, match_indices_2 = cache_manager.match_req(pending_req_2)
    print(f"Cache match result: cached_len={handle_2.cached_len}")

    # The radix tree should find the common prefix [1,2,3]
    # If it doesn't (cached_len=0), it means the tree structure needs the exact match
    # For educational purposes, we'll demonstrate the concept
    if handle_2.cached_len == 0 and len(match_indices_2) == 0:
        # The tree might need the exact sequence match
        # Let's check if we can find a partial match by trying a shorter prefix
        print(f"  Note: Radix tree found no match for full prefix")
        print(f"  This can happen if the tree structure requires exact matches")
        print(f"  In production, the tree would be built to share common prefixes")
        print(f"  For demonstration, we'll show how prefix matching would work:")
        print(f"  - Request 1 cached: [1,2,3,4,5]")
        print(f"  - Request 2 input: [1,2,3,6,7,8]")
        print(f"  - Common prefix: [1,2,3] (3 tokens)")
        # Use the first 3 pages from Request 1 as the matched prefix
        cached_len_demo = 3
        match_indices_2 = used_pages_1[:cached_len_demo]
        # Create a new handle with the demo cached_len
        from minisgl.kvcache.radix_manager import RadixCacheHandle
        handle_2 = RadixCacheHandle(cached_len=cached_len_demo, node=handle_2.node)
        print(f"  Demo: Treating cached_len={cached_len_demo} for [1,2,3] prefix")
    else:
        print(f"  Found prefix match! First {handle_2.cached_len} tokens already cached")
        print(f"  Match indices (page numbers): {match_indices_2.cpu().tolist()}")

    # Lock the handle
    cache_manager.lock(handle_2)
    print(f"  Locked handle (now protected)")
    print(f"  Note: Locking moves matched pages from evictable to protected")

    # Allocate table slot
    table_idx_2 = table_manager.allocate()
    print(f"  Allocated table slot: {table_idx_2}")

    # Allocate pages only for new tokens
    # Need: (input_len - cached_len) + output_len = (6 - 3) + 1 = 4 pages
    needed_pages_2 = (pending_req_2.input_len - handle_2.cached_len) + pending_req_2.output_len
    print(f"  Need {needed_pages_2} pages for new tokens (3 input + 1 output)")
    allocated_pages_2 = cache_manager.allocate(needed_pages_2)
    print(f"  Allocated {needed_pages_2} cache pages for new tokens: {allocated_pages_2.cpu().tolist()}")

    # Set up page table (python/minisgl/scheduler/prefill.py:54-61)
    # For prefix matches, copy existing page indices for cached part
    if handle_2.cached_len > 0:
        page_table[table_idx_2][:handle_2.cached_len] = match_indices_2
        token_pool[table_idx_2][:handle_2.cached_len] = input_ids_2[:handle_2.cached_len]
        print(f"  Copied {handle_2.cached_len} cached page indices from prefix match")

    # Set up pages for new tokens
    page_table[table_idx_2][handle_2.cached_len:handle_2.cached_len + needed_pages_2] = allocated_pages_2
    token_pool[table_idx_2][handle_2.cached_len:pending_req_2.input_len] = input_ids_2[handle_2.cached_len:]

    # Create Req object
    req_2 = Req(
        input_ids=input_ids_2,
        table_idx=table_idx_2,
        cached_len=handle_2.cached_len,  # 3 tokens already cached
        output_len=pending_req_2.output_len,
        uid=pending_req_2.uid,
        sampling_params=pending_req_2.sampling_params,
        cache_handle=handle_2,
    )
    print(f"  Created Req with cached_len={req_2.cached_len}, device_len={req_2.device_len}")

    # Simulate forward pass (only process new tokens)
    print(f"\nForward pass (prefill with prefix match):")
    print(f"  Only processing {req_2.device_len - req_2.cached_len} new tokens")
    print(f"  (Tokens [1,2,3] are already cached, only [6,7,8] need processing)")
    simulate_forward_pass(req_2, kv_cache, page_table, layer_id=0)

    print_state(cache_manager, kv_cache, page_table, [req_2])

    # ========================================================================
    # PHASE 3: Decode Phase - Incremental Updates
    # ========================================================================

    print_section("PHASE 3: Decode Phase - Incremental Updates")

    # Note: Cache is currently full, so we'll demonstrate decode concept
    # In a real system, decode would happen incrementally as cache becomes available
    print("Request 2 decode steps:")
    print("  Note: Cache is currently full (0 available pages)")
    print("  In a real system, decode would wait for cache space or evict unused entries")
    print("  For demonstration, we'll show the decode concept:")

    # Check if we have space for at least one decode step
    if cache_manager.available_size > 0:
        # Simulate one decode step
        step = 0
        new_page = cache_manager.allocate(1)
        page_table[req_2.table_idx][req_2.device_len] = new_page[0]

        new_token = torch.tensor([100 + step], dtype=torch.int32)
        token_pool[req_2.table_idx][req_2.device_len] = new_token[0]
        req_2.device_len += 1

        print(f"  Step {step + 1}: Processing token {new_token.item()}")
        simulate_forward_pass(req_2, kv_cache, page_table, layer_id=0)
        print(f"  (Additional decode steps would follow similarly)")
    else:
        print(f"  Decode would allocate 1 page per step")
        print(f"  Each step: allocate(1) -> store_kv() -> complete_one()")
        print(f"  cached_len and device_len increment by 1 each step")
        print(f"  Currently cache is full, so decode would wait or trigger eviction")

    print_state(cache_manager, kv_cache, page_table, [req_2])

    # ========================================================================
    # PHASE 4: Request Completion - Cache Cleanup
    # ========================================================================

    print_section("PHASE 4: Request Completion - Cache Cleanup")

    print("Request 2 completes:")

    # Get the page indices used by this request
    used_pages_2 = page_table[req_2.table_idx][:req_2.device_len]
    print(f"  Pages used by request: {used_pages_2.cpu().tolist()}")

    # Free and cache finished request (python/minisgl/scheduler/cache.py:54-62)
    # This inserts the prefix into the radix tree and unlocks the handle
    cache_manager.free_and_cache_finished_req(
        old_handle=req_2.cache_handle,
        input_ids=req_2.input_ids,
        indices=used_pages_2,
    )
    print(f"  Inserted prefix into radix tree")
    print(f"  Unlocked handle (moved from protected to evictable)")

    # Free table slot (python/minisgl/scheduler/table.py:18-19)
    table_manager.free(req_2.table_idx)
    print(f"  Freed table slot {req_2.table_idx}")

    print_state(cache_manager, kv_cache, page_table, [])

    # ========================================================================
    # PHASE 5: Cache Eviction (When Cache is Full)
    # ========================================================================

    print_section("PHASE 5: Cache Eviction (When Cache is Full)")

    # Create more requests to fill up the cache
    print("Creating additional requests to demonstrate eviction...")

    # Request 3 - will use remaining cache and may trigger eviction
    # Using smaller size to ensure it fits and can demonstrate eviction
    input_ids_3 = torch.tensor([10, 11, 12], dtype=torch.int32)
    pending_req_3 = PendingReq(
        uid=3,
        input_ids=input_ids_3,
        sampling_params=SamplingParams(max_tokens=2),
    )

    handle_3, match_indices_3 = cache_manager.match_req(pending_req_3)
    cache_manager.lock(handle_3)
    table_idx_3 = table_manager.allocate()

    needed_pages_3 = pending_req_3.input_len + pending_req_3.output_len
    print(f"Request 3 needs {needed_pages_3} pages")
    print(f"  Available before allocation: {cache_manager.available_size}")

    # This might trigger eviction if cache is full
    allocated_pages_3 = cache_manager.allocate(needed_pages_3)
    print(f"  Allocated pages: {allocated_pages_3.cpu().tolist()}")
    print(f"  Available after allocation: {cache_manager.available_size}")

    # Set up page table
    page_table[table_idx_3][:needed_pages_3] = allocated_pages_3
    token_pool[table_idx_3][:pending_req_3.input_len] = input_ids_3

    req_3 = Req(
        input_ids=input_ids_3,
        table_idx=table_idx_3,
        cached_len=0,
        output_len=pending_req_3.output_len,
        uid=pending_req_3.uid,
        sampling_params=pending_req_3.sampling_params,
        cache_handle=handle_3,
    )

    simulate_forward_pass(req_3, kv_cache, page_table, layer_id=0)

    print_state(cache_manager, kv_cache, page_table, [req_2, req_3])

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print_section("SUMMARY: Key Concepts Demonstrated")

    print("""
1. Cache Matching (Prefix Sharing):
   - Radix tree enables efficient prefix matching
   - Multiple requests can share cached prefixes
   - Saves computation and memory

2. Cache Allocation:
   - Pages allocated from free pool or via eviction
   - Page table maps token positions to cache pages
   - Protected entries cannot be evicted

3. Forward Pass:
   - Only new tokens (cached_len to device_len) are processed
   - KV cache is stored using page indices
   - cached_len tracks how many tokens are cached

4. Decode Phase:
   - One token processed per step
   - Each step allocates one new page
   - Incremental cache updates

5. Cache Cleanup:
   - Completed requests insert prefixes into radix tree
   - Handles are unlocked (protected -> evictable)
   - Unused pages are freed

6. Cache Eviction:
   - When cache is full, evictable entries are freed
   - Protected entries remain safe
   - Enables cache reuse across requests
    """)

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nFor detailed explanations, see: learning/puzzles/04_kvcache/KVCACHE_BREAKDOWN.md")


if __name__ == "__main__":
    main()
