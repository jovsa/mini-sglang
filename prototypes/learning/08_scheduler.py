#!/usr/bin/env python3
"""
Learning Prototype 8: Scheduler and Managers
==============================================

Demonstrates Mini-SGLang's scheduling system:
- PrefillManager: Manages prefill requests with chunked prefill
- DecodeManager: Manages decode batch
- TableManager: Manages page table allocation
- CacheManager: Wraps RadixCacheManager
- Chunked prefill: Splits long prompts to avoid OOM

Uses real code from minisgl.scheduler.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Set

# Import REAL Mini-SGLang components
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.core import Req, Batch, SamplingParams


# Minimal cache handle for demo
@dataclass(frozen=True)
class DummyCacheHandle:
    cached_len: int = 0


def demo_decode_manager():
    """Demonstrate DecodeManager."""
    print("=" * 60)
    print("1. DecodeManager - Managing Decode Requests")
    print("=" * 60)

    print("""
DecodeManager tracks requests in decode phase:

class DecodeManager:
    running_reqs: Set[Req]  # All requests currently decoding

    def add_reqs(reqs): Add completed prefill requests
    def remove_req(req): Remove finished request
    def schedule_next_batch() -> Batch: Create decode batch
    @property inflight_tokens: Total tokens left to generate
    @property runnable: True if has requests to decode
""")

    # Create a DecodeManager
    dm = DecodeManager()

    print(f"\nInitial state:")
    print(f"  running_reqs: {len(dm.running_reqs)}")
    print(f"  runnable: {dm.runnable}")
    print(f"  inflight_tokens: {dm.inflight_tokens}")

    # Create some requests (simulate after prefill)
    reqs = []
    for i in range(3):
        # Start with cached_len=0, device_len=3 (initial state)
        req = Req(
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            table_idx=i,
            cached_len=0,  # Nothing cached initially
            output_len=10 - i * 2,  # Varying output lengths
            uid=i,
            sampling_params=SamplingParams(max_tokens=10 - i * 2),
            cache_handle=DummyCacheHandle(0),
        )
        # Simulate prefill completion: cached_len becomes device_len
        req.complete_one()
        # Add generated token
        req.append_host(torch.tensor([100], dtype=torch.int32))
        reqs.append(req)

    # Add requests
    dm.add_reqs(reqs)

    print(f"\nAfter adding 3 requests:")
    print(f"  running_reqs: {len(dm.running_reqs)}")
    print(f"  runnable: {dm.runnable}")
    print(f"  inflight_tokens: {dm.inflight_tokens}")

    # Schedule a decode batch
    batch = dm.schedule_next_batch()
    print(f"\nScheduled decode batch:")
    print(f"  size: {batch.size}")
    print(f"  phase: {batch.phase}")
    print(f"  is_decode: {batch.is_decode}")


def demo_chunked_prefill_concept():
    """Explain chunked prefill."""
    print("\n" + "=" * 60)
    print("2. Chunked Prefill - Splitting Long Prompts")
    print("=" * 60)

    print("""
Problem: Long prompts can cause OOM
  - Prefill computes attention for ALL input tokens
  - Very long prompts (4K, 8K, 32K tokens) need lots of memory
  - Peak memory can exceed GPU capacity

Solution: Chunked Prefill (from Sarathi-Serve)

Split long prompts into chunks:
┌─────────────────────────────────────────────────────────┐
│ Original prompt: 4096 tokens                            │
│ Chunk size: 1024 tokens                                 │
│                                                         │
│ Chunk 1: tokens[0:1024]     -> prefill, no sampling    │
│ Chunk 2: tokens[1024:2048]  -> prefill, no sampling    │
│ Chunk 3: tokens[2048:3072]  -> prefill, no sampling    │
│ Chunk 4: tokens[3072:4096]  -> prefill, SAMPLE token   │
│                                                         │
│ Then: decode loop starts                               │
└─────────────────────────────────────────────────────────┘

Benefits:
  - Bounded peak memory (only chunk_size tokens at once)
  - Still uses KV cache efficiently
  - Overlaps with other requests

ChunkedReq: Special Req that can't decode yet
  - can_decode() returns False
  - Must complete all chunks first
""")


def demo_prefill_manager_concept():
    """Explain PrefillManager."""
    print("\n" + "=" * 60)
    print("3. PrefillManager - Scheduling Prefill")
    print("=" * 60)

    print("""
PrefillManager handles incoming requests:

class PrefillManager:
    pending_list: List[PendingReq]  # Waiting requests
    cache_manager: CacheManager     # For prefix matching
    table_manager: TableManager     # For page allocation
    decode_manager: DecodeManager   # To track inflight tokens

    def add_one_req(msg: UserMsg):
        # Add new request to pending list
        pending_list.append(PendingReq(...))

    def schedule_next_batch(prefill_budget) -> Batch:
        # 1. For each pending request:
        #    - Match prefix in cache
        #    - Allocate table entry
        #    - Check token budget
        #
        # 2. Create prefill batch
        #    - May create ChunkedReq if over budget
        #
        # 3. Return batch or None

Key concepts:
  - prefill_budget: Max tokens per prefill batch
  - reserved_size: Account for decode inflight tokens
  - ChunkedReq: Request split across multiple batches
""")


def demo_prefill_adder_logic():
    """Explain PrefillAdder logic."""
    print("\n" + "=" * 60)
    print("4. PrefillAdder - Adding Requests to Batch")
    print("=" * 60)

    print("""
PrefillAdder decides how many tokens to add per request:

class PrefillAdder:
    token_budget: int      # Remaining tokens we can add
    reserved_size: int     # Space reserved for decode

    def try_add_one(pending_req) -> Req | None:
        # 1. Check if budget allows adding
        if token_budget <= 0:
            return None

        # 2. If continuing chunked request, just add next chunk
        if pending_req.chunked_req:
            return _add_one_req(...)

        # 3. Try to allocate resources
        handle, match_indices = cache_manager.match_req(req)
        cached_len = handle.cached_len  # Prefix cache hit!

        # 4. Check if we have enough cache space
        extend_len = input_len - cached_len
        estimated_len = extend_len + output_len
        if estimated_len > available_size:
            return None  # Not enough space

        # 5. Allocate table entry and add request
        table_idx = table_manager.allocate()
        return _add_one_req(...)

    def _add_one_req(...) -> Req:
        remain_len = input_len - cached_len
        chunk_size = min(token_budget, remain_len)

        if chunk_size < remain_len:
            # Can't fit whole request, create ChunkedReq
            return ChunkedReq(...)
        else:
            # Can fit whole request
            return Req(...)
""")


def demo_chunked_req():
    """Demonstrate ChunkedReq behavior."""
    print("\n" + "=" * 60)
    print("5. ChunkedReq - Partial Prefill Request")
    print("=" * 60)

    print("""
ChunkedReq extends Req but prevents decoding:

class ChunkedReq(Req):
    def append_host(self, next_token):
        raise NotImplementedError("ChunkedReq should be sampled")

    def can_decode(self) -> bool:
        return False  # Can't decode until all chunks done

This ensures:
  - Scheduler doesn't add to decode batch prematurely
  - Must complete remaining chunks first
  - Then converted to regular Req for decode
""")

    # Create example
    input_ids = torch.tensor(list(range(100)), dtype=torch.int32)

    # Regular Req (can decode after prefill)
    regular_req = Req(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=10,
        uid=1,
        sampling_params=SamplingParams(max_tokens=10),
        cache_handle=DummyCacheHandle(0),
    )
    regular_req.complete_one()
    regular_req.append_host(torch.tensor([999], dtype=torch.int32))

    print(f"\nRegular Req after prefill:")
    print(f"  can_decode(): {regular_req.can_decode()}")

    # ChunkedReq (can't decode yet)
    # Note: ChunkedReq is created internally by PrefillManager


def demo_table_manager_concept():
    """Explain TableManager."""
    print("\n" + "=" * 60)
    print("6. TableManager - Page Table Management")
    print("=" * 60)

    print("""
TableManager allocates page table entries:

class TableManager:
    page_table: Tensor     # [max_requests, max_seq_len]
    token_pool: Tensor     # [max_requests, max_seq_len] for input_ids
    free_slots: List[int]  # Available table indices

    def allocate() -> int:
        # Get next free table index
        return free_slots.pop()

    def free(table_idx):
        # Return table index to free pool
        free_slots.append(table_idx)

    @property
    def available_size -> int:
        return len(free_slots)

Page table structure:
┌─────────────────────────────────────────────────────────┐
│ page_table[req_idx, pos] = physical_page_index         │
├─────────────────────────────────────────────────────────┤
│ Row 0: [page_5, page_12, page_3, ...]  (Request 0)     │
│ Row 1: [page_8, page_1, page_9, ...]   (Request 1)     │
│ Row 2: [FREE]                                           │
│ Row 3: [FREE]                                           │
└─────────────────────────────────────────────────────────┘
""")


def demo_scheduler_flow():
    """Show overall scheduler flow."""
    print("\n" + "=" * 60)
    print("7. Scheduler Flow - Putting It Together")
    print("=" * 60)

    print("""
Main scheduler loop (simplified):

while True:
    # 1. Receive new requests from tokenizer
    msgs = receive_msg(blocking=not has_work)
    for msg in msgs:
        if isinstance(msg, UserMsg):
            prefill_manager.add_one_req(msg)

    # 2. Try to create prefill batch
    if prefill_manager.runnable:
        batch = prefill_manager.schedule_next_batch(prefill_budget)
        if batch:
            # Forward prefill batch
            output = engine.forward_batch(batch, ...)
            # Add completed prefills to decode
            decode_manager.add_reqs(batch.reqs)
            # Send results for chunked/completed
            send_results(...)

    # 3. Create decode batch
    if decode_manager.runnable:
        batch = decode_manager.schedule_next_batch()
        # Pad batch for CUDA graph
        graph_runner.pad_batch(batch)
        # Forward decode batch
        output = engine.forward_batch(batch, ...)
        # Update requests, send results
        for req, token in zip(batch.reqs, output):
            req.append_host(token)
            if not req.can_decode():
                decode_manager.remove_req(req)
            send_results(...)
""")


def demo_overlap_scheduling():
    """Explain overlap scheduling."""
    print("\n" + "=" * 60)
    print("8. Overlap Scheduling - Hiding CPU Overhead")
    print("=" * 60)

    print("""
Overlap scheduling hides CPU overhead (from NanoFlow):

Normal scheduling:
┌─────────────────────────────────────────────────────────┐
│ Time →                                                  │
│                                                         │
│ CPU: [schedule][───wait───][schedule][───wait───]      │
│ GPU: [───────forward───────][───────forward───────]    │
│                                                         │
│ CPU scheduling waits for GPU to finish                  │
└─────────────────────────────────────────────────────────┘

Overlap scheduling:
┌─────────────────────────────────────────────────────────┐
│ Time →                                                  │
│                                                         │
│ CPU: [sched][sched][sched][sched][sched][sched]        │
│ GPU: [──forward──][──forward──][──forward──]──]        │
│             ↑                                           │
│             └─ GPU starts when batch ready              │
│                                                         │
│ CPU schedules next batch while GPU runs current         │
└─────────────────────────────────────────────────────────┘

Implementation:
  - Keep forward_data for current batch
  - Schedule next batch while current runs
  - Synchronize only when needed
""")


if __name__ == "__main__":
    print("Mini-SGLang Scheduler Demo")
    print("Using REAL code from minisgl.scheduler\n")

    demo_decode_manager()
    demo_chunked_prefill_concept()
    demo_prefill_manager_concept()
    demo_prefill_adder_logic()
    demo_chunked_req()
    demo_table_manager_concept()
    demo_scheduler_flow()
    demo_overlap_scheduling()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. PrefillManager handles incoming requests
   - Matches prefix in cache
   - Allocates page table entries
   - Implements chunked prefill

2. DecodeManager tracks decoding requests
   - Simple set of running requests
   - Creates decode batches

3. Chunked Prefill prevents OOM
   - Splits long prompts into chunks
   - ChunkedReq can't decode until complete
   - Bounded peak memory

4. TableManager allocates page table rows
   - Each request gets a row
   - Maps logical positions to pages

5. Overlap Scheduling hides CPU overhead
   - Schedule next batch while GPU runs
   - Improves overall throughput
""")
