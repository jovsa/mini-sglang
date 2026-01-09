#!/usr/bin/env python3
"""
Learning Prototype 7: CUDA Graph
=================================

Demonstrates Mini-SGLang's CUDA graph optimization:
- GraphRunner: Captures and replays decode graphs
- Why CUDA graphs matter for LLM inference
- How graphs are captured for different batch sizes
- Padding strategy for efficient replay

Uses real code concepts from minisgl.engine.graph.
"""

import torch

# Import REAL Mini-SGLang components
from minisgl.engine.graph import (
    _determine_cuda_graph_bs,
    mem_GB,
    get_free_memory,
)


def demo_why_cuda_graphs():
    """Explain why CUDA graphs matter."""
    print("=" * 60)
    print("1. Why CUDA Graphs Matter")
    print("=" * 60)

    print("""
Problem: CPU Launch Overhead

In decode phase, each forward pass:
1. CPU schedules many small CUDA kernels
2. Each kernel launch has ~5-10μs overhead
3. For a 32-layer model: hundreds of kernel launches
4. Total overhead: 1-2ms per forward pass

This matters because:
- Decode is memory-bound (fast GPU computation)
- Actual GPU work might be only 2-3ms
- 1-2ms overhead = 30-50% slowdown!

Solution: CUDA Graphs

1. CAPTURE: Record a "trace" of all kernel launches
2. REPLAY: Execute entire trace with single CPU call

Before (no graph):
┌─────────────────────────────────────────────────────────┐
│ CPU: launch → launch → launch → launch → ... (100+ ops)│
│ GPU: ▓▓▓░░░▓▓▓░░░▓▓▓░░░▓▓▓░░░ ... (gaps between kernels)│
└─────────────────────────────────────────────────────────┘

After (with graph):
┌─────────────────────────────────────────────────────────┐
│ CPU: replay_graph() (single call)                       │
│ GPU: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (continuous execution)  │
└─────────────────────────────────────────────────────────┘
""")


def demo_graph_bs_selection():
    """Show how batch sizes for graphs are determined."""
    print("\n" + "=" * 60)
    print("2. CUDA Graph Batch Size Selection")
    print("=" * 60)

    print("""
CUDA graphs are captured for specific batch sizes.
Mini-SGLang uses a smart selection strategy:

_determine_cuda_graph_bs(cuda_graph_bs, cuda_graph_max_bs, free_memory):
  - If custom bs list provided: use it
  - Otherwise: [1, 2, 4, 8, 16, 24, 32, ..., max_bs]
  - max_bs depends on GPU memory:
    - H200 (>80GB): up to 256
    - Others: up to 160
""")

    # Demo the function
    print("\nExample batch size lists:")

    # Simulate different memory amounts
    for free_gb, name in [(90, "H200"), (40, "A100-40G"), (16, "RTX 4090")]:
        free_bytes = free_gb * (1 << 30)
        bs_list = _determine_cuda_graph_bs(None, None, free_bytes)
        print(f"  {name} ({free_gb}GB): {bs_list[:5]}...{bs_list[-3:]}")
        print(f"    Total graphs: {len(bs_list)}, max_bs: {max(bs_list)}")


def demo_graph_runner_concept():
    """Explain GraphRunner class."""
    print("\n" + "=" * 60)
    print("3. GraphRunner - Capture and Replay")
    print("=" * 60)

    print("""
GraphRunner manages CUDA graph lifecycle:

class GraphRunner:
    def __init__(self, ...):
        # 1. Determine batch sizes to capture
        self.graph_bs_list = [1, 2, 4, 8, 16, ...]

        # 2. Pre-allocate output buffer (shared across all graphs)
        self.logits = torch.empty(max_bs, vocab_size)

        # 3. Initialize attention backend for capture
        attn_backend.init_capture_graph(max_seq_len, bs_list)

        # 4. Capture graphs for each batch size
        self.graph_map = self._capture_graphs(...)

Key methods:
  - can_use_cuda_graph(batch): Check if graph available
  - replay(batch): Execute captured graph
  - pad_batch(batch): Pad to valid graph size
""")


def demo_capture_process():
    """Show the graph capture process."""
    print("\n" + "=" * 60)
    print("4. Graph Capture Process")
    print("=" * 60)

    print("""
Capture happens at initialization:

for bs in [256, 248, 240, ..., 8, 4, 2, 1]:  # reverse order
    1. Create dummy batch with bs requests
       batch = Batch(reqs=[dummy_req] * bs, phase="decode")

    2. Prepare attention metadata for capture
       attn_backend.prepare_for_capture(batch)

    3. Warm-up run (ensures all memory allocated)
       logits[:bs] = model.forward()

    4. Capture the graph
       with torch.cuda.graph(graph, pool=pool):
           logits[:bs] = model.forward()

    5. Store in graph_map
       graph_map[bs] = graph

Memory pool sharing:
  - All graphs share same memory pool
  - Capture largest first to allocate pool
  - Smaller graphs reuse pool memory
""")


def demo_replay_process():
    """Show how replay works."""
    print("\n" + "=" * 60)
    print("5. Graph Replay Process")
    print("=" * 60)

    print("""
During inference, GraphRunner.replay(batch):

1. Check if graph available
   assert can_use_cuda_graph(batch)  # decode + bs <= max

2. Get padded batch size
   padded_size = next(bs for bs in graph_bs_list if bs >= batch.size)
   # e.g., batch.size=5 -> padded_size=8

3. Update attention metadata with real data
   attn_backend.prepare_for_replay(batch)
   # Copies: input_ids, out_loc, positions, seq_lens, page_table

4. Replay the graph
   graph_map[padded_size].replay()

5. Return real results (ignoring padding)
   return logits[:batch.size]

The key insight:
  - Graph records kernel launches, not data
  - Same graph works for different inputs
  - Just update the input tensors before replay
""")


def demo_padding_strategy():
    """Explain the padding strategy."""
    print("\n" + "=" * 60)
    print("6. Batch Padding Strategy")
    print("=" * 60)

    print("""
Problem: Batch sizes vary during inference
  - User requests arrive at different times
  - Actual batch might be 1, 5, 13, 27, etc.
  - Can't capture graph for every possible size

Solution: Pad to nearest captured size

Graph batch sizes: [1, 2, 4, 8, 16, 24, 32, ...]

Padding examples:
  batch.size=1  -> padded=1  (exact match)
  batch.size=3  -> padded=4  (pad 1 dummy req)
  batch.size=5  -> padded=8  (pad 3 dummy reqs)
  batch.size=9  -> padded=16 (pad 7 dummy reqs)
  batch.size=17 -> padded=24 (pad 7 dummy reqs)

Implementation in pad_batch():
  padded_size = next(bs for bs in graph_bs_list if bs >= batch.size)
  batch.padded_reqs = batch.reqs + [dummy_req] * (padded_size - batch.size)
""")

    # Demo padding
    graph_bs_list = [1, 2, 4, 8, 16, 24, 32, 40, 48]

    print("\nDemo padding calculation:")
    for actual_bs in [1, 3, 5, 9, 17, 25]:
        padded = next(bs for bs in graph_bs_list if bs >= actual_bs)
        overhead = padded - actual_bs
        pct = overhead / padded * 100
        print(f"  actual={actual_bs:2d} -> padded={padded:2d} (overhead: {overhead:2d}, {pct:.0f}%)")


def demo_dummy_request():
    """Explain the dummy request used for padding."""
    print("\n" + "=" * 60)
    print("7. Dummy Request for Padding")
    print("=" * 60)

    print("""
Padding uses a "dummy request" that:
  - Has valid structure but produces ignored outputs
  - Uses a dedicated page table row (dummy_page)
  - Has table_idx = max_running_req (last row)

In Engine.__init__:
  self.dummy_req = Req(
      input_ids=tensor([0]),
      table_idx=config.max_running_req,  # dedicated row
      cached_len=0,
      output_len=1,
      uid=-1,
      sampling_params=None,
      cache_handle=None,
  )

  # Fill dummy page table with dummy_page index
  self.page_table[dummy_req.table_idx].fill_(self.dummy_page)

This ensures:
  - Padding doesn't corrupt real KV cache
  - Attention reads from valid memory
  - Output for padding is safely ignored
""")


def demo_when_graphs_used():
    """Show when CUDA graphs are used vs not used."""
    print("\n" + "=" * 60)
    print("8. When CUDA Graphs Are Used")
    print("=" * 60)

    print("""
can_use_cuda_graph(batch) returns True when:
  1. batch.is_decode == True  (not prefill)
  2. batch.size <= max_graph_bs

CUDA graphs are NOT used for:

┌─────────────────────────────────────────────────────────┐
│ PREFILL: Variable sequence lengths                      │
│   - Each request has different input length             │
│   - Can't capture graph for all combinations            │
│   - Compute-bound anyway, less overhead impact          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ LARGE DECODE BATCHES: bs > max_graph_bs                │
│   - Would need too many graphs                          │
│   - Memory overhead not worth it                        │
│   - Falls back to normal execution                      │
└─────────────────────────────────────────────────────────┘

In Engine.forward_batch():
  if graph_runner.can_use_cuda_graph(batch):
      logits = graph_runner.replay(batch)
  else:
      logits = model.forward()  # normal execution
""")


if __name__ == "__main__":
    print("Mini-SGLang CUDA Graph Demo")
    print("Using concepts from minisgl.engine.graph\n")

    demo_why_cuda_graphs()
    demo_graph_bs_selection()
    demo_graph_runner_concept()
    demo_capture_process()
    demo_replay_process()
    demo_padding_strategy()
    demo_dummy_request()
    demo_when_graphs_used()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. CUDA graphs eliminate CPU launch overhead
   - Capture: Record all kernel launches once
   - Replay: Execute with single CPU call

2. GraphRunner manages graph lifecycle
   - Captures graphs for multiple batch sizes
   - Shares memory pool across graphs
   - Handles padding to valid sizes

3. Graphs work because:
   - Same kernel sequence for all decode
   - Only data changes, not structure
   - Update input tensors before replay

4. Padding strategy:
   - Round up to nearest captured size
   - Use dummy requests for padding
   - Output for padding is ignored

5. Graphs NOT used for:
   - Prefill (variable lengths)
   - Large batches (> max_graph_bs)
""")
