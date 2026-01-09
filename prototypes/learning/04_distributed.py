#!/usr/bin/env python3
"""
Learning Prototype 4: Distributed Communication
================================================

Demonstrates Mini-SGLang's tensor parallelism communication:
- DistributedInfo: TP rank/size tracking
- DistributedCommunicator: Unified interface for all_reduce/all_gather
- TorchDistributedImpl vs PyNCCLDistributedImpl

Uses real code from minisgl.distributed.

Note: Full multi-GPU demo requires launching multiple processes.
This prototype shows the interfaces and simulates single-rank behavior.
"""

import torch

# Import REAL Mini-SGLang components
from minisgl.distributed.info import DistributedInfo, set_tp_info, get_tp_info
from minisgl.distributed.impl import (
    DistributedImpl,
    TorchDistributedImpl,
    DistributedCommunicator,
)

# Reset for demo
import minisgl.distributed.info as info_module
info_module._TP_INFO = None


def demo_distributed_info():
    """Demonstrate DistributedInfo - TP rank/size tracking."""
    print("=" * 60)
    print("1. DistributedInfo - TP Rank and Size")
    print("=" * 60)

    # Create DistributedInfo directly
    tp_info = DistributedInfo(rank=0, size=4)

    print(f"\nDistributedInfo(rank=0, size=4):")
    print(f"  rank: {tp_info.rank}")
    print(f"  size: {tp_info.size}")
    print(f"  is_primary(): {tp_info.is_primary()}")

    # Non-primary rank
    tp_info2 = DistributedInfo(rank=2, size=4)
    print(f"\nDistributedInfo(rank=2, size=4):")
    print(f"  is_primary(): {tp_info2.is_primary()}")

    # Global TP info (set once at startup)
    info_module._TP_INFO = None
    set_tp_info(rank=0, size=2)
    global_tp = get_tp_info()
    print(f"\nGlobal TP info (via set_tp_info/get_tp_info):")
    print(f"  {global_tp}")


def demo_distributed_impl():
    """Demonstrate the DistributedImpl interface."""
    print("\n" + "=" * 60)
    print("2. DistributedImpl - Communication Interface")
    print("=" * 60)

    print("""
DistributedImpl is an abstract interface with two methods:
  - all_reduce(x): Sum tensors across all ranks (in-place)
  - all_gather(x): Concatenate tensors from all ranks

Two implementations exist:
  1. TorchDistributedImpl: Uses torch.distributed (NCCL backend)
  2. PyNCCLDistributedImpl: Uses custom PyNCCL bindings (lower overhead)
""")

    # Show the interface
    print("TorchDistributedImpl methods:")
    impl = TorchDistributedImpl()
    print(f"  all_reduce(x) -> Tensor  # torch.distributed.all_reduce")
    print(f"  all_gather(x) -> Tensor  # torch.distributed.all_gather_into_tensor")


def demo_communicator():
    """Demonstrate DistributedCommunicator - the unified interface."""
    print("\n" + "=" * 60)
    print("3. DistributedCommunicator - Plugin Architecture")
    print("=" * 60)

    print("""
DistributedCommunicator uses a plugin architecture:
  - plugins: List[DistributedImpl] (stack of implementations)
  - Default: [TorchDistributedImpl()]
  - PyNCCL can be added on top when initialized

The last plugin in the list is used (allows switching implementations).
""")

    # Show the communicator
    comm = DistributedCommunicator()
    print(f"Current plugins: {len(comm.plugins)}")
    print(f"  Active: {type(comm.plugins[-1]).__name__}")

    # Note: Actually calling all_reduce requires torch.distributed to be initialized
    # In a real system, this happens in Engine.__init__
    print(f"\nNote: Actual communication requires torch.distributed.init_process_group()")
    print(f"      which is done in minisgl.engine.Engine.__init__")

    # Simulated single-rank behavior
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"\nSimulated single-rank (TP=1) behavior:")
    print(f"  Input: {x.tolist()}")
    print(f"  all_reduce: {x.tolist()} (no-op when TP=1)")


def demo_all_reduce_concept():
    """Explain all_reduce with a diagram."""
    print("\n" + "=" * 60)
    print("4. All-Reduce Operation (Concept)")
    print("=" * 60)

    print("""
All-Reduce: Each rank has partial results, sum across all ranks.

Example with TP=4, computing output of Row-Parallel Linear:

Before all_reduce:
┌─────────────────────────────────────────────────────────┐
│  Rank 0: [1, 2, 3]  (partial from input columns 0-3)   │
│  Rank 1: [4, 5, 6]  (partial from input columns 4-7)   │
│  Rank 2: [2, 3, 4]  (partial from input columns 8-11)  │
│  Rank 3: [1, 1, 1]  (partial from input columns 12-15) │
└─────────────────────────────────────────────────────────┘
                           │
                      all_reduce(SUM)
                           │
                           ▼
After all_reduce (same on ALL ranks):
┌─────────────────────────────────────────────────────────┐
│  All Ranks: [8, 11, 14]  (sum of all partials)         │
└─────────────────────────────────────────────────────────┘

Used in: LinearRowParallel, LinearOProj (attention output projection)
""")

    # Simulate what would happen
    partials = [
        torch.tensor([1., 2., 3.]),
        torch.tensor([4., 5., 6.]),
        torch.tensor([2., 3., 4.]),
        torch.tensor([1., 1., 1.]),
    ]
    result = sum(partials)
    print(f"Simulated result: {result.tolist()}")


def demo_all_gather_concept():
    """Explain all_gather with a diagram."""
    print("\n" + "=" * 60)
    print("5. All-Gather Operation (Concept)")
    print("=" * 60)

    print("""
All-Gather: Each rank has a slice, concatenate across all ranks.

Example with TP=4, gathering embedding outputs:

Before all_gather:
┌─────────────────────────────────────────────────────────┐
│  Rank 0: [emb_0, emb_1]   (vocab slice 0-999)          │
│  Rank 1: [emb_2, emb_3]   (vocab slice 1000-1999)      │
│  Rank 2: [emb_4, emb_5]   (vocab slice 2000-2999)      │
│  Rank 3: [emb_6, emb_7]   (vocab slice 3000-3999)      │
└─────────────────────────────────────────────────────────┘
                           │
                       all_gather
                           │
                           ▼
After all_gather (same on ALL ranks):
┌─────────────────────────────────────────────────────────┐
│  All: [emb_0, emb_1, emb_2, emb_3, emb_4, ..., emb_7]  │
└─────────────────────────────────────────────────────────┘

Used in: Vocab parallel embedding, gathering outputs for sampling
""")

    # Simulate
    slices = [
        torch.tensor([[1., 2.], [3., 4.]]),
        torch.tensor([[5., 6.], [7., 8.]]),
    ]
    result = torch.cat(slices, dim=0)
    print(f"Simulated gather (2 ranks, 2 items each):")
    print(f"  Result shape: {result.shape}")
    print(f"  Result:\n{result}")


def demo_communication_in_model():
    """Show where communication happens in a typical LLM."""
    print("\n" + "=" * 60)
    print("6. Communication Points in LLM Forward Pass")
    print("=" * 60)

    print("""
In a Tensor-Parallel LLM, communication happens at specific points:

┌─────────────────────────────────────────────────────────┐
│                    Decoder Layer                        │
├─────────────────────────────────────────────────────────┤
│  Input Norm (no comm)                                   │
│      │                                                  │
│  ┌───▼───────────────────────────────────────────────┐  │
│  │ QKV Projection (Column Parallel)                  │  │
│  │   - Each rank: [hidden] -> [local_qkv]            │  │
│  │   - NO COMMUNICATION                              │  │
│  └───────────────────────────────────────────────────┘  │
│      │                                                  │
│  ┌───▼───────────────────────────────────────────────┐  │
│  │ Attention (each rank on local heads)              │  │
│  │   - NO COMMUNICATION                              │  │
│  └───────────────────────────────────────────────────┘  │
│      │                                                  │
│  ┌───▼───────────────────────────────────────────────┐  │
│  │ Output Projection (Row Parallel)                  │  │
│  │   - Each rank: [local_heads] -> [hidden]          │  │
│  │   - ALL_REDUCE after linear                       │◄─── Communication!
│  └───────────────────────────────────────────────────┘  │
│      │                                                  │
│  Residual Add (no comm)                                 │
│      │                                                  │
│  MLP Norm (no comm)                                     │
│      │                                                  │
│  ┌───▼───────────────────────────────────────────────┐  │
│  │ Gate/Up Projection (Column Parallel)              │  │
│  │   - NO COMMUNICATION                              │  │
│  └───────────────────────────────────────────────────┘  │
│      │                                                  │
│  ┌───▼───────────────────────────────────────────────┐  │
│  │ Down Projection (Row Parallel)                    │  │
│  │   - ALL_REDUCE after linear                       │◄─── Communication!
│  └───────────────────────────────────────────────────┘  │
│      │                                                  │
│  Residual Add (no comm)                                 │
└─────────────────────────────────────────────────────────┘

Total per layer: 2 all_reduce operations
Total for N layers: 2N all_reduce operations
""")


if __name__ == "__main__":
    print("Mini-SGLang Distributed Communication Demo")
    print("Using REAL code from minisgl.distributed\n")

    demo_distributed_info()
    demo_distributed_impl()
    demo_communicator()
    demo_all_reduce_concept()
    demo_all_gather_concept()
    demo_communication_in_model()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. DistributedInfo tracks TP rank/size globally
   - is_primary() returns True for rank 0 (handles I/O)

2. DistributedCommunicator provides unified interface:
   - all_reduce: Sum partial results (used in row-parallel)
   - all_gather: Concatenate slices (used in vocab-parallel)

3. Two backends available:
   - TorchDistributedImpl: Standard torch.distributed
   - PyNCCLDistributedImpl: Custom NCCL bindings (lower latency)

4. Communication is minimized in TP:
   - Column parallel: No communication needed
   - Row parallel: all_reduce after linear
   - Per layer: Only 2 all_reduce ops (attn_out + mlp_down)
""")
