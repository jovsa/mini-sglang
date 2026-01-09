#!/usr/bin/env python3
"""
Learning Prototype 3: Layers and BaseOP
========================================

Demonstrates Mini-SGLang's layer building blocks:
- BaseOP: Custom module pattern (like nn.Module but simpler)
- OPList: Container for multiple BaseOPs (like nn.ModuleList)
- TP Linear layers: Column-parallel and Row-parallel patterns

Uses real code from minisgl.layers and minisgl.distributed.
"""

import torch
import torch.nn.functional as F

# Import REAL Mini-SGLang components
from minisgl.layers.base import BaseOP, OPList, StateLessOP
from minisgl.layers.linear import (
    _LinearTPImpl,
    LinearColParallelMerged,
    LinearRowParallel,
)
from minisgl.distributed.info import set_tp_info, get_tp_info, DistributedInfo


# Reset TP info for demo (normally only set once)
import minisgl.distributed.info as info_module
info_module._TP_INFO = None


def demo_baseop():
    """Demonstrate BaseOP - the foundation of all layers."""
    print("=" * 60)
    print("1. BaseOP - Custom Module Pattern")
    print("=" * 60)

    # Create a simple custom layer using BaseOP
    class SimpleLinear(BaseOP):
        def __init__(self, in_features: int, out_features: int):
            self.weight = torch.randn(out_features, in_features)
            self.bias = torch.randn(out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, self.weight, self.bias)

    layer = SimpleLinear(4, 3)

    print("\nSimpleLinear layer created:")
    print(f"  weight shape: {layer.weight.shape}")
    print(f"  bias shape: {layer.bias.shape}")

    # state_dict works like PyTorch
    state = layer.state_dict()
    print(f"\nstate_dict keys: {list(state.keys())}")

    # Forward pass
    x = torch.randn(2, 4)
    y = layer.forward(x)
    print(f"\nForward: input {x.shape} -> output {y.shape}")

    # Nested BaseOP - weights are collected recursively
    class NestedLayer(BaseOP):
        def __init__(self):
            self.linear1 = SimpleLinear(4, 8)
            self.linear2 = SimpleLinear(8, 2)

        def forward(self, x):
            return self.linear2.forward(F.relu(self.linear1.forward(x)))

    nested = NestedLayer()
    state = nested.state_dict()
    print(f"\nNested layer state_dict keys:")
    for k in state.keys():
        print(f"  {k}: {state[k].shape}")


def demo_oplist():
    """Demonstrate OPList - container for multiple layers."""
    print("\n" + "=" * 60)
    print("2. OPList - Layer Container")
    print("=" * 60)

    class SimpleLinear(BaseOP):
        def __init__(self, size: int, idx: int):
            self.weight = torch.randn(size, size)
            self.idx = idx  # Non-tensor attributes are ignored in state_dict

        def forward(self, x):
            return F.linear(x, self.weight)

    # Create list of layers (like nn.ModuleList)
    layers = OPList([SimpleLinear(4, i) for i in range(3)])

    print(f"\nOPList with {len(layers.op_list)} layers")

    # state_dict uses numeric indices
    state = layers.state_dict()
    print(f"\nstate_dict keys (numeric indices):")
    for k in state.keys():
        print(f"  {k}")

    # Forward through all layers
    x = torch.randn(2, 4)
    for i, layer in enumerate(layers.op_list):
        x = layer.forward(x)
        print(f"  After layer {i}: {x.shape}")


def demo_stateless_op():
    """Demonstrate StateLessOP - for layers without parameters."""
    print("\n" + "=" * 60)
    print("3. StateLessOP - Parameterless Layers")
    print("=" * 60)

    class ReLUActivation(StateLessOP):
        def forward(self, x):
            return F.relu(x)

    relu = ReLUActivation()

    # StateLessOP has empty state_dict
    state = relu.state_dict()
    print(f"\nStateLessOP state_dict: {state} (empty)")

    x = torch.randn(3)
    print(f"\nInput: {x.tolist()}")
    print(f"After ReLU: {relu.forward(x).tolist()}")


def demo_tp_linear():
    """Demonstrate Tensor Parallel linear layers."""
    print("\n" + "=" * 60)
    print("4. Tensor Parallel Linear Layers")
    print("=" * 60)

    # Set TP info (simulating TP=2)
    info_module._TP_INFO = None  # Reset
    set_tp_info(rank=0, size=2)
    tp = get_tp_info()

    print(f"\nTP Config: rank={tp.rank}, size={tp.size}")

    # Column Parallel: split output dimension
    # Full: [in=8, out=16] -> Each rank: [in=8, out=8]
    print("\n--- LinearColParallelMerged (Column Parallel) ---")
    col_parallel = LinearColParallelMerged(
        input_size=8,
        output_sizes=[16],  # Can merge multiple outputs
        has_bias=False,
    )
    print(f"  Full output size: {col_parallel.full_output_size}")
    print(f"  Local output size: {col_parallel.local_output_size}")
    print(f"  Weight shape: {col_parallel.weight.shape}")
    print("  -> Each rank computes different output columns")

    # Row Parallel: split input dimension
    # Full: [in=16, out=8] -> Each rank: [in=8, out=8], then all_reduce
    print("\n--- LinearRowParallel (Row Parallel) ---")
    row_parallel = LinearRowParallel(
        input_size=16,
        output_size=8,
        has_bias=False,
    )
    print(f"  Full input size: {row_parallel.full_input_size}")
    print(f"  Local input size: {row_parallel.local_input_size}")
    print(f"  Weight shape: {row_parallel.weight.shape}")
    print("  -> Each rank processes different input columns, then all_reduce")


def demo_tp_pattern():
    """Show how Column + Row parallel work together in a typical MLP."""
    print("\n" + "=" * 60)
    print("5. TP Pattern in MLP (Column -> Row)")
    print("=" * 60)

    info_module._TP_INFO = None
    set_tp_info(rank=0, size=2)

    hidden = 8
    intermediate = 16  # 2x hidden

    print(f"\nMLP: hidden={hidden}, intermediate={intermediate}, TP=2")

    # Gate/Up projection: Column parallel (no communication)
    gate_up = LinearColParallelMerged(
        input_size=hidden,
        output_sizes=[intermediate, intermediate],  # Merged gate + up
        has_bias=False,
    )

    # Down projection: Row parallel (needs all_reduce)
    down = LinearRowParallel(
        input_size=intermediate,
        output_size=hidden,
        has_bias=False,
    )

    print(f"\nGate/Up (Column Parallel):")
    print(f"  Input: [{hidden}] (full)")
    print(f"  Weight: {gate_up.weight.shape} (local)")
    print(f"  Output: [{gate_up.local_output_size}] (local, no comm needed)")

    print(f"\nDown (Row Parallel):")
    print(f"  Input: [{down.local_input_size}] (local slice)")
    print(f"  Weight: {down.weight.shape} (local)")
    print(f"  Output: [{down.local_output_size}] + all_reduce")

    print("""
Data Flow (TP=2):
┌─────────────────────────────────────────────────────────┐
│ Input [hidden=8]                                        │
├─────────────────────────────────────────────────────────┤
│              Column Parallel (no comm)                  │
│  Rank 0: W[16,8] -> out[16]    Rank 1: W[16,8] -> out[16]│
├─────────────────────────────────────────────────────────┤
│              Row Parallel + All-Reduce                  │
│  Rank 0: W[8,16] -> out[8]  ─┬─> all_reduce -> final[8] │
│  Rank 1: W[8,16] -> out[8]  ─┘                          │
└─────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    print("Mini-SGLang Layers and BaseOP Demo")
    print("Using REAL code from minisgl.layers\n")

    demo_baseop()
    demo_oplist()
    demo_stateless_op()
    demo_tp_linear()
    demo_tp_pattern()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. BaseOP is like nn.Module but simpler:
   - Tensors in __dict__ are automatically tracked
   - state_dict() / load_state_dict() work recursively

2. OPList is like nn.ModuleList:
   - Uses numeric indices in state_dict

3. StateLessOP for parameterless layers (activations, etc.)

4. Tensor Parallelism in Linear Layers:
   - Column Parallel: Split output, no communication needed
   - Row Parallel: Split input, requires all_reduce after
   - MLP pattern: Column (gate/up) -> Row (down) minimizes comm
""")
