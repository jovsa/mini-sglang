#!/usr/bin/env python3
"""
Learning Prototype 5: Attention Backends
=========================================

Demonstrates Mini-SGLang's attention backend abstraction:
- BaseAttnBackend: Interface for attention implementations
- BaseAttnMetadata: Metadata for attention computation
- HybridBackend: Different backends for prefill vs decode
- FlashAttention and FlashInfer backends

Uses real code from minisgl.attention.
"""

import torch

# Import REAL Mini-SGLang components
from minisgl.attention.base import BaseAttnBackend, BaseAttnMetadata, HybridBackend
from minisgl.attention import (
    SUPPORTED_ATTENTION_BACKENDS,
    resolve_auto_backend,
    validate_backend,
)


def demo_backend_interface():
    """Demonstrate the BaseAttnBackend interface."""
    print("=" * 60)
    print("1. BaseAttnBackend Interface")
    print("=" * 60)

    print("""
BaseAttnBackend is the abstract interface for attention:

class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(self, q, k, v, layer_id, batch) -> Tensor:
        '''Compute attention: output = softmax(QK^T / sqrt(d)) V'''
        ...

    @abstractmethod
    def prepare_metadata(self, batch) -> None:
        '''Prepare attention metadata (cu_seqlens, positions, etc.)'''
        ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len, bs_list) -> None:
        '''Initialize CUDA graph capture structures'''
        ...

    @abstractmethod
    def prepare_for_capture(self, batch) -> None:
        '''Prepare batch for CUDA graph capture'''
        ...

    @abstractmethod
    def prepare_for_replay(self, batch) -> None:
        '''Prepare batch for CUDA graph replay'''
        ...
""")


def demo_metadata():
    """Demonstrate BaseAttnMetadata."""
    print("\n" + "=" * 60)
    print("2. BaseAttnMetadata - Attention Metadata")
    print("=" * 60)

    print("""
BaseAttnMetadata stores data needed for attention computation:

@dataclass
class BaseAttnMetadata(ABC):
    positions: torch.Tensor  # Position indices for RoPE

    @abstractmethod
    def get_last_indices(self, bs: int) -> Tensor:
        '''Get indices of last token per sequence (for sampling)'''
        ...

FlashAttention adds:
  - cu_seqlens_k: Cumulative sequence lengths for keys
  - cu_seqlens_q: Cumulative sequence lengths for queries
  - cache_seqlens: Current KV cache lengths per sequence
  - max_seqlen_k/q: Maximum sequence lengths
  - page_table: Mapping from logical to physical pages
""")

    # Show a simple positions tensor
    positions = torch.tensor([0, 1, 2, 3, 4])  # Prefill positions
    print(f"\nExample positions (prefill of 5 tokens): {positions.tolist()}")

    decode_positions = torch.tensor([5, 3, 7])  # Decode positions (batch of 3)
    print(f"Example positions (decode, batch=3): {decode_positions.tolist()}")


def demo_supported_backends():
    """Show supported attention backends."""
    print("\n" + "=" * 60)
    print("3. Supported Attention Backends")
    print("=" * 60)

    supported = SUPPORTED_ATTENTION_BACKENDS.supported_names()
    print(f"\nRegistered backends: {supported}")

    print("""
Backend details:

1. FlashAttention ('fa'):
   - Based on sgl_kernel.flash_attn (SGLang's optimized FA)
   - Uses FA3 on Hopper (SM90+)
   - Efficient for prefill (long sequences)

2. FlashInfer ('fi'):
   - Based on flashinfer library
   - Optimized for decode (small batches)
   - Excellent KV cache management
""")

    # Show auto-resolution
    print("Auto-selection by GPU architecture:")
    print("  - Blackwell (SM100+): 'fi'")
    print("  - Hopper (SM90): 'fa,fi' (hybrid)")
    print("  - Pre-Hopper: 'fi'")


def demo_hybrid_backend():
    """Demonstrate HybridBackend - different backends for prefill/decode."""
    print("\n" + "=" * 60)
    print("4. HybridBackend - Prefill vs Decode")
    print("=" * 60)

    print("""
HybridBackend uses different backends for different phases:

class HybridBackend(BaseAttnBackend):
    def __init__(self, prefill_backend, decode_backend):
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(self, q, k, v, layer_id, batch):
        # Choose backend based on batch phase
        if batch.is_prefill:
            return self.prefill_backend.forward(q, k, v, layer_id, batch)
        else:
            return self.decode_backend.forward(q, k, v, layer_id, batch)
""")

    print("\nWhy use different backends?")
    print("""
┌─────────────────────────────────────────────────────────┐
│                    PREFILL PHASE                        │
├─────────────────────────────────────────────────────────┤
│ Characteristics:                                        │
│   - Long sequences (hundreds to thousands of tokens)    │
│   - Compute-bound (lots of Q*K^T operations)           │
│   - Batch size = 1 typically (one request at a time)   │
│                                                         │
│ Best backend: FlashAttention (FA3)                     │
│   - Optimized for long sequences                       │
│   - Excellent memory efficiency with tiling            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    DECODE PHASE                         │
├─────────────────────────────────────────────────────────┤
│ Characteristics:                                        │
│   - One token per request (extend_len=1)               │
│   - Memory-bound (read KV cache, little compute)       │
│   - Large batch sizes (many requests in parallel)      │
│                                                         │
│ Best backend: FlashInfer                               │
│   - Optimized for paged KV cache access                │
│   - Great for batched decode                           │
└─────────────────────────────────────────────────────────┘
""")


def demo_backend_creation():
    """Show how backends are created."""
    print("\n" + "=" * 60)
    print("5. Backend Creation Flow")
    print("=" * 60)

    print("""
Backend creation in Mini-SGLang:

1. User specifies backend via --attn argument:
   --attn auto      # Auto-select based on GPU
   --attn fa        # FlashAttention only
   --attn fi        # FlashInfer only
   --attn fa,fi     # Hybrid (FA prefill, FI decode)

2. create_attention_backend() resolves the backend:
""")

    # Show validation
    print("Validating backends:")
    for backend in ["fa", "fi", "fa,fi", "auto"]:
        result = validate_backend(backend)
        print(f"  '{backend}' -> valid")

    print("""
3. Backend initialization requires:
   - ModelConfig: head_dim, num_heads, etc.
   - BaseKVCache: The KV cache pool
   - page_table: Tensor mapping logical to physical pages
""")


def demo_forward_flow():
    """Show the attention forward pass flow."""
    print("\n" + "=" * 60)
    print("6. Attention Forward Pass")
    print("=" * 60)

    print("""
Data flow through attention:

1. AttentionLayer receives hidden states
       │
       ▼
2. QKV Projection (Column Parallel)
   hidden [batch, seq, hidden_dim]
       │
       ├─> Q [batch, seq, num_q_heads * head_dim]
       ├─> K [batch, seq, num_kv_heads * head_dim]
       └─> V [batch, seq, num_kv_heads * head_dim]
       │
       ▼
3. Apply RoPE (Rotary Position Embedding)
   Q, K = apply_rotary_emb(Q, K, positions)
       │
       ▼
4. Reshape for attention
   Q: [total_tokens, num_q_heads, head_dim]
   K: [total_tokens, num_kv_heads, head_dim]
   V: [total_tokens, num_kv_heads, head_dim]
       │
       ▼
5. Backend.forward(Q, K, V, layer_id, batch)
   ├─> Store K, V to KV cache
   ├─> Compute attention with cached K, V
   └─> Return attention output
       │
       ▼
6. Output Projection (Row Parallel + all_reduce)
   output [batch, seq, hidden_dim]
""")


def demo_kv_cache_interaction():
    """Show how attention interacts with KV cache."""
    print("\n" + "=" * 60)
    print("7. KV Cache Interaction")
    print("=" * 60)

    print("""
Inside backend.forward():

1. STORE new K, V to cache:
   kvcache.store_kv(k, v, out_loc, layer_id)

   out_loc: indices where to store (from page table)

2. READ cached K, V for attention:
   k_cache = kvcache.k_cache(layer_id)
   v_cache = kvcache.v_cache(layer_id)

3. Compute paged attention:
   - Use page_table to map logical positions to physical pages
   - cu_seqlens to handle variable-length sequences
   - cache_seqlens for current sequence lengths

Page table example (3 requests):
┌─────────────────────────────────────────────────────────┐
│ Req 0: [page_5, page_12, page_3, ...]                  │
│ Req 1: [page_8, page_1, page_9, ...]                   │
│ Req 2: [page_2, page_7, page_4, ...]                   │
└─────────────────────────────────────────────────────────┘

Each page holds one token's K and V for all layers.
""")


if __name__ == "__main__":
    print("Mini-SGLang Attention Backends Demo")
    print("Using REAL code from minisgl.attention\n")

    demo_backend_interface()
    demo_metadata()
    demo_supported_backends()
    demo_hybrid_backend()
    demo_backend_creation()
    demo_forward_flow()
    demo_kv_cache_interaction()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. BaseAttnBackend is the interface for attention implementations
   - forward(): Compute attention with KV cache
   - prepare_metadata(): Set up cu_seqlens, positions, etc.
   - CUDA graph methods for decode optimization

2. Two backends available:
   - FlashAttention (fa): Best for long prefill
   - FlashInfer (fi): Best for batched decode

3. HybridBackend combines both:
   - Automatically routes based on batch.is_prefill
   - Default on Hopper: fa,fi

4. Attention interacts with KV cache:
   - Store new K, V each forward pass
   - Read cached K, V using page table
   - Paged attention for memory efficiency
""")
