"""
Explore KV Cache Shapes and Dimensions

This interactive file demonstrates how tensor shapes transform from the transformer
block through to KV cache storage and retrieval.

Run this file to see:
1. Model configuration and dimensions
2. QKV projection shapes
3. Q, K, V split shapes
4. KV cache storage dimensions
5. How data flows through the system

Usage:
    python learning/puzzles/04_kvcache/explore_kvcache_shapes.py

To Explore Different Configurations:
    Modify the model_config in explore_kvcache_shapes() function:
    - num_layers: Number of transformer layers
    - num_qo_heads: Number of query/output attention heads
    - num_kv_heads: Number of key/value attention heads (for GQA)
    - head_dim: Dimension of each attention head
    - hidden_size: Hidden dimension of the model

    Modify batch configuration:
    - batch_size: Number of requests in batch
    - extend_len: Number of new tokens to process
    - cached_len: Number of already cached tokens

    Modify cache configuration:
    - num_pages: Total number of cache pages
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

# Try to import minisgl modules, but work without them if not available
try:
    from minisgl.distributed import set_tp_info
    from minisgl.models import ModelConfig, RotaryConfig
    from minisgl.kvcache import create_kvcache
    from minisgl.utils import divide_even
    HAS_MINISGL = True
except ImportError:
    print("Warning: minisgl not available, using mock implementations")
    HAS_MINISGL = False

    # Mock implementations for exploration
    def set_tp_info(rank=0, size=1):
        pass

    def divide_even(n, size):
        return n // size

    class ModelConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class RotaryConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def create_kvcache(model_config, num_pages, dtype, device, **kwargs):
        # Create a mock cache tensor
        local_kv_heads = divide_even(model_config.num_kv_heads, 1)
        kv_buffer = torch.empty(
            (2, model_config.num_layers, num_pages, 1, local_kv_heads, model_config.head_dim),
            device=device,
            dtype=dtype,
        )

        class MockKVCache:
            def __init__(self, buffer):
                self._kv_buffer = buffer
                self._k_buffer = buffer[0]
                self._v_buffer = buffer[1]
                self._num_layers = model_config.num_layers
                self._storage_shape = (num_pages, local_kv_heads, model_config.head_dim)

            def k_cache(self, index):
                return self._k_buffer[index]

            def v_cache(self, index):
                return self._v_buffer[index]

            @property
            def device(self):
                return device

            @property
            def dtype(self):
                return dtype

        return MockKVCache(kv_buffer)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_shape(name: str, shape: tuple, dtype: torch.dtype = None, device: str = None):
    """Print tensor shape information."""
    info = f"Shape: {shape}"
    if dtype:
        info += f", dtype: {dtype}"
    if device:
        info += f", device: {device}"
    print(f"  {name:30s} {info}")


def explore_kvcache_shapes():
    """Explore KV cache shapes step by step."""

    print_section("1. MODEL CONFIGURATION")

    # Example model configuration (similar to Llama-7B)
    model_config = ModelConfig(
        num_layers=32,
        num_qo_heads=32,
        num_kv_heads=32,  # For GQA, this would be smaller (e.g., 8)
        head_dim=128,
        hidden_size=4096,
        vocab_size=32000,
        intermediate_size=11008,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        rotary_config=RotaryConfig(
            head_dim=128,
            rotary_dim=128,
            max_position=2048,
            base=10000.0,
            scaling=None,
        ),
    )

    print("Model Configuration:")
    print(f"  num_layers:        {model_config.num_layers}")
    print(f"  num_qo_heads:     {model_config.num_qo_heads}")
    print(f"  num_kv_heads:      {model_config.num_kv_heads}")
    print(f"  head_dim:          {model_config.head_dim}")
    print(f"  hidden_size:       {model_config.hidden_size}")

    # Calculate dimensions
    qo_attn_dim = model_config.num_qo_heads * model_config.head_dim
    kv_attn_dim = model_config.num_kv_heads * model_config.head_dim
    qkv_dim = qo_attn_dim + 2 * kv_attn_dim

    print(f"\nComputed Dimensions:")
    print(f"  qo_attn_dim:       {qo_attn_dim} (num_qo_heads * head_dim)")
    print(f"  kv_attn_dim:       {kv_attn_dim} (num_kv_heads * head_dim)")
    print(f"  qkv_dim:           {qkv_dim} (qo_attn_dim + 2 * kv_attn_dim)")

    print_section("2. QKV PROJECTION SHAPES")

    # Set up tensor parallelism (single GPU for simplicity)
    set_tp_info(rank=0, size=1)
    tp_size = 1
    local_kv_heads = divide_even(model_config.num_kv_heads, tp_size)
    local_qo_heads = divide_even(model_config.num_qo_heads, tp_size)

    print(f"Tensor Parallelism: rank=0, size={tp_size}")
    print(f"  local_qo_heads:    {local_qo_heads}")
    print(f"  local_kv_heads:    {local_kv_heads}")

    # Simulate batch processing
    batch_size = 4
    extend_len = 5  # New tokens to process
    cached_len = 3  # Already cached tokens
    device_len = cached_len + extend_len  # Total sequence length

    print(f"\nBatch Configuration:")
    print(f"  batch_size:        {batch_size}")
    print(f"  extend_len:        {extend_len} (new tokens per request)")
    print(f"  cached_len:        {cached_len} (cached tokens per request)")
    print(f"  device_len:        {device_len} (total length per request)")

    # ========================================================================
    # SEQUENCE LENGTH DIMENSION EXPLANATION
    # ========================================================================
    print(f"\nSequence Length Dimension:")
    print(f"  Conceptually: Input shape is (B, T, D) where:")
    print(f"    B = batch_size = {batch_size}")
    print(f"    T = sequence_length = {extend_len} (new tokens to process)")
    print(f"    D = hidden_size = {model_config.hidden_size}")
    print(f"  Conceptual shape: ({batch_size}, {extend_len}, {model_config.hidden_size})")
    print(f"\n  However, in the actual implementation:")
    print(f"    - Sequences are flattened for efficient batch processing")
    print(f"    - Different requests can have different sequence lengths")
    print(f"    - The batch is concatenated: (B*T, D)")
    print(f"    - Sequence length info is tracked separately in metadata")
    print(f"  Actual shape: ({batch_size * extend_len}, {model_config.hidden_size})")
    print(f"\n  Why flatten?")
    print(f"    1. Efficient matrix operations on GPU")
    print(f"    2. Handles variable sequence lengths in batch")
    print(f"    3. Attention backend uses metadata for sequence info")
    print(f"    See: python/minisgl/attention/fa.py:67-105 (prepare_metadata)")

    # Create conceptual 3D tensor first to show the relationship
    hidden_states_3d = torch.randn(batch_size, extend_len, model_config.hidden_size, dtype=torch.float32)
    print_shape("hidden_states (conceptual 3D)", hidden_states_3d.shape, hidden_states_3d.dtype)

    # Flatten to actual implementation shape: (B*T, D)
    # In real code: sequences are concatenated during batch preparation
    # See: python/minisgl/scheduler/scheduler.py:180-201 (_prepare_batch)
    hidden_states = hidden_states_3d.view(batch_size * extend_len, model_config.hidden_size)
    print_shape("hidden_states (flattened for processing)", hidden_states.shape, hidden_states.dtype)
    print(f"  Note: .view({batch_size} * {extend_len}, {model_config.hidden_size}) flattens (B, T, D) ? (B*T, D)")

    # ========================================================================
    # IMPLEMENT TRANSFORMER BLOCK: QKV Projection
    # ========================================================================
    # Create QKV projection layer (simplified - no tensor parallelism)
    # In real code: python/minisgl/layers/linear.py:50-67 (LinearQKVMerged)
    GQA_ratio = model_config.num_qo_heads // model_config.num_kv_heads
    qkv_proj_weight = torch.randn(
        qkv_dim,  # output: (GQA_ratio + 2) * num_kv_heads * head_dim
        model_config.hidden_size,  # input
        dtype=torch.float32
    )
    qkv_proj_bias = None  # Typically no bias in QKV projection

    # Apply QKV projection: x @ W^T
    # In real code: python/minisgl/models/utils.py:92 (qkv_proj.forward)
    qkv_output = torch.nn.functional.linear(hidden_states, qkv_proj_weight, qkv_proj_bias)
    print_shape("qkv_proj output", qkv_output.shape, qkv_output.dtype)

    print_section("3. Q, K, V SPLIT AND RESHAPE")

    # ========================================================================
    # IMPLEMENT TRANSFORMER BLOCK: Split QKV
    # ========================================================================
    # Split QKV into separate Q, K, V tensors
    # In real code: python/minisgl/layers/attention.py:50
    qo_attn_dim_local = local_qo_heads * model_config.head_dim
    kv_attn_dim_local = local_kv_heads * model_config.head_dim

    q, k, v = qkv_output.split([qo_attn_dim_local, kv_attn_dim_local, kv_attn_dim_local], dim=-1)

    print_shape("q (after split)", q.shape, q.dtype)
    print_shape("k (after split)", k.shape, k.dtype)
    print_shape("v (after split)", v.shape, v.dtype)

    # ========================================================================
    # IMPLEMENT TRANSFORMER BLOCK: Reshape for Attention
    # ========================================================================
    # Reshape Q, K, V for multi-head attention
    # In real code: python/minisgl/layers/attention.py:57
    q_reshaped = q.view(-1, local_qo_heads, model_config.head_dim)
    k_reshaped = k.view(-1, local_kv_heads, model_config.head_dim)
    v_reshaped = v.view(-1, local_kv_heads, model_config.head_dim)

    print(f"\nAfter reshaping for attention:")
    print_shape("q (reshaped)", q_reshaped.shape, q_reshaped.dtype)
    print_shape("k (reshaped)", k_reshaped.shape, k_reshaped.dtype)
    print_shape("v (reshaped)", v_reshaped.shape, v_reshaped.dtype)

    # ========================================================================
    # IMPLEMENT TRANSFORMER BLOCK: Simplified Attention Computation
    # ========================================================================
    # For demonstration, we'll compute attention on new tokens only
    # In production, this uses FlashAttention with KV cache
    # Real code: python/minisgl/attention/fa.py:49-65

    # For this demo, we'll show a simplified attention computation
    # that demonstrates the shape transformations
    print(f"\nAttention Computation (simplified):")
    print(f"  Note: In production, this uses FlashAttention with cached K/V")
    print(f"  Here we show the shape transformations for new tokens only")

    # Scale factor for attention
    scale = 1.0 / (model_config.head_dim ** 0.5)

    # For new tokens: Q shape is conceptually (batch_size, extend_len, num_qo_heads, head_dim)
    # But flattened to: (batch_size * extend_len, num_qo_heads, head_dim)
    # In real FlashAttention, this combines cached K/V with new K/V
    # The attention backend uses metadata to track which tokens belong to which request
    q_for_attn = q_reshaped  # (batch_size * extend_len, num_qo_heads, head_dim)
    k_for_attn = k_reshaped   # (batch_size * extend_len, num_kv_heads, head_dim)
    v_for_attn = v_reshaped   # (batch_size * extend_len, num_kv_heads, head_dim)

    print(f"\n  Attention tensor shapes (flattened):")
    print(f"    Q: ({batch_size * extend_len}, {local_qo_heads}, {model_config.head_dim})")
    print(f"    K: ({batch_size * extend_len}, {local_kv_heads}, {model_config.head_dim})")
    print(f"    V: ({batch_size * extend_len}, {local_kv_heads}, {model_config.head_dim})")
    print(f"  Conceptually (if unflattened):")
    print(f"    Q: ({batch_size}, {extend_len}, {local_qo_heads}, {model_config.head_dim})")
    print(f"    K: ({batch_size}, {extend_len}, {local_kv_heads}, {model_config.head_dim})")
    print(f"    V: ({batch_size}, {extend_len}, {local_kv_heads}, {model_config.head_dim})")

    # Compute attention scores: Q @ K^T
    # Shape: (batch_size * extend_len, num_qo_heads, num_kv_heads)
    # Note: In GQA, num_qo_heads may be > num_kv_heads
    attn_scores = torch.einsum('bqd,bkd->bqk', q_for_attn, k_for_attn) * scale
    print_shape("attn_scores (Q @ K^T)", attn_scores.shape, attn_scores.dtype)

    # Apply softmax (causal mask would be applied here in production)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    print_shape("attn_probs (softmax)", attn_probs.shape, attn_probs.dtype)

    # Apply attention to values: attn_probs @ V
    # Shape: (batch_size * extend_len, num_qo_heads, head_dim)
    attn_output = torch.einsum('bqk,bkd->bqd', attn_probs, v_for_attn)
    print_shape("attn_output (probs @ V)", attn_output.shape, attn_output.dtype)

    # ========================================================================
    # IMPLEMENT TRANSFORMER BLOCK: Output Projection
    # ========================================================================
    # Reshape attention output back to flat format
    # In real code: python/minisgl/layers/attention.py:59
    attn_output_flat = attn_output.view(-1, qo_attn_dim_local)
    print_shape("attn_output (flattened)", attn_output_flat.shape, attn_output_flat.dtype)

    # Apply output projection: O_proj
    # In real code: python/minisgl/models/utils.py:84-88, 95
    o_proj_weight = torch.randn(
        model_config.hidden_size,  # output
        qo_attn_dim_local,  # input: num_qo_heads * head_dim
        dtype=torch.float32
    )
    o_proj_bias = None

    # Output projection: attn_output @ O_proj^T
    attn_final = torch.nn.functional.linear(attn_output_flat, o_proj_weight, o_proj_bias)
    print_shape("attn_final (after O_proj)", attn_final.shape, attn_final.dtype)

    # ========================================================================
    # IMPLEMENT TRANSFORMER BLOCK: Residual Connection and MLP
    # ========================================================================
    print(f"\nComplete Transformer Block Flow:")

    # Residual connection after attention
    # In real code: python/minisgl/models/qwen3.py:38 (x = x + self_attn(x))
    attn_with_residual = hidden_states + attn_final
    print_shape("attn + residual", attn_with_residual.shape, attn_with_residual.dtype)

    # Post-attention layer norm (simplified - just showing shape)
    # In real code: python/minisgl/models/qwen3.py:39 (post_attention_layernorm)
    mlp_input = attn_with_residual  # Would apply layer norm here
    print_shape("mlp_input (after norm)", mlp_input.shape, mlp_input.dtype)

    # MLP: Gate and Up projection
    # In real code: python/minisgl/models/utils.py:24-28 (gate_up_proj)
    gate_up_proj_weight = torch.randn(
        2 * model_config.intermediate_size,  # output: [gate, up]
        model_config.hidden_size,  # input
        dtype=torch.float32
    )
    gate_up = torch.nn.functional.linear(mlp_input, gate_up_proj_weight, None)
    gate, up = gate_up.split([model_config.intermediate_size, model_config.intermediate_size], dim=-1)
    print_shape("gate", gate.shape, gate.dtype)
    print_shape("up", up.shape, up.dtype)

    # Activation: SiLU(gate) * up
    # In real code: python/minisgl/models/utils.py:32, 46 (silu_and_mul)
    activated = torch.nn.functional.silu(gate) * up
    print_shape("mlp_activated (SiLU(gate) * up)", activated.shape, activated.dtype)

    # MLP: Down projection
    # In real code: python/minisgl/models/utils.py:36-40, 48 (down_proj)
    down_proj_weight = torch.randn(
        model_config.hidden_size,  # output
        model_config.intermediate_size,  # input
        dtype=torch.float32
    )
    mlp_output = torch.nn.functional.linear(activated, down_proj_weight, None)
    print_shape("mlp_output (after down_proj)", mlp_output.shape, mlp_output.dtype)

    # Final residual connection
    # In real code: python/minisgl/models/qwen3.py:40 (x = x + mlp(x))
    final_output = attn_with_residual + mlp_output
    print_shape("final_output (block output)", final_output.shape, final_output.dtype)

    print(f"\n  Complete transformer block:")
    print(f"    1. Input: hidden_states")
    print(f"    2. QKV projection -> Q, K, V")
    print(f"    3. Attention computation -> attn_output")
    print(f"    4. Output projection -> attn_final")
    print(f"    5. Residual: hidden_states + attn_final")
    print(f"    6. MLP (gate_up -> activation -> down) -> mlp_output")
    print(f"    7. Final residual: (hidden_states + attn_final) + mlp_output")

    print_section("4. KV CACHE STORAGE DIMENSIONS")

    # Create KV cache
    num_pages = 1000
    # Use CPU for compatibility (production uses CUDA)
    device = torch.device("cpu")
    dtype = torch.float32  # Production uses float16 on GPU

    print(f"KV Cache Configuration:")
    print(f"  num_pages:         {num_pages}")
    print(f"  device:            {device}")
    print(f"  dtype:             {dtype}")

    kv_cache = create_kvcache(
        model_config=model_config,
        num_pages=num_pages,
        dtype=dtype,
        device=device,
    )

    # KV cache structure
    print(f"\nKV Cache Buffer Structure:")
    print(f"  Full buffer shape: {kv_cache._kv_buffer.shape}")
    print(f"    - 2: K cache (0) and V cache (1)")
    print(f"    - {model_config.num_layers}: num_layers")
    print(f"    - {num_pages}: num_pages")
    print(f"    - 1: page_size (tokens per page)")
    print(f"    - {local_kv_heads}: local_kv_heads")
    print(f"    - {model_config.head_dim}: head_dim")

    # Per-layer cache access
    layer_id = 0
    k_cache_layer = kv_cache.k_cache(layer_id)
    v_cache_layer = kv_cache.v_cache(layer_id)

    print(f"\nPer-Layer Cache Access (layer_id={layer_id}):")
    print_shape("k_cache[layer_id]", k_cache_layer.shape, k_cache_layer.dtype, str(device))
    print_shape("v_cache[layer_id]", v_cache_layer.shape, v_cache_layer.dtype, str(device))

    # Storage shape (what store_kv uses)
    storage_shape = (num_pages, local_kv_heads, model_config.head_dim)
    print(f"\nStorage Shape (for store_kv):")
    print(f"  storage_shape:     {storage_shape}")
    print(f"    - {num_pages}: num_pages")
    print(f"    - {local_kv_heads}: local_kv_heads")
    print(f"    - {model_config.head_dim}: head_dim")

    print_section("5. STORING K AND V TO CACHE")

    # Simulate storing new K and V
    # k and v shapes: (extend_len, local_kv_heads, head_dim)
    k_new = torch.randn(extend_len, local_kv_heads, model_config.head_dim,
                        device=device, dtype=dtype)
    v_new = torch.randn(extend_len, local_kv_heads, model_config.head_dim,
                        device=device, dtype=dtype)

    print("New K and V to store:")
    print_shape("k (new tokens)", k_new.shape, k_new.dtype, str(device))
    print_shape("v (new tokens)", v_new.shape, v_new.dtype, str(device))

    # Page indices for new tokens
    out_loc = torch.tensor([10, 11, 12, 13, 14], device=device, dtype=torch.int32)
    print(f"\nPage indices (out_loc): {out_loc.cpu().tolist()}")
    print(f"  These are the cache pages where K and V will be stored")

    print(f"\nStorage Operation:")
    print(f"  kv_cache.store_kv(k, v, out_loc, layer_id)")
    print(f"  - k shape: {k_new.shape}")
    print(f"  - v shape: {v_new.shape}")
    print(f"  - out_loc: {out_loc.shape} (page indices)")
    print(f"  - layer_id: {layer_id}")
    print(f"  - Writes to k_cache[{layer_id}] and v_cache[{layer_id}]")
    print(f"  - Uses CUDA kernel to scatter k/v to pages specified by out_loc")

    print_section("6. RETRIEVING CACHED K AND V FOR ATTENTION")

    # For attention, we need:
    # - Q: new tokens only (extend_len, num_qo_heads, head_dim)
    # - K: all tokens (cached + new) retrieved from cache
    # - V: all tokens (cached + new) retrieved from cache

    print("Attention Computation Requirements:")
    print(f"  Q (new tokens): computed from hidden_states via QKV projection")
    print_shape("q (new tokens only)", q_reshaped[:extend_len].shape, q_reshaped.dtype)

    print(f"\nK and V Retrieval:")
    print(f"  k_cache = kv_cache.k_cache(layer_id)")
    print(f"  v_cache = kv_cache.v_cache(layer_id)")
    print_shape("k_cache (all pages)", k_cache_layer.shape, k_cache_layer.dtype, str(device))
    print_shape("v_cache (all pages)", v_cache_layer.shape, v_cache_layer.dtype, str(device))

    # Page table maps token positions to cache pages
    max_seq_len = 128
    page_table = torch.zeros((batch_size, max_seq_len), dtype=torch.int32, device=device)

    # Example: Request 0 has tokens at pages [5, 6, 7, 10, 11, 12, 13, 14]
    # (pages 5,6,7 are cached, pages 10-14 are new)
    page_table[0, :device_len] = torch.tensor([5, 6, 7, 10, 11, 12, 13, 14], device=device)

    print(f"\nPage Table:")
    print(f"  Shape: {page_table.shape}")
    print(f"  Maps token positions ? cache page indices")
    print(f"  Example for request 0:")
    for i in range(device_len):
        page_idx = page_table[0, i].item()
        print(f"    Token position {i:2d} ? Page {page_idx:2d}")

    print(f"\nAttention Kernel Usage:")
    print(f"  flash_attn_with_kvcache(")
    print(f"    q=q,  # Shape: (extend_len, num_qo_heads, head_dim)")
    print(f"    k_cache=k_cache,  # Shape: (num_pages, 1, local_kv_heads, head_dim)")
    print(f"    v_cache=v_cache,  # Shape: (num_pages, 1, local_kv_heads, head_dim)")
    print(f"    page_table=page_table,  # Maps positions ? pages")
    print(f"    cache_seqlens=[device_len],  # Total cached length per request")
    print(f"    ...")
    print(f"  )")
    print(f"  - Kernel uses page_table to gather K/V from correct pages")
    print(f"  - Combines cached K/V with new K/V for attention computation")

    print_section("7. MEMORY SIZE CALCULATION")

    # Calculate cache size per page
    cache_per_page = (
        2  # key + value
        * model_config.head_dim
        * local_kv_heads
        * 1  # page_size
        * dtype.itemsize  # bytes per element
        * model_config.num_layers
    )

    total_cache_size = num_pages * cache_per_page

    print(f"Cache Size Calculation:")
    print(f"  cache_per_page = 2 � head_dim � num_kv_heads � page_size � dtype_size � num_layers")
    print(f"  cache_per_page = 2 � {model_config.head_dim} � {local_kv_heads} � 1 � {dtype.itemsize} � {model_config.num_layers}")
    print(f"  cache_per_page = {cache_per_page:,} bytes = {cache_per_page / 1024:.2f} KB")
    print(f"\n  Total cache size = {num_pages} � {cache_per_page:,} bytes")
    print(f"  Total cache size = {total_cache_size:,} bytes = {total_cache_size / (1024**2):.2f} MB = {total_cache_size / (1024**3):.2f} GB")

    print_section("8. SUMMARY: SHAPE TRANSFORMATIONS")

    print("Complete Shape Flow:")
    print()
    print("1. Input Hidden States:")
    print(f"   Conceptual:    ({batch_size}, {extend_len}, {model_config.hidden_size})  [B, T, D]")
    print(f"   Flattened:     ({batch_size * extend_len}, {model_config.hidden_size})  [B*T, D]")
    print()
    print("2. QKV Projection:")
    print(f"   Conceptual:    ({batch_size}, {extend_len}, {qkv_dim})  [B, T, QKV_dim]")
    print(f"   Flattened:     ({batch_size * extend_len}, {qkv_dim})  [B*T, QKV_dim]")
    print()
    print("3. Split Q, K, V:")
    print(f"   q:             ({batch_size * extend_len}, {qo_attn_dim_local})  [B*T, Q_dim]")
    print(f"   k:             ({batch_size * extend_len}, {kv_attn_dim_local})  [B*T, K_dim]")
    print(f"   v:             ({batch_size * extend_len}, {kv_attn_dim_local})  [B*T, V_dim]")
    print()
    print("4. Reshape for Attention:")
    print(f"   Conceptual:    ({batch_size}, {extend_len}, {local_qo_heads}, {model_config.head_dim})  [B, T, H_q, D]")
    print(f"   Flattened:     ({batch_size * extend_len}, {local_qo_heads}, {model_config.head_dim})  [B*T, H_q, D]")
    print(f"   k:             ({batch_size * extend_len}, {local_kv_heads}, {model_config.head_dim})  [B*T, H_kv, D]")
    print(f"   v:             ({batch_size * extend_len}, {local_kv_heads}, {model_config.head_dim})  [B*T, H_kv, D]")
    print()
    print("5. KV Cache Storage:")
    print(f"   k_cache:       ({num_pages}, 1, {local_kv_heads}, {model_config.head_dim})")
    print(f"   v_cache:       ({num_pages}, 1, {local_kv_heads}, {model_config.head_dim})")
    print(f"   (per layer, total: {model_config.num_layers} layers)")
    print()
    print("6. Attention Retrieval:")
    print(f"   q (new, per request): ({extend_len}, {local_qo_heads}, {model_config.head_dim})  [T, H_q, D]")
    print(f"   q (batch, flattened): ({batch_size * extend_len}, {local_qo_heads}, {model_config.head_dim})  [B*T, H_q, D]")
    print(f"   k (all):       Retrieved from k_cache via page_table")
    print(f"   v (all):       Retrieved from v_cache via page_table")
    print(f"   (page_table maps {device_len} token positions → cache pages)")
    print(f"   Note: In batch, Q is flattened but attention processes per-request sequences")
    print(f"         separately using cu_seqlens metadata")

    print("\n" + "=" * 80)
    print("Exploration complete! Modify the configuration above to explore different shapes.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    explore_kvcache_shapes()
