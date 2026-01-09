#!/usr/bin/env python3
"""
Learning Prototype 10: Engine
==============================

Demonstrates Mini-SGLang's Engine class:
- Engine initialization and component setup
- Model loading and weight sharding
- KV cache allocation
- Forward batch execution
- CUDA graph integration

Uses real code concepts from minisgl.engine.
"""

import torch

# Import REAL Mini-SGLang components
from minisgl.engine.config import EngineConfig
from minisgl.engine.graph import mem_GB, get_free_memory
from minisgl.engine.sample import Sampler, BatchSamplingArgs


def demo_engine_overview():
    """Overview of Engine class."""
    print("=" * 60)
    print("1. Engine Overview")
    print("=" * 60)

    print("""
Engine is the core component that runs on each GPU:

class Engine:
    def __init__(self, config: EngineConfig):
        # 1. Set up distributed (TP rank, NCCL)
        # 2. Load model weights (sharded for TP)
        # 3. Allocate KV cache
        # 4. Create attention backend
        # 5. Set up CUDA graphs

    def forward_batch(batch, args) -> ForwardOutput:
        # Run model forward pass
        # Sample next tokens
        # Return results

    def shutdown():
        # Clean up resources

Key attributes:
  - model: The LLM model
  - kv_cache: MHAKVCache for K/V storage
  - attn_backend: FlashAttention/FlashInfer
  - graph_runner: CUDA graph manager
  - sampler: Token sampling
  - ctx: Global Context
""")


def demo_engine_config():
    """Show EngineConfig structure."""
    print("\n" + "=" * 60)
    print("2. EngineConfig - Engine Configuration")
    print("=" * 60)

    print("""
EngineConfig contains all engine settings:

@dataclass
class EngineConfig:
    # Model settings
    model_path: str           # HuggingFace model path
    model_config: ModelConfig # Model architecture config
    dtype: torch.dtype        # float16, bfloat16

    # TP settings
    tp_info: DistributedInfo  # rank, size
    distributed_addr: str     # for torch.distributed
    distributed_timeout: int
    use_pynccl: bool

    # Memory settings
    max_running_req: int      # Max concurrent requests
    max_seq_len: int          # Max sequence length
    memory_ratio: float       # % of GPU memory for KV cache
    num_page_override: int    # Override page count

    # Performance settings
    attention_backend: str    # "auto", "fa", "fi", "fa,fi"
    cuda_graph_bs: List[int]  # Batch sizes for graphs
    cuda_graph_max_bs: int    # Max batch size for graphs
    max_forward_len: int      # Max tokens per forward

    # Other
    use_dummy_weight: bool    # For testing
    page_size: int            # Usually 1
    offline_mode: bool
""")


def demo_initialization_flow():
    """Show Engine initialization flow."""
    print("\n" + "=" * 60)
    print("3. Engine Initialization Flow")
    print("=" * 60)

    print("""
Engine.__init__(config) does the following:

1. Set TP info and CUDA device
   ┌─────────────────────────────────────────────────────┐
   │ set_tp_info(rank=config.tp_info.rank, ...)         │
   │ self.device = torch.device(f"cuda:{rank}")         │
   │ torch.cuda.set_device(self.device)                 │
   └─────────────────────────────────────────────────────┘

2. Initialize distributed communication
   ┌─────────────────────────────────────────────────────┐
   │ torch.distributed.init_process_group(...)          │
   │ enable_pynccl_distributed(...)  # if using pynccl  │
   └─────────────────────────────────────────────────────┘

3. Load model (on meta device first, then real weights)
   ┌─────────────────────────────────────────────────────┐
   │ with torch.device("meta"):                         │
   │     self.model = create_model(path, config)        │
   │ self.model.load_state_dict(load_hf_weight(...))   │
   └─────────────────────────────────────────────────────┘

4. Determine KV cache size based on free memory
   ┌─────────────────────────────────────────────────────┐
   │ available = free_memory * memory_ratio - model_mem │
   │ num_pages = available // cache_per_page            │
   └─────────────────────────────────────────────────────┘

5. Create KV cache and attention backend
   ┌─────────────────────────────────────────────────────┐
   │ self.kv_cache = create_kvcache(...)               │
   │ self.page_table = create_page_table(...)          │
   │ self.attn_backend = create_attention_backend(...) │
   └─────────────────────────────────────────────────────┘

6. Create Context and set global
   ┌─────────────────────────────────────────────────────┐
   │ self.ctx = Context(page_size=1, attn_backend=...)  │
   │ set_global_ctx(self.ctx)                           │
   └─────────────────────────────────────────────────────┘

7. Capture CUDA graphs for decode
   ┌─────────────────────────────────────────────────────┐
   │ self.graph_runner = GraphRunner(...)               │
   │ # Captures graphs for bs=[1,2,4,8,16,...]         │
   └─────────────────────────────────────────────────────┘
""")


def demo_kv_cache_sizing():
    """Show KV cache size calculation."""
    print("\n" + "=" * 60)
    print("4. KV Cache Size Calculation")
    print("=" * 60)

    # Example values
    num_layers = 32
    num_kv_heads = 8
    head_dim = 128
    tp_size = 2
    dtype_bytes = 2  # bfloat16

    local_kv_heads = num_kv_heads // tp_size  # 4 per rank

    cache_per_page = (
        2  # K + V
        * head_dim
        * local_kv_heads
        * 1  # page_size
        * dtype_bytes
        * num_layers
    )

    print(f"""
KV cache size calculation:

Model config:
  num_layers: {num_layers}
  num_kv_heads: {num_kv_heads}
  head_dim: {head_dim}
  TP size: {tp_size}
  dtype: bfloat16 ({dtype_bytes} bytes)

Per-rank calculation:
  local_kv_heads = {num_kv_heads} / {tp_size} = {local_kv_heads}

  cache_per_page = 2 * head_dim * local_kv_heads * num_layers * dtype_bytes
                 = 2 * {head_dim} * {local_kv_heads} * {num_layers} * {dtype_bytes}
                 = {cache_per_page} bytes/page ({cache_per_page/1024:.1f} KB)

Example sizing (16GB available):
  num_pages = 16GB / {cache_per_page/1024:.1f}KB ≈ {int(16*1024*1024*1024 / cache_per_page):,} pages
""")


def demo_forward_batch():
    """Show forward_batch execution."""
    print("\n" + "=" * 60)
    print("5. Forward Batch Execution")
    print("=" * 60)

    print("""
Engine.forward_batch(batch, args) -> ForwardOutput:

def forward_batch(self, batch: Batch, args: BatchSamplingArgs):
    # 1. Enter batch context
    with self.ctx.forward_batch(batch):

        # 2. Check if can use CUDA graph (decode + valid size)
        if self.graph_runner.can_use_cuda_graph(batch):
            # Replay captured graph
            logits = self.graph_runner.replay(batch)
        else:
            # Normal forward pass
            logits = self.model.forward()

    # 3. Mark tokens as computed
    for req in batch.reqs:
        req.complete_one()

    # 4. Sample next tokens
    next_tokens_gpu = self.sampler.sample(logits[:batch.size], args)

    # 5. Copy to CPU asynchronously
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
    copy_done_event = torch.cuda.Event()
    copy_done_event.record()

    # 6. Return results
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

ForwardOutput contains:
  - next_tokens_gpu: Tokens on GPU (for KV cache update)
  - next_tokens_cpu: Tokens on CPU (for detokenization)
  - copy_done_event: CUDA event for synchronization
""")


def demo_sampler():
    """Demonstrate the Sampler."""
    print("\n" + "=" * 60)
    print("6. Sampler - Token Sampling")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        sampler = Sampler(device, vocab_size=32000)
        print(f"Created Sampler on {device}")
    else:
        print("(Skipping GPU demo - CUDA not available)")
        return

    print("""
Sampler handles token selection:

class Sampler:
    def __init__(self, device, vocab_size):
        # Pre-allocate random buffer for sampling
        self.uniform_samples = torch.empty(max_bs, device=device)

    def sample(logits, args: BatchSamplingArgs) -> Tensor:
        # 1. Apply temperature scaling
        # 2. Apply top-k filtering
        # 3. Apply top-p (nucleus) filtering
        # 4. Sample from distribution

BatchSamplingArgs:
  - temperatures: Tensor[batch_size]
  - top_ks: Tensor[batch_size]
  - top_ps: Tensor[batch_size]
  - greedy_mask: Tensor[batch_size] (bool)
""")


def demo_model_forward():
    """Show what happens in model.forward()."""
    print("\n" + "=" * 60)
    print("7. Model Forward Pass")
    print("=" * 60)

    print("""
model.forward() executes the transformer:

1. Get batch info from global context
   ctx = get_global_ctx()
   batch = ctx.batch

2. Get input embeddings
   input_ids = batch.input_ids  # [total_tokens]
   hidden = self.embed(input_ids)

3. Run through decoder layers
   for i, layer in enumerate(self.layers):
       hidden = layer.forward(hidden, i)

   Each layer:
   ├── Input LayerNorm
   ├── Attention (QKV proj -> attention -> output proj)
   │   └── Uses ctx.batch.attn_metadata
   ├── Residual add
   ├── Post LayerNorm
   ├── MLP (gate/up -> activation -> down)
   └── Residual add

4. Final LayerNorm
   hidden = self.final_norm(hidden)

5. LM head projection
   logits = self.lm_head(hidden)

6. Select last token per sequence
   last_indices = batch.attn_metadata.get_last_indices(batch.size)
   return logits[last_indices]  # [batch_size, vocab_size]
""")


def demo_context_manager():
    """Show how Context manages batch."""
    print("\n" + "=" * 60)
    print("8. Context - Global Batch Management")
    print("=" * 60)

    print("""
Context holds the current batch for model access:

@dataclass
class Context:
    page_size: int
    attn_backend: BaseAttnBackend
    _batch: Batch | None = None

    @property
    def batch(self) -> Batch:
        return self._batch  # Used by layers during forward

    @contextmanager
    def forward_batch(self, batch: Batch):
        self._batch = batch
        yield
        self._batch = None

Usage in Engine:
  with self.ctx.forward_batch(batch):
      logits = self.model.forward()

Usage in layers (e.g., AttentionLayer):
  ctx = get_global_ctx()
  batch = ctx.batch
  metadata = batch.attn_metadata
  positions = metadata.positions  # For RoPE
  # ... use metadata for attention ...
""")


if __name__ == "__main__":
    print("Mini-SGLang Engine Demo")
    print("Using concepts from minisgl.engine\n")

    demo_engine_overview()
    demo_engine_config()
    demo_initialization_flow()
    demo_kv_cache_sizing()
    demo_forward_batch()
    demo_sampler()
    demo_model_forward()
    demo_context_manager()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. Engine is the core GPU worker
   - One Engine per GPU (per TP rank)
   - Owns model, KV cache, attention backend

2. Initialization:
   - Sets up distributed communication
   - Loads model with TP weight sharding
   - Sizes KV cache based on available memory
   - Captures CUDA graphs for decode

3. Forward pass:
   - Uses CUDA graph when possible (decode)
   - Falls back to normal forward (prefill)
   - Samples next tokens on GPU
   - Returns both GPU and CPU tokens

4. Context pattern:
   - Global context holds current batch
   - Layers access batch via get_global_ctx()
   - Enables clean API without passing batch everywhere

5. Memory management:
   - KV cache sized to use available GPU memory
   - Page-based allocation for flexibility
   - TP shards KV heads across GPUs
""")
