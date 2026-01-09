#!/usr/bin/env python3
"""
Learning Prototype 11: Component Connections (End-to-End)
==========================================================

Demonstrates how all Mini-SGLang components connect:
- Process architecture and data flow
- Request lifecycle from user to response
- Internal component interactions
- The LLM class as a simplified interface

This prototype ties together all previous concepts.
"""

import torch

# Import REAL Mini-SGLang components
from minisgl.core import SamplingParams, Req, Batch
from minisgl.message import UserMsg, DetokenizeMsg


def demo_process_architecture():
    """Show the multi-process architecture."""
    print("=" * 60)
    print("1. Process Architecture")
    print("=" * 60)

    print("""
Mini-SGLang runs as multiple processes:

┌─────────────────────────────────────────────────────────┐
│                    MAIN PROCESS                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │           API Server (FastAPI)                   │    │
│  │  - HTTP endpoint /v1/chat/completions           │    │
│  │  - Manages async request queues                 │    │
│  │  - Streams responses via SSE                    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
            │ ZMQ                           ▲ ZMQ
            ▼                               │
┌─────────────────────────────────────────────────────────┐
│                 TOKENIZER WORKERS                        │
│  ┌──────────────┐              ┌──────────────┐         │
│  │  Tokenizer   │              │ Detokenizer  │         │
│  │  (1-N procs) │              │  (1 proc)    │         │
│  │  Text→Tokens │              │  Token→Text  │         │
│  └──────────────┘              └──────────────┘         │
└─────────────────────────────────────────────────────────┘
            │ ZMQ                           ▲ ZMQ
            ▼                               │
┌─────────────────────────────────────────────────────────┐
│              SCHEDULER WORKERS (1 per GPU)               │
│  ┌──────────────────────────────────────────────────┐   │
│  │               Scheduler Rank 0                    │   │
│  │  - Receives msgs from tokenizer                   │   │
│  │  - Broadcasts to other ranks                      │   │
│  │  - Sends results to detokenizer                   │   │
│  │  ┌────────────────────────────────────────────┐  │   │
│  │  │                 Engine                     │  │   │
│  │  │  - Model, KV Cache, Attention, Graphs     │  │   │
│  │  └────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                        │ NCCL ▲                         │
│                        ▼      │                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Scheduler Rank 1, 2, ...               │   │
│  │  - Receives msgs from Rank 0                      │   │
│  │  - Runs same computation                          │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

Process startup (in launch_server):
  1. Start Scheduler processes (one per TP rank)
  2. Start Detokenizer process
  3. Start Tokenizer processes
  4. Wait for all to be ready
  5. Start API server (main process)
""")


def demo_request_lifecycle():
    """Show complete request lifecycle."""
    print("\n" + "=" * 60)
    print("2. Request Lifecycle")
    print("=" * 60)

    print("""
Complete flow of a user request:

1. USER SENDS REQUEST
   ┌─────────────────────────────────────────────────────┐
   │ POST /v1/chat/completions                           │
   │ {                                                    │
   │   "model": "Qwen/Qwen3-0.6B",                       │
   │   "messages": [{"role": "user", "content": "Hi"}],  │
   │   "max_tokens": 100                                 │
   │ }                                                    │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
2. API SERVER CREATES REQUEST
   ┌─────────────────────────────────────────────────────┐
   │ - Assign unique request ID (uid)                    │
   │ - Parse SamplingParams                              │
   │ - Send TokenizeMsg to tokenizer                     │
   │ - Create async queue for response streaming         │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
3. TOKENIZER PROCESSES TEXT
   ┌─────────────────────────────────────────────────────┐
   │ - Apply chat template                               │
   │ - Convert text to token IDs                         │
   │ - Create UserMsg with input_ids                     │
   │ - Send to Scheduler                                 │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
4. SCHEDULER RECEIVES REQUEST
   ┌─────────────────────────────────────────────────────┐
   │ PrefillManager.add_one_req(user_msg)               │
   │ - Add to pending_list                               │
   │ - Match prefix in RadixCache                        │
   │ - Allocate table entry                              │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
5. PREFILL BATCH CREATED
   ┌─────────────────────────────────────────────────────┐
   │ batch = prefill_manager.schedule_next_batch()       │
   │ - Allocate KV cache pages                           │
   │ - Prepare attention metadata                        │
   │ - May chunk if too long                             │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
6. ENGINE FORWARD (PREFILL)
   ┌─────────────────────────────────────────────────────┐
   │ output = engine.forward_batch(prefill_batch, args)  │
   │ - No CUDA graph (variable length)                   │
   │ - Store all K,V to cache                            │
   │ - Sample first token                                │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
7. ADD TO DECODE MANAGER
   ┌─────────────────────────────────────────────────────┐
   │ decode_manager.add_reqs(prefill_batch.reqs)         │
   │ - Request now in decode phase                       │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
8. DECODE LOOP (repeats until done)
   ┌─────────────────────────────────────────────────────┐
   │ batch = decode_manager.schedule_next_batch()        │
   │ graph_runner.pad_batch(batch)  # Pad for graph      │
   │ output = engine.forward_batch(decode_batch, args)   │
   │ - Uses CUDA graph                                   │
   │ - Generates one token per request                   │
   │                                                     │
   │ For each request:                                   │
   │   req.append_host(token)                           │
   │   scheduler.send_result(DetokenizeMsg(...))        │
   │   if req.finished or !req.can_decode():            │
   │       decode_manager.remove_req(req)               │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
9. DETOKENIZER SENDS RESPONSE
   ┌─────────────────────────────────────────────────────┐
   │ - Receive DetokenizeMsg                             │
   │ - Decode token to text                              │
   │ - Send to API server                                │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
10. API SERVER STREAMS TO USER
    ┌─────────────────────────────────────────────────────┐
    │ - SSE stream: data: {"content": "Hello"}           │
    │ - Final: data: [DONE]                              │
    └─────────────────────────────────────────────────────┘
""")


def demo_component_connections():
    """Show internal component connections."""
    print("\n" + "=" * 60)
    print("3. Internal Component Connections")
    print("=" * 60)

    print("""
Within a Scheduler, components connect as follows:

┌──────────────────────────────────────────────────────────┐
│                      SCHEDULER                            │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │              SchedulerIOMixin                     │    │
│  │  - ZMQ sockets for tokenizer comm                │    │
│  │  - Broadcast for multi-rank                      │    │
│  └──────────────────────────────────────────────────┘    │
│              │                      │                     │
│              ▼                      ▼                     │
│  ┌──────────────────┐    ┌──────────────────┐            │
│  │  PrefillManager  │    │  DecodeManager   │            │
│  │  - pending_list  │    │  - running_reqs  │            │
│  └──────────────────┘    └──────────────────┘            │
│         │    │    │              │                        │
│         ▼    │    └──────────────┼─────────┐             │
│  ┌──────────────────┐            │         │             │
│  │   CacheManager   │◄───────────┘         │             │
│  │  - RadixCache    │                      │             │
│  └──────────────────┘                      │             │
│              │                             │             │
│              ▼                             ▼             │
│  ┌──────────────────┐           ┌──────────────────┐    │
│  │   TableManager   │           │      ENGINE      │    │
│  │  - page_table    │──────────►│  - model         │    │
│  │  - token_pool    │           │  - kv_cache      │    │
│  └──────────────────┘           │  - attn_backend  │    │
│                                 │  - graph_runner  │    │
│                                 │  - sampler       │    │
│                                 └──────────────────┘    │
└──────────────────────────────────────────────────────────┘

Data flow:
1. UserMsg → PrefillManager (new request)
2. PrefillManager → CacheManager (match prefix)
3. PrefillManager → TableManager (allocate entry)
4. PrefillManager → Engine (prefill forward)
5. PrefillManager → DecodeManager (after prefill)
6. DecodeManager → Engine (decode forward)
7. Engine → Sampler (token selection)
8. SchedulerIOMixin → Detokenizer (results)
""")


def demo_engine_internal():
    """Show Engine internal connections."""
    print("\n" + "=" * 60)
    print("4. Engine Internal Connections")
    print("=" * 60)

    print("""
Inside Engine, components interact during forward:

┌──────────────────────────────────────────────────────────┐
│                        ENGINE                             │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │                     Context                         │  │
│  │  ┌──────────────┐      ┌──────────────────────┐   │  │
│  │  │    Batch     │      │    attn_backend      │   │  │
│  │  │ (current)    │      │ (HybridBackend)      │   │  │
│  │  └──────────────┘      └──────────────────────┘   │  │
│  └────────────────────────────────────────────────────┘  │
│         │                          │                      │
│         ▼                          ▼                      │
│  ┌──────────────────────────────────────────────────┐    │
│  │                     MODEL                         │    │
│  │  ┌────────────────────────────────────────────┐  │    │
│  │  │              DecoderLayer × N               │  │    │
│  │  │  ┌──────────────────┐  ┌───────────────┐   │  │    │
│  │  │  │  AttentionLayer  │  │      MLP      │   │  │    │
│  │  │  │  uses Context    │  │  all_reduce   │   │  │    │
│  │  │  └────────┬─────────┘  └───────────────┘   │  │    │
│  │  └───────────┼───────────────────────────────┘  │    │
│  └──────────────┼──────────────────────────────────┘    │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │              KV CACHE SYSTEM                      │   │
│  │  ┌────────────────┐    ┌────────────────────┐    │   │
│  │  │   MHAKVCache   │◄───│  FlashAttention/   │    │   │
│  │  │   (storage)    │    │  FlashInfer        │    │   │
│  │  └────────────────┘    └────────────────────┘    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │               CUDA GRAPHS                         │   │
│  │  GraphRunner.replay() for decode                 │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
""")


def demo_llm_interface():
    """Show the LLM class as simplified interface."""
    print("\n" + "=" * 60)
    print("5. LLM Class - Simplified Interface")
    print("=" * 60)

    print("""
LLM class wraps everything for easy offline usage:

from minisgl.llm import LLM
from minisgl.core import SamplingParams

# Create LLM (starts all internal components)
llm = LLM(model_path="Qwen/Qwen3-0.6B")

# Generate text
results = llm.generate(
    prompts=["Hello, world!", "What is 2+2?"],
    sampling_params=SamplingParams(max_tokens=100),
)

for result in results:
    print(result["text"])

Under the hood:
┌─────────────────────────────────────────────────────────┐
│ LLM inherits from Scheduler                             │
│   - Creates Engine internally                           │
│   - Manages prefill/decode loop                         │
│   - Handles tokenization inline                         │
│                                                         │
│ generate() method:                                      │
│   1. Tokenize all prompts                              │
│   2. Add as pending requests                           │
│   3. Run scheduler loop until all done                 │
│   4. Decode outputs and return                         │
└─────────────────────────────────────────────────────────┘

No separate processes needed - all in one Python process!
""")


def demo_multi_gpu_flow():
    """Show multi-GPU tensor parallelism flow."""
    print("\n" + "=" * 60)
    print("6. Multi-GPU (Tensor Parallelism) Flow")
    print("=" * 60)

    print("""
With TP=2, data flows through both GPUs:

┌─────────────────────────────────────────────────────────┐
│                    RANK 0 (GPU 0)                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Scheduler receives UserMsg                        │   │
│  │ Broadcasts to Rank 1 via ZMQ PUB                 │   │
│  └──────────────────────────────────────────────────┘   │
│                        │                                 │
│                        ▼                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Engine (half the model weights)                   │   │
│  │ - Column parallel: compute half of Q,K,V         │   │
│  │ - Row parallel + all_reduce: full output          │   │
│  └──────────────────────────────────────────────────┘   │
│                        │                                 │
│                NCCL all_reduce                          │
│                        │                                 │
└────────────────────────┼────────────────────────────────┘
                         │
                         ▼
┌────────────────────────┼────────────────────────────────┐
│                    RANK 1 (GPU 1)                        │
│                        │                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Scheduler receives broadcast from Rank 0          │   │
│  └──────────────────────────────────────────────────┘   │
│                        │                                 │
│                        ▼                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Engine (other half of model weights)              │   │
│  │ - Same computation on different shards           │   │
│  │ - Synchronizes via all_reduce                    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  (Only Rank 0 sends results to detokenizer)             │
└──────────────────────────────────────────────────────────┘

Key TP operations:
  1. QKV projection: Column parallel (no comm)
  2. Attention: Local heads only (no comm)
  3. Output projection: Row parallel + all_reduce
  4. MLP up/gate: Column parallel (no comm)
  5. MLP down: Row parallel + all_reduce
""")


def demo_summary():
    """Summary of all connections."""
    print("\n" + "=" * 60)
    print("7. Summary: How It All Connects")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────┐
│                  MINI-SGLANG ARCHITECTURE               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User Request                                          │
│       │                                                │
│       ▼                                                │
│  [API Server] ──ZMQ──► [Tokenizer]                    │
│                              │                         │
│                              ▼                         │
│                        [Scheduler]                     │
│                        ┌────────────────────┐          │
│                        │ PrefillManager     │          │
│                        │ DecodeManager      │          │
│                        │ CacheManager       │──► RadixCache
│                        │ TableManager       │          │
│                        └────────┬───────────┘          │
│                                 │                      │
│                                 ▼                      │
│                           [Engine]                     │
│                        ┌────────────────────┐          │
│                        │ Model (TP sharded) │          │
│                        │ KV Cache (paged)   │          │
│                        │ Attention Backend  │          │
│                        │ CUDA Graphs        │          │
│                        │ Sampler            │          │
│                        └────────────────────┘          │
│                                 │                      │
│                                 ▼                      │
│  [Detokenizer] ◄──ZMQ── [Scheduler]                   │
│       │                                                │
│       ▼                                                │
│  [API Server] ──SSE──► User Response                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

Key Connection Types:
  • ZMQ: Inter-process messaging
  • NCCL: GPU-GPU tensor sync (TP)
  • Context: Intra-process batch sharing
  • Page Table: KV cache addressing
""")


if __name__ == "__main__":
    print("Mini-SGLang Component Connections Demo")
    print("End-to-End Architecture Overview\n")

    demo_process_architecture()
    demo_request_lifecycle()
    demo_component_connections()
    demo_engine_internal()
    demo_llm_interface()
    demo_multi_gpu_flow()
    demo_summary()

    print("\n" + "=" * 60)
    print("Prototype Summary:")
    print("=" * 60)
    print("""
This prototype showed how all Mini-SGLang components connect:

Previous Prototypes Covered:
  01: Core data structures (Req, Batch, Context)
  02: RadixCache (prefix sharing)
  03: Layers and BaseOP (TP linear layers)
  04: Distributed (all_reduce, all_gather)
  05: Attention backends (FA, FI, Hybrid)
  06: KV Cache pool (MHAKVCache)
  07: CUDA graphs (capture/replay)
  08: Scheduler (PrefillManager, DecodeManager)
  09: Messages (UserMsg, DetokenizeMsg)
  10: Engine (initialization, forward)

This Prototype Tied Together:
  - Multi-process architecture
  - Complete request lifecycle
  - Internal component interactions
  - Multi-GPU data flow
  - LLM as simplified interface

You now have a complete picture of Mini-SGLang!
""")
