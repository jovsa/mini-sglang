git # Mini-SGLang Architecture Overview

This document provides a comprehensive overview of how data flows through Mini-SGLang, how components connect, and the technical decisions behind the design. Use this as a reference while working through the puzzles.

## System Architecture

Mini-SGLang is a **distributed LLM inference system** with multiple independent processes:

- **API Server**: FastAPI server providing OpenAI-compatible endpoints
- **Tokenizer Worker**: Converts text ↔ tokens (separate process for batching efficiency)
- **Scheduler Worker**: Core inference engine (one per GPU in multi-GPU setups)
- **Detokenizer Worker**: Converts generated tokens back to text

### Communication Architecture

Components communicate via **ZeroMQ (ZMQ)** for control messages:
- **Why ZMQ?** Fast, async, handles process boundaries, supports pub/sub patterns
- **Message Types**: TokenizeMsg, UserMsg, DetokenizeMsg, UserReply (see `python/minisgl/message/`)
- **Serialization**: Automatic via dataclass encoder/decoder pattern

For multi-GPU setups, schedulers use **NCCL** (via `torch.distributed`) for tensor data.

## Complete Request Lifecycle

### 1. User Request → API Server
```
User → HTTP POST /v1/chat/completions
  - Prompt: "Hello, how are you?"
  - Parameters: temperature=0.7, max_tokens=100
```

**Location**: `python/minisgl/server/api_server.py:245-277`

### 2. API Server → Tokenizer
```
API Server creates TokenizeMsg:
  - uid: 1 (unique request ID)
  - text: "Hello, how are you?" (string)
  - sampling_params: SamplingParams(temperature=0.7, max_tokens=100)

Sent via: ZmqAsyncPushQueue → Tokenizer Worker
```

**Location**: `python/minisgl/server/api_server.py:256-267`
**Message Type**: `python/minisgl/message/tokenizer.py:34-38`

**Technical Decision**: Why separate tokenizer process?
- Allows batching multiple tokenization requests
- Keeps API server responsive (tokenization can be slow)
- Enables async processing

### 3. Tokenizer → Scheduler
```
Tokenizer processes TokenizeMsg:
  - Converts text to token IDs: [1, 2, 3, 4, 5]
  - Creates UserMsg:
    - uid: 1 (same as TokenizeMsg)
    - input_ids: torch.Tensor([1, 2, 3, 4, 5]) (CPU tensor)
    - sampling_params: (forwarded from TokenizeMsg)

Sent via: ZmqPushQueue → Scheduler (Rank 0)
```

**Location**: `python/minisgl/tokenizer/server.py:84-98`
**Message Type**: `python/minisgl/message/backend.py:33-36`

**Technical Decision**: Why CPU tensor in UserMsg?
- Messages cross process boundaries (ZMQ sends bytes)
- GPU tensors can't be serialized
- Tensor moved to GPU when Batch is created

### 4. Scheduler (Rank 0) → Other Schedulers (Multi-GPU)
```
If using multiple GPUs:
  - Rank 0 receives UserMsg
  - Broadcasts raw message bytes to other ranks
  - All ranks decode and process the same request
```

**Location**: `python/minisgl/scheduler/io.py:88-122`

**Technical Decision**: Why broadcast raw bytes?
- Avoids re-serialization overhead
- Ensures all ranks see identical messages
- Efficient for tensor parallelism

### 5. Scheduler Creates Req
```
Scheduler receives UserMsg:
  - Allocates table slot: table_idx = 0
  - Creates Req:
    - input_ids: [1, 2, 3, 4, 5] (CPU tensor)
    - table_idx: 0
    - cached_len: 0 (no tokens cached yet)
    - device_len: 5 (length of input)
    - max_device_len: 105 (input_len + max_tokens)
    - uid: 1
    - sampling_params: (from UserMsg)
    - cache_handle: (allocated from cache manager)

Added to PrefillManager (for prefill phase)
```

**Location**: `python/minisgl/scheduler/scheduler.py:161-175`

**Technical Decision**: Why separate Req from UserMsg?
- Req is internal scheduler state (not serialized)
- Req tracks processing state (cached_len, device_len)
- UserMsg is just input data (crosses boundaries)

### 6. Prefill Phase
```
Scheduler collects Reqs from PrefillManager:
  - Creates Batch(phase="prefill", reqs=[req1, req2, ...])
  - Prepares batch:
    - Concatenates input_ids (with padding)
    - Creates out_loc (maps outputs to Reqs)
    - Sets padded_reqs (may include dummy Reqs)

Sends Batch to Engine.forward()
```

**Location**: `python/minisgl/scheduler/scheduler.py:180-220`

**Technical Decision**: Why padding?
- GPU processes batches in parallel
- All sequences must be same length for matrix ops
- Padding tokens masked in attention

### 7. Engine Forward Pass
```
Engine.forward(batch):
  1. Sets global context: ctx.forward_batch(batch)
  2. Moves input_ids to GPU
  3. Forward pass through model:
     - Embedding layer
     - Transformer layers (attention + MLP)
     - Output logits
  4. For each token in batch:
     - complete_one() called (updates cached_len, device_len)
  5. Sampling: Engine.sample() applies sampling_params
  6. Returns next_token for each request
```

**Location**: `python/minisgl/engine/engine.py`

**Technical Decision**: Why global context?
- Deep call stack: Engine → Model → Layers → Attention
- Passing context everywhere is verbose
- Thread-local (safe for single-threaded GPU ops)

### 8. Sampling
```
Engine.sample(logits, sampling_params):
  1. Apply temperature: logits = logits / temperature
  2. Apply top_k: Keep only top k tokens
  3. Apply top_p: Keep tokens until cumulative prob >= top_p
  4. Sample from filtered distribution
  5. Return next_token (token ID)
```

**Location**: `python/minisgl/engine/sample.py`

**Technical Decision**: Why is_greedy optimization?
- When greedy (temp=0, top_k=1, top_p=1.0), skip sampling
- Just take argmax(logits) - much faster!

### 9. Decode Phase
```
After prefill complete:
  - Req moves from PrefillManager to DecodeManager
  - Each decode step:
    1. Batch created (phase="decode", all sequences length 1)
    2. Engine generates next_token
    3. append_host(next_token): Adds token to input_ids (CPU)
    4. complete_one(): Updates cached_len, device_len
    5. Repeat until max_tokens or EOS
```

**Location**: `python/minisgl/scheduler/scheduler.py`

**Technical Decision**: Why separate decode phase?
- All sequences length 1 (no padding needed)
- Different attention pattern (incremental vs full)
- More efficient batching

### 10. Scheduler → Detokenizer
```
For each generated token:
  - Scheduler creates DetokenizeMsg:
    - uid: 1
    - next_token: 42 (token ID)
    - finished: False

  Sent via: ZmqPushQueue → Detokenizer Worker
```

**Location**: `python/minisgl/scheduler/scheduler.py:153`
**Message Type**: `python/minisgl/message/tokenizer.py:28-31`

### 11. Detokenizer → API Server
```
Detokenizer processes DetokenizeMsg:
  - Converts token ID to text: "Hello"
  - Creates UserReply:
    - uid: 1
    - incremental_output: "Hello"
    - finished: False

  Sent via: ZmqAsyncPullQueue → API Server
```

**Location**: `python/minisgl/tokenizer/server.py:65-82`
**Message Type**: `python/minisgl/message/frontend.py`

### 12. API Server → User
```
API Server receives UserReply:
  - Streams to user: "data: Hello\n"
  - When finished: "data: [DONE]\n"

HTTP Response: Server-Sent Events (SSE) stream
```

**Location**: `python/minisgl/server/api_server.py:151-156`

## Key Data Structures

### SamplingParams
- **Flow**: API → TokenizeMsg → UserMsg → Req → Engine.sample()
- **Purpose**: Controls token generation (temperature, top_k, top_p)
- **Location**: `python/minisgl/core.py:14-24`

### Req
- **Flow**: Created in Scheduler, stored in PrefillManager/DecodeManager
- **Purpose**: Tracks single request state (cached_len, device_len, etc.)
- **Location**: `python/minisgl/core.py:27-66`

### Batch
- **Flow**: Created in Scheduler, sent to Engine.forward()
- **Purpose**: Groups multiple Reqs for parallel GPU processing
- **Location**: `python/minisgl/core.py:69-94`

### Context
- **Flow**: Set in Engine.forward(), accessed by Layers
- **Purpose**: Global state during forward pass (current batch, config)
- **Location**: `python/minisgl/core.py:97-129`

## Component Connections

### Message Passing (ZMQ)
```
API Server ←→ Tokenizer ←→ Scheduler ←→ Detokenizer
   (HTTP)      (ZMQ)        (ZMQ)        (ZMQ)
```

### Scheduler Internal
```
UserMsg → Req → PrefillManager → Batch → Engine
                              ↓
                         DecodeManager → Batch → Engine
```

### Engine Internal
```
Batch → Context → Model → Layers → Attention → KV Cache
```

## Technical Decisions Summary

1. **Why separate processes?**
   - Isolation: Tokenizer crashes don't affect scheduler
   - Scalability: Can run tokenizer on different machine
   - Batching: Each process can batch independently

2. **Why ZMQ for messages?**
   - Fast async communication
   - Handles process boundaries
   - Supports pub/sub for multi-GPU

3. **Why CPU tensors in messages?**
   - GPU tensors can't cross process boundaries
   - Serialization requires CPU tensors
   - Moved to GPU when Batch created

4. **Why separate prefill/decode?**
   - Different batching characteristics
   - Different optimizations (chunked prefill, continuous decode)
   - Cleaner separation of concerns

5. **Why global context?**
   - Deep call stack makes passing context verbose
   - Thread-local is safe for GPU ops
   - Cleaner API for layers

6. **Why KV cache?**
   - Avoids recomputing attention for previous tokens
   - Critical for decode phase performance
   - Radix cache enables prefix sharing

## Puzzle Learning Path

The puzzles are designed to teach these concepts in order:

1. **01_core_structures**: Core data types (SamplingParams, Req, Batch, Context)
2. **02_simple_llm**: Sampling behavior and token generation
3. **03_messages**: Message passing and request lifecycle
4. **04_kvcache**: KV cache management and prefix sharing
5. **05_scheduler**: Request scheduling and batching
6. **06_context**: Global state management

Each puzzle includes:
- **Data Flow**: How data moves through the system
- **Connections**: How components connect
- **Technical Decisions**: Why design choices were made

## Next Steps

After completing the puzzles:
1. Read the actual implementation files referenced in each puzzle
2. Trace a request through the codebase using a debugger
3. Experiment with different configurations
4. Contribute improvements based on your understanding!
