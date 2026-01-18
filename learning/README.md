# Mini-SGLang Learning Puzzles

Welcome! These puzzles are designed to help you understand how Mini-SGLang works by teaching you:

1. **How data flows** through the system
2. **How components connect** with each other
3. **Why technical decisions** were made

After completing these puzzles, you'll be ready to contribute to the Mini-SGLang repository!

## Getting Started

### Prerequisites

- Basic Python knowledge
- Understanding of dataclasses, context managers, and async programming
- Familiarity with PyTorch tensors (helpful but not required)

### How to Use These Puzzles

1. **Read the puzzle file** - Each puzzle has:
   - **Data Flow Context**: How data moves through the system
   - **Connections**: How components connect
   - **Technical Decisions**: Why design choices were made
   - **Challenge**: What you need to implement

2. **Read the actual code** - Puzzles reference real implementation files:
   - Example: `python/minisgl/core.py:22-24` means lines 22-24 in that file
   - Use these references to understand the real implementation

3. **Implement the solution** - Fill in the `# YOUR CODE HERE` sections

4. **Run the tests** - Verify your solution:
   ```bash
   pytest learning/puzzles/XX_topic/test_X.Y.py -v
   ```

5. **Read the architecture overview** - See `ARCHITECTURE_OVERVIEW.md` for the big picture

## Puzzle Organization

### 01_core_structures
**Core data types that flow through the entire system**

- **1.1 SamplingParams**: Parameters that control token generation
  - Learn: How sampling params flow from API → Engine
  - Decision: Why is_greedy is a property, not stored state

- **1.2 Req Creation**: Request object representing a single inference request
  - Learn: How Req tracks request state (cached_len, device_len)
  - Decision: Why input_ids must be CPU tensor

- **1.3 Request State Transitions**: How Req state changes during processing
  - Learn: Prefill vs decode phases, complete_one() vs append_host()
  - Decision: Why separate CPU-side append from GPU-side processing

- **1.4 Batch Creation**: Grouping requests for parallel GPU processing
  - Learn: How batches are created, why padding is needed
  - Decision: Why separate prefill and decode batches

### 02_simple_llm
**Understanding token generation**

- **2.1 Sampling Behavior**: How sampling parameters affect output
  - Learn: Temperature, top_k, top_p interactions
  - Decision: Why greedy sampling is optimized differently

### 03_messages
**Message passing between processes**

- **3.1 Message Classes**: TokenizeMsg and UserMsg
  - Learn: Complete request lifecycle through messages
  - Decision: Why separate TokenizeMsg (text) from UserMsg (tokens)
  - Decision: Why ZMQ for inter-process communication

### 04_kvcache
**KV cache management for efficient attention**

- **4.1 Cache Handle Basics**: SizeInfo and CacheHandle
  - Learn: How cache handles connect Reqs to cache memory
  - Decision: Why evictable vs protected cache entries

- **4.2 Radix Tree**: Prefix sharing for common request prefixes
  - Learn: How radix tree enables cache sharing
  - Decision: Why radix tree (not simple prefix matching)

### 05_scheduler
**Request scheduling and batching**

- **5.1 TableManager**: Slot allocation for requests
  - Learn: How requests map to fixed-size table slots
  - Decision: Why fixed-size table (not dynamic allocation)

- **5.2 DecodeManager**: Managing decode-phase requests
  - Learn: How decode phase differs from prefill
  - Decision: Why separate PrefillManager and DecodeManager

### 06_context
**Global state management during inference**

- **6.1 Context Management**: Global context for forward passes
  - Learn: How context flows through Engine → Layers
  - Decision: Why global context (not passing everywhere)

## Learning Path

### Recommended Order

1. Start with **01_core_structures** - These are the building blocks
2. Then **02_simple_llm** - Understand token generation
3. Then **03_messages** - See how requests flow through the system
4. Then **04_kvcache** - Understand performance optimizations
5. Then **05_scheduler** - See how requests are managed
6. Finally **06_context** - Understand global state

### After Each Puzzle

1. **Read the referenced code** - See the actual implementation
2. **Trace the data flow** - Follow a request through the codebase
3. **Ask questions** - Why was this designed this way?
4. **Experiment** - Try modifying the code and see what happens

## Key Concepts

### Data Flow

Data flows through the system in this order:
```
User → API Server → Tokenizer → Scheduler → Engine → Scheduler → Detokenizer → API Server → User
```

Each puzzle teaches you about one part of this flow.

### Component Connections

Components connect via:
- **ZMQ messages**: Between processes (API, Tokenizer, Scheduler, Detokenizer)
- **Python objects**: Within a process (Req, Batch, Context)
- **GPU tensors**: Within Engine (input_ids, logits, KV cache)

### Technical Decisions

Each puzzle explains:
- **Why** a design choice was made
- **What** alternatives were considered
- **How** it affects the system

## Architecture Overview

See `ARCHITECTURE_OVERVIEW.md` for:
- Complete request lifecycle (step-by-step)
- Component connections diagram
- Technical decisions summary
- Key data structures and their flows

## Contributing After Learning

Once you've completed the puzzles, you'll understand:
- How requests flow through the system
- How components connect
- Why design decisions were made

You'll be ready to:
- Fix bugs
- Add features
- Optimize performance
- Write tests
- Improve documentation

## Getting Help

- **Read the code**: Puzzles reference actual implementation files
- **Check ARCHITECTURE_OVERVIEW.md**: Big picture explanations
- **Run the tests**: Tests show expected behavior
- **Trace through the codebase**: Use a debugger to follow a request

## Next Steps

1. Start with puzzle 1.1
2. Read `ARCHITECTURE_OVERVIEW.md` for context
3. Work through puzzles in order
4. Read the actual implementation files
5. Start contributing!

Good luck! 🚀
