import torch
"""
PUZZLE 3.1: Create Message Classes - [Easy]

TASK:
Implement TokenizeMsg and UserMsg classes following the message system patterns.

GIVEN:
- Base message class structure
- Understanding of message types

================================================================================
DATA FLOW: Complete Message Flow Through the System
================================================================================

Messages flow through multiple processes via ZeroMQ. Here's the complete lifecycle:

```mermaid
flowchart LR
    A[HTTP Request] -->|"POST /v1/chat/completions"| B[API Server]
    B -->|"Create TokenizeMsg"| C[ZmqAsyncPushQueue]
    C -->|"ZMQ PUSH"| D[Tokenizer Worker]
    D -->|"Tokenize text"| E[Create UserMsg]
    E -->|"ZmqPushQueue"| F[Scheduler Rank 0]
    F -->|"Broadcast bytes"| G[Other Schedulers]
    G -->|"Decode UserMsg"| H[All Schedulers]
    H -->|"Process request"| I[Generate tokens]
    I -->|"Create DetokenizeMsg"| J[Detokenizer Worker]
    J -->|"Detokenize"| K[Create UserReply]
    K -->|"ZmqAsyncPullQueue"| L[API Server]
    L -->|"SSE stream"| M[HTTP Response]
```

Step-by-step flow:
1. **API Server** (`python/minisgl/server/api_server.py:245-268`):
   - Receives HTTP request with prompt and sampling parameters
   - Creates `TokenizeMsg(uid, text, sampling_params)`
   - Sends via `ZmqAsyncPushQueue` to Tokenizer Worker

2. **Tokenizer Worker** (`python/minisgl/tokenizer/server.py:84-98`):
   - Receives `TokenizeMsg`
   - Tokenizes text: `text → token_ids` (list of integers)
   - Creates `UserMsg(uid, input_ids, sampling_params)` where `input_ids` is CPU tensor
   - Sends via `ZmqPushQueue` to Scheduler (Rank 0)

3. **Scheduler Rank 0** (`python/minisgl/scheduler/io.py:88-122`):
   - Receives `UserMsg`
   - Broadcasts raw message bytes to other ranks (for multi-GPU)
   - All ranks decode and process the same request

4. **Detokenizer** (`python/minisgl/tokenizer/server.py:65-82`):
   - Receives `DetokenizeMsg(uid, next_token, finished)` from Scheduler
   - Converts token ID to text
   - Creates `UserReply(uid, incremental_output, finished)`
   - Sends via `ZmqAsyncPullQueue` to API Server

5. **API Server** (`python/minisgl/server/api_server.py:151-156`):
   - Receives `UserReply`
   - Streams to user via Server-Sent Events (SSE)

================================================================================
EXERCISE 1: Trace Message Flow - [Interactive]
================================================================================

TASK: Implement `trace_message_flow()` to trace how data moves through the system.

This exercise requires you to write code that simulates the message flow and tracks
data transformations. You'll see firsthand how UID and SamplingParams are preserved.

INSTRUCTIONS:
1. Implement `trace_message_flow(tokenize_msg: TokenizeMsg) -> dict`
2. The function should:
   - Simulate tokenization: convert text to input_ids tensor
   - Create UserMsg from tokenized input
   - Return a dictionary with keys:
     - "step1_tokenize_msg": The original TokenizeMsg
     - "step2_tokenized": The tokenized input_ids tensor
     - "step3_user_msg": The created UserMsg
     - "uid_preserved": bool - whether UID matches
     - "sampling_params_preserved": bool - whether SamplingParams match
3. Verify that UID and SamplingParams are preserved through the flow

HINT:
- Use `_simulate_tokenization()` helper function (provided below)
- Compare uid and sampling_params between TokenizeMsg and UserMsg
- This demonstrates how data flows: TokenizeMsg → tokenization → UserMsg

LEARNING GOAL: Hands-on experience tracing data through the system
"""


def _simulate_tokenization(text: str) -> torch.Tensor:
    """
    Simulate tokenization by creating a simple tensor.
    In real system, this would use actual tokenizer.
    For this exercise, we just create a tensor with length based on text.
    """
    # Simple simulation: create tensor with length proportional to text length
    num_tokens = max(1, len(text) // 3)  # Rough approximation
    return torch.tensor(list(range(1, num_tokens + 1)), dtype=torch.int32)


def trace_message_flow(tokenize_msg: "TokenizeMsg") -> dict:
    """
    Trace message flow from TokenizeMsg to UserMsg.

    This function simulates the complete flow:
    1. Start with TokenizeMsg (from API Server)
    2. Tokenize the text (simulate Tokenizer Worker)
    3. Create UserMsg (simulate Tokenizer creating message for Scheduler)
    4. Verify data preservation (UID, SamplingParams)

    Args:
        tokenize_msg: The initial TokenizeMsg

    Returns:
        Dictionary with flow steps and verification results

    TODO: Implement this function
    """
    # Step 1: Extract data from TokenizeMsg
    uid = tokenize_msg.uid
    text = tokenize_msg.text
    sampling_params = tokenize_msg.sampling_params

    # Step 2: Simulate tokenization using _simulate_tokenization()
    input_ids = _simulate_tokenization(text)

    # Step 3: Create UserMsg using create_user_msg()
    user_msg = create_user_msg(uid, input_ids, sampling_params.max_tokens)

    # Step 4: Verify UID and SamplingParams are preserved
    uid_preserved = user_msg.uid == uid
    sampling_params_preserved = (
        user_msg.sampling_params.max_tokens == sampling_params.max_tokens and
        user_msg.sampling_params.temperature == sampling_params.temperature and
        user_msg.sampling_params.top_k == sampling_params.top_k and
        user_msg.sampling_params.top_p == sampling_params.top_p
    )



    # Step 5: Return dictionary with all steps and verification
    return {
        "step1_tokenize_msg": tokenize_msg,
        "step2_tokenized": input_ids,
        "step3_user_msg": user_msg,
        "uid_preserved": uid_preserved,
        "sampling_params_preserved": sampling_params_preserved
    }

"""
================================================================================
COMPONENT CONNECTIONS: ZMQ Message Passing Architecture
================================================================================

```mermaid
graph TB
    subgraph "API Server Process"
        API[API Server]
        TQueue1[ZmqAsyncPushQueue]
        RQueue1[ZmqAsyncPullQueue]
    end

    subgraph "Tokenizer Process"
        Tokenizer[Tokenizer Worker]
        TQueue2[ZmqPushQueue]
    end

    subgraph "Scheduler Process"
        Scheduler[Scheduler]
        SQueue1[ZmqPushQueue]
        SQueue2[ZmqPushQueue]
    end

    subgraph "Detokenizer Process"
        Detokenizer[Detokenizer Worker]
        DQueue1[ZmqAsyncPullQueue]
    end

    subgraph "Message Types"
        TMsg[TokenizeMsg]
        UMsg[UserMsg]
        DMsg[DetokenizeMsg]
        Reply[UserReply]
    end

    API -->|"Sends"| TQueue1
    TQueue1 -->|"ZMQ PUSH"| Tokenizer
    Tokenizer -->|"Creates"| TMsg
    Tokenizer -->|"Creates"| UMsg
    UMsg -->|"Sends"| TQueue2
    TQueue2 -->|"ZMQ PUSH"| Scheduler
    Scheduler -->|"Creates"| DMsg
    DMsg -->|"Sends"| SQueue1
    SQueue1 -->|"ZMQ PUSH"| Detokenizer
    Detokenizer -->|"Creates"| Reply
    Reply -->|"Sends"| DQueue1
    DQueue1 -->|"ZMQ PULL"| RQueue1
    RQueue1 -->|"Receives"| API
```

Key Connections:
- **ZMQ Queues**: Handle async message passing between processes
- **PUSH/PULL Pattern**: Tokenizer/Scheduler use PUSH, Detokenizer/API use PULL
- **Async Queues**: API Server uses async queues to avoid blocking

================================================================================
EXERCISE 2: Map Component Connections - [Interactive]
================================================================================

TASK: Implement `map_message_connections()` to understand how components connect via messages.

This exercise requires you to write code that maps which components handle which messages.
You'll see how messages connect different processes in the system.

INSTRUCTIONS:
1. Implement `map_message_connections(messages: list) -> dict`
2. The function should:
   - Take a list of messages (TokenizeMsg and/or UserMsg)
   - Return a dictionary mapping message types to their handlers:
     - "TokenizeMsg" -> list of component names that handle it
     - "UserMsg" -> list of component names that handle it
   - Component names should be: "API Server", "Tokenizer Worker", "Scheduler"
3. Verify message types match expected components

HINT:
- TokenizeMsg is created by API Server, handled by Tokenizer Worker
- UserMsg is created by Tokenizer Worker, handled by Scheduler
- Use isinstance() to check message types

LEARNING GOAL: Understand how components connect via messages
"""

def map_message_connections(messages: list) -> dict:
    """
    Map which components handle which message types.

    This function demonstrates how messages connect components:
    - TokenizeMsg: API Server → Tokenizer Worker
    - UserMsg: Tokenizer Worker → Scheduler

    Args:
        messages: List of message objects (TokenizeMsg, UserMsg)

    Returns:
        Dictionary mapping message type names to lists of component names

    TODO: Implement this function
    """
    # Step 1: Initialize result dictionary with empty lists for each message type
    result = {
        "TokenizeMsg": [],
        "UserMsg": []
    }

    # Step 2: Iterate through messages
    for msg in messages:
        # Step 3: For each message, determine its type (TokenizeMsg or UserMsg)
        if isinstance(msg, TokenizeMsg):
            # Step 4: Add appropriate component names to the mapping
            #   - TokenizeMsg: created by "API Server", handled by "Tokenizer Worker"
            if "API Server" not in result["TokenizeMsg"]:
                result["TokenizeMsg"].append("API Server")
            if "Tokenizer Worker" not in result["TokenizeMsg"]:
                result["TokenizeMsg"].append("Tokenizer Worker")
        elif isinstance(msg, UserMsg):
            #   - UserMsg: created by "Tokenizer Worker", handled by "Scheduler"
            if "Tokenizer Worker" not in result["UserMsg"]:
                result["UserMsg"].append("Tokenizer Worker")
            if "Scheduler" not in result["UserMsg"]:
                result["UserMsg"].append("Scheduler")

    # Step 5: Return the mapping
    return result

"""
================================================================================
TECHNICAL DECISIONS: Why Separate TokenizeMsg and UserMsg
================================================================================

**Decision**: Separate message types for text (TokenizeMsg) and tokens (UserMsg).

**Why?**
1. **Separation of Concerns**:
   - **TokenizeMsg**: Contains text (string) - human-readable input
   - **UserMsg**: Contains tokens (tensor) - machine representation
   - Different processes handle different representations

2. **Batching Efficiency**:
   - Tokenizer can batch multiple `TokenizeMsg` requests together
   - Tokenization is CPU-bound, so batching improves throughput
   - Scheduler receives already-tokenized `UserMsg` (ready for GPU)

3. **Process Isolation**:
   - Tokenizer runs in separate process (can crash without affecting scheduler)
   - Tokenization can be slow for long texts (keeps API server responsive)
   - Can scale tokenizer independently

**Alternative Considered**: Single message type with both text and tokens.
   - **Problem**: Can't batch tokenization efficiently
   - **Problem**: Scheduler would need tokenizer (adds dependency)
   - **Chosen**: Two-message approach for better separation and performance

**Why ZMQ Instead of HTTP?**
1. **Performance**: ZMQ is much faster for inter-process communication
   - Lower latency (no HTTP overhead)
   - Higher throughput (binary protocol)
   - Zero-copy when possible

2. **Async Patterns**: ZMQ supports pub/sub, push/pull patterns natively
   - Perfect for producer/consumer patterns
   - Handles backpressure automatically
   - Built-in message queuing

3. **Process Boundaries**: ZMQ is designed for inter-process communication
   - Handles process crashes gracefully
   - Automatic reconnection
   - No need for HTTP server in each process

**Alternative Considered**: HTTP for all communication.
   - **Problem**: Too much overhead for high-frequency messages
   - **Problem**: Each process needs HTTP server (complexity)
   - **Chosen**: ZMQ for internal, HTTP only for external API

**Why CPU Tensor in UserMsg?**
- GPU tensors cannot be serialized for ZMQ messages
- Must move to CPU before sending across process boundaries
- Moved to GPU when Batch is created in Scheduler

================================================================================
EXERCISE 3: Simulate Message Serialization - [Interactive]
================================================================================

TASK: Implement `simulate_serialization()` to understand why CPU tensors are required.

This exercise requires you to write code that simulates serialization/deserialization.
You'll see firsthand why GPU tensors can't cross process boundaries.

INSTRUCTIONS:
1. Implement `simulate_serialization(user_msg: UserMsg) -> dict`
2. The function should:
   - Simulate serialization: convert UserMsg to a serializable dict
   - Simulate deserialization: reconstruct UserMsg from dict
   - Verify CPU tensor requirement (try to serialize, verify it works)
   - Return dictionary with:
     - "original_msg": The original UserMsg
     - "serialized": The serialized representation (dict)
     - "deserialized_msg": The reconstructed UserMsg (if possible)
     - "cpu_tensor_required": bool - whether CPU tensor was required
3. Demonstrate why CPU tensors are needed for ZMQ serialization

HINT:
- Convert tensor to list for serialization: `tensor.tolist()`
- Reconstruct tensor from list: `torch.tensor(list, dtype=torch.int32)`
- This shows why GPU tensors can't be serialized (they can't cross process boundaries)

LEARNING GOAL: Understand why CPU tensors are required (ZMQ serialization)
"""

def simulate_serialization(user_msg: "UserMsg") -> dict:
    """
    Simulate serialization and deserialization of UserMsg.

    This function demonstrates why CPU tensors are required:
    - ZMQ messages must be serializable (convert to bytes)
    - GPU tensors cannot be serialized directly
    - CPU tensors can be converted to lists/bytes for serialization

    Args:
        user_msg: The UserMsg to serialize/deserialize

    Returns:
        Dictionary with serialization results and verification

    TODO: Implement this function
    """
    # Step 1: Verify input has CPU tensor (should already be enforced)
    assert user_msg.input_ids.is_cpu, "input_ids must be on CPU for serialization"

    # Step 2: Serialize: convert UserMsg to dict
    #   - Convert tensor to list: input_ids.tolist()
    #   - Store uid, sampling_params (they're already serializable)
    serialized = {
        "uid": user_msg.uid,
        "input_ids": user_msg.input_ids.tolist(),  # Convert tensor to list
        "sampling_params": user_msg.sampling_params,  # Already serializable
        "max_tokens": user_msg.sampling_params.max_tokens
    }

    # Step 3: Deserialize: reconstruct UserMsg from dict
    #   - Convert list back to tensor: torch.tensor(..., dtype=torch.int32)
    #   - Reconstruct UserMsg using create_user_msg()
    input_ids_reconstructed = torch.tensor(serialized["input_ids"], dtype=torch.int32)
    deserialized_msg = create_user_msg(
        uid=serialized["uid"],
        input_ids=input_ids_reconstructed,
        max_tokens=serialized["max_tokens"]
    )

    # Step 4: Verify deserialized message matches original
    uid_matches = deserialized_msg.uid == user_msg.uid
    input_ids_match = torch.equal(deserialized_msg.input_ids, user_msg.input_ids)
    params_match = deserialized_msg.sampling_params.max_tokens == user_msg.sampling_params.max_tokens

    # Step 5: Return dictionary with results
    return {
        "original_msg": user_msg,
        "serialized": serialized,
        "deserialized_msg": deserialized_msg,
        "cpu_tensor_required": True,  # CPU tensor is required for serialization
        "serialization_successful": uid_matches and input_ids_match and params_match
    }

"""
================================================================================
EXERCISE 4: Compare Message Types - [Interactive]
================================================================================

TASK: Implement `compare_message_types()` to understand why TokenizeMsg and UserMsg are separate.

This exercise requires you to write code that compares the two message types and explains
the technical decision to keep them separate.

INSTRUCTIONS:
1. Implement `compare_message_types(tokenize_msg: TokenizeMsg, user_msg: UserMsg) -> dict`
2. The function should:
   - Compare TokenizeMsg and UserMsg
   - Return a dictionary with:
     - "common_fields": list of field names present in both
     - "tokenize_msg_only": list of fields only in TokenizeMsg
     - "user_msg_only": list of fields only in UserMsg
     - "why_separate": list of reasons (strings) explaining separation
   - Reasons should reference: batching, process isolation, separation of concerns
3. Explain why they're separate (batching efficiency, process isolation)

HINT:
- Use dataclasses.fields() to get field names
- Common fields: uid, sampling_params
- TokenizeMsg has: text
- UserMsg has: input_ids
- Think about why text vs tokens are separate

LEARNING GOAL: Understand technical decision to separate message types
"""

def compare_message_types(tokenize_msg: "TokenizeMsg", user_msg: "UserMsg") -> dict:
    """
    Compare TokenizeMsg and UserMsg to understand why they're separate.

    This function demonstrates the technical decision:
    - TokenizeMsg: text representation (human-readable)
    - UserMsg: token representation (machine-readable)
    - Separation enables: batching, process isolation, efficiency

    Args:
        tokenize_msg: A TokenizeMsg instance
        user_msg: A UserMsg instance

    Returns:
        Dictionary with comparison results and explanations

    TODO: Implement this function
    """
    from dataclasses import fields

    # Step 1: Get field names from both message types using fields()
    tokenize_fields = {f.name for f in fields(TokenizeMsg)}
    user_fields = {f.name for f in fields(UserMsg)}

    # Step 2: Find common fields (present in both)
    common_fields = list(tokenize_fields & user_fields)

    # Step 3: Find fields only in TokenizeMsg
    tokenize_msg_only = list(tokenize_fields - user_fields)

    # Step 4: Find fields only in UserMsg
    user_msg_only = list(user_fields - tokenize_fields)

    # Step 5: Generate reasons for separation:
    #   - "Batching efficiency: Tokenizer can batch TokenizeMsg requests"
    #   - "Process isolation: Tokenizer runs in separate process"
    #   - "Separation of concerns: text vs tokens are different representations"
    why_separate = [
        "Batching efficiency: Tokenizer can batch TokenizeMsg requests",
        "Process isolation: Tokenizer runs in separate process",
        "Separation of concerns: text vs tokens are different representations"
    ]

    # Step 6: Return comparison dictionary
    return {
        "common_fields": common_fields,
        "tokenize_msg_only": tokenize_msg_only,
        "user_msg_only": user_msg_only,
        "why_separate": why_separate
    }


"""
================================================================================
CHALLENGE
================================================================================

- Decide what fields each message needs
- Follow the dataclass pattern
- Understand message serialization

HINT:
- Reference: `python/minisgl/message/backend.py:33-36` for UserMsg fields
- Reference: `python/minisgl/message/tokenizer.py:34-38` for TokenizeMsg fields
- Both messages need: uid, sampling_params
- TokenizeMsg has text (str), UserMsg has input_ids (torch.Tensor)

QUESTIONS:
1. What fields does UserMsg need? Why?
2. What's the difference between TokenizeMsg and UserMsg?
3. Why do messages need serialization?
4. **NEW**: Trace through the code: How does `UserMsg` get serialized for ZMQ?
   (Hint: Read `python/minisgl/message/backend.py:14-15` - encoder method)
5. **NEW**: Why does Scheduler broadcast raw bytes instead of re-serializing?
   (Hint: Read `python/minisgl/scheduler/io.py:88-122` - efficiency consideration)

TEST:
Run: pytest learning/puzzles/03_messages/test_3.1.py -v
"""

from dataclasses import dataclass
from typing import Optional
import torch
import sys
import importlib.util
from pathlib import Path

# Import puzzle_1.1 using importlib (module name has dot)
puzzle_dir = Path(__file__).parent.parent / "01_core_structures"
puzzle_1_1_file = puzzle_dir / "puzzle_1.1.py"
spec = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_1_1)
create_sampling_params = puzzle_1_1.create_sampling_params


@dataclass
class BaseMessage:
    """Base class for all messages."""
    pass


@dataclass
class TokenizeMsg(BaseMessage):
    """
    Message to request tokenization of text.

    TODO: Add the required fields:
    - uid: int - Unique request ID
    - text: str - Text to tokenize
    - sampling_params: object - Sampling parameters (use SamplingParams from puzzle_1_1)
    """
    uid: int
    text: str
    sampling_params: object  # SamplingParams


@dataclass
class UserMsg(BaseMessage):
    """
    Message containing tokenized user request.

    TODO: Add the required fields:
    - uid: int - Unique request ID
    - input_ids: torch.Tensor - Tokenized input (CPU tensor)
    - sampling_params: object - Sampling parameters
    """
    uid: int
    input_ids: torch.Tensor  # CPU tensor
    sampling_params: object  # SamplingParams


def create_tokenize_msg(uid: int, text: str, max_tokens: int = 1024) -> "TokenizeMsg":
    """
    Create a TokenizeMsg.

    TODO: Implement this function
    """
    sampling_params = create_sampling_params(max_tokens=max_tokens)
    return TokenizeMsg(uid=uid, text=text, sampling_params=sampling_params)


def create_user_msg(uid: int, input_ids: torch.Tensor, max_tokens: int = 1024) -> "UserMsg":
    """
    Create a UserMsg from tokenized input.

    TODO: Implement this function
    """
    # Ensure input_ids is on CPU (required for ZMQ serialization)
    if not input_ids.is_cpu:
        input_ids = input_ids.cpu()

    sampling_params = create_sampling_params(max_tokens=max_tokens)
    return UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params)

"""
================================================================================
EXERCISE 5: Track Request Through System - [Comprehensive Interactive]
================================================================================

TASK: Implement `track_request_lifecycle()` to understand the complete message flow.

This is a comprehensive exercise that ties together all concepts:
- Data flow through the system
- Component connections
- Technical decisions in practice

INSTRUCTIONS:
1. Implement `track_request_lifecycle(uid: int, text: str, max_tokens: int) -> dict`
2. The function should:
   - Simulate complete request lifecycle: TokenizeMsg → UserMsg → (simulated processing)
   - Track UID through all steps
   - Verify data transformations at each step
   - Return a dictionary with:
     - "step1_api_server": TokenizeMsg created
     - "step2_tokenizer": UserMsg created (with tokenization)
     - "step3_scheduler": Simulated processing (just acknowledge receipt)
     - "uid_tracked": bool - UID consistent through all steps
     - "data_transformed": bool - text successfully converted to tokens
     - "sampling_params_preserved": bool - SamplingParams unchanged
3. This demonstrates the complete flow from API to Scheduler

HINT:
- Use create_tokenize_msg() for step 1
- Use _simulate_tokenization() and create_user_msg() for step 2
- For step 3, just verify the UserMsg would be received by Scheduler
- Track UID through all steps to verify request tracking

LEARNING GOAL: Complete understanding of message flow and data transformations

"""
def track_request_lifecycle(uid: int, text: str, max_tokens: int) -> dict:
    """
    Track a complete request through the message system.

    This function simulates the full lifecycle:
    1. API Server creates TokenizeMsg
    2. Tokenizer Worker processes TokenizeMsg, creates UserMsg
    3. Scheduler receives UserMsg (simulated)

    Args:
        uid: Unique request ID
        text: Input text to process
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with lifecycle steps and verification results

    TODO: Implement this function
    """
    # Step 1: API Server creates TokenizeMsg
    #   - Use create_tokenize_msg() with provided parameters
    tokenize_msg = create_tokenize_msg(uid=uid, text=text, max_tokens=max_tokens)

    # Step 2: Tokenizer Worker processes TokenizeMsg
    #   - Extract text and sampling_params from TokenizeMsg
    #   - Simulate tokenization using _simulate_tokenization()
    #   - Create UserMsg using create_user_msg() with same UID
    input_ids = _simulate_tokenization(text)
    user_msg = create_user_msg(
        uid=uid,
        input_ids=input_ids,
        max_tokens=max_tokens
    )

    # Step 3: Scheduler receives UserMsg (simulated)
    #   - Just verify UserMsg would be received (check it exists, has correct UID)
    scheduler_received = user_msg is not None and user_msg.uid == uid

    # Step 4: Verify UID tracking
    #   - Check UID matches in all steps
    uid_tracked = (
        tokenize_msg.uid == uid and
        user_msg.uid == uid and
        tokenize_msg.uid == user_msg.uid
    )

    # Step 5: Verify data transformation
    #   - Check text was converted to tokens (input_ids created)
    data_transformed = (
        len(user_msg.input_ids) > 0 and
        isinstance(user_msg.input_ids, torch.Tensor) and
        user_msg.input_ids.is_cpu
    )

    # Step 6: Verify SamplingParams preservation
    #   - Check SamplingParams match between TokenizeMsg and UserMsg
    sampling_params_preserved = (
        user_msg.sampling_params.max_tokens == tokenize_msg.sampling_params.max_tokens and
        user_msg.sampling_params.temperature == tokenize_msg.sampling_params.temperature and
        user_msg.sampling_params.top_k == tokenize_msg.sampling_params.top_k and
        user_msg.sampling_params.top_p == tokenize_msg.sampling_params.top_p
    )

    # Step 7: Return comprehensive dictionary with all steps and verifications
    return {
        "step1_api_server": tokenize_msg,
        "step2_tokenizer": user_msg,
        "step3_scheduler": "Received" if scheduler_received else None,
        "uid_tracked": uid_tracked,
        "data_transformed": data_transformed,
        "sampling_params_preserved": sampling_params_preserved
    }
