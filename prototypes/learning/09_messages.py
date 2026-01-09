#!/usr/bin/env python3
"""
Learning Prototype 9: Message Types and Communication
======================================================

Demonstrates Mini-SGLang's inter-process messaging:
- Message types: UserMsg, DetokenizeMsg, TokenizeMsg, etc.
- Serialization: JSON-based with tensor support
- ZMQ communication patterns

Uses real code from minisgl.message.
"""

import torch

# Import REAL Mini-SGLang components
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
    UserMsg,
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
)
from minisgl.message.utils import serialize_type, deserialize_type
from minisgl.core import SamplingParams


def demo_message_types():
    """Demonstrate the different message types."""
    print("=" * 60)
    print("1. Message Types Overview")
    print("=" * 60)

    print("""
Mini-SGLang uses typed messages for inter-process communication:

Backend Messages (Tokenizer -> Scheduler):
┌─────────────────────────────────────────────────────────┐
│ UserMsg: New user request with tokenized input          │
│   - uid: Request ID                                     │
│   - input_ids: Tensor of token IDs                      │
│   - sampling_params: Generation parameters              │
├─────────────────────────────────────────────────────────┤
│ ExitMsg: Signal to shut down                            │
├─────────────────────────────────────────────────────────┤
│ BatchBackendMsg: Batch of messages                      │
└─────────────────────────────────────────────────────────┘

Tokenizer Messages (Scheduler -> Tokenizer/Detokenizer):
┌─────────────────────────────────────────────────────────┐
│ DetokenizeMsg: Request to decode a token                │
│   - uid: Request ID                                     │
│   - next_token: Token to decode                         │
│   - finished: Is generation complete?                   │
├─────────────────────────────────────────────────────────┤
│ TokenizeMsg: Request to tokenize text                   │
│   - uid: Request ID                                     │
│   - text: Input text or chat messages                   │
│   - sampling_params: Generation parameters              │
├─────────────────────────────────────────────────────────┤
│ BatchTokenizerMsg: Batch of messages                    │
└─────────────────────────────────────────────────────────┘
""")


def demo_user_msg():
    """Demonstrate UserMsg creation and usage."""
    print("\n" + "=" * 60)
    print("2. UserMsg - User Request Message")
    print("=" * 60)

    # Create a UserMsg
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    params = SamplingParams(temperature=0.7, max_tokens=100)

    msg = UserMsg(
        uid=42,
        input_ids=input_ids,
        sampling_params=params,
    )

    print(f"\nUserMsg created:")
    print(f"  uid: {msg.uid}")
    print(f"  input_ids: {msg.input_ids.tolist()}")
    print(f"  sampling_params: {msg.sampling_params}")

    return msg


def demo_detokenize_msg():
    """Demonstrate DetokenizeMsg."""
    print("\n" + "=" * 60)
    print("3. DetokenizeMsg - Token to Text")
    print("=" * 60)

    # Create DetokenizeMsg (sent after each token is generated)
    msg1 = DetokenizeMsg(uid=42, next_token=100, finished=False)
    msg2 = DetokenizeMsg(uid=42, next_token=101, finished=False)
    msg3 = DetokenizeMsg(uid=42, next_token=2, finished=True)  # EOS token

    print(f"\nSequence of DetokenizeMsg:")
    print(f"  1. {msg1}")
    print(f"  2. {msg2}")
    print(f"  3. {msg3} (finished=True)")

    return msg1


def demo_serialization():
    """Demonstrate message serialization."""
    print("\n" + "=" * 60)
    print("4. Serialization - JSON with Tensor Support")
    print("=" * 60)

    # Create a message with a tensor
    input_ids = torch.tensor([10, 20, 30], dtype=torch.int32)
    params = SamplingParams(temperature=0.5, max_tokens=50)
    msg = UserMsg(uid=1, input_ids=input_ids, sampling_params=params)

    print(f"\nOriginal message:")
    print(f"  {msg}")

    # Serialize
    serialized = msg.encoder()
    print(f"\nSerialized (dict):")
    print(f"  __type__: {serialized['__type__']}")
    print(f"  uid: {serialized['uid']}")
    print(f"  input_ids: {serialized['input_ids']}")
    print(f"  sampling_params: {serialized['sampling_params']}")

    # Deserialize
    # Need to include SamplingParams in the class map for deserialization
    import minisgl.message.backend as backend_module
    backend_module.SamplingParams = SamplingParams
    deserialized = BaseBackendMsg.decoder(serialized)
    print(f"\nDeserialized:")
    print(f"  {deserialized}")
    print(f"  input_ids match: {torch.equal(msg.input_ids, deserialized.input_ids)}")


def demo_tensor_serialization():
    """Show how tensors are serialized."""
    print("\n" + "=" * 60)
    print("5. Tensor Serialization Detail")
    print("=" * 60)

    print("""
Tensors are serialized as:
{
    "__type__": "Tensor",
    "buffer": bytes,  # numpy tobytes()
    "dtype": "torch.int32"
}

This allows efficient transfer of tensor data through ZMQ.

Limitations:
  - Only 1D tensors supported (for input_ids)
  - Uses numpy as intermediate format
""")

    # Demo tensor serialization
    tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    serialized = serialize_type(tensor)

    print(f"\nTensor: {tensor.tolist()}")
    print(f"Serialized:")
    print(f"  __type__: {serialized['__type__']}")
    print(f"  dtype: {serialized['dtype']}")
    print(f"  buffer: {len(serialized['buffer'])} bytes")


def demo_batch_messages():
    """Demonstrate batch messages."""
    print("\n" + "=" * 60)
    print("6. Batch Messages")
    print("=" * 60)

    print("""
BatchBackendMsg and BatchTokenizerMsg batch multiple messages:

class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]

Used to:
  - Send multiple tokens back in one ZMQ message
  - Reduce communication overhead
  - Efficient when many requests finish simultaneously
""")

    # Create batch of detokenize messages
    msgs = [
        DetokenizeMsg(uid=1, next_token=100, finished=False),
        DetokenizeMsg(uid=2, next_token=200, finished=False),
        DetokenizeMsg(uid=3, next_token=300, finished=True),
    ]
    batch = BatchTokenizerMsg(data=msgs)

    print(f"\nBatchTokenizerMsg with {len(batch.data)} messages:")
    for i, msg in enumerate(batch.data):
        print(f"  [{i}] uid={msg.uid}, token={msg.next_token}, done={msg.finished}")


def demo_communication_flow():
    """Show how messages flow through the system."""
    print("\n" + "=" * 60)
    print("7. Message Flow in Mini-SGLang")
    print("=" * 60)

    print("""
Message flow through the system:

User Request:
┌─────────────────────────────────────────────────────────┐
│ 1. User -> API Server: HTTP POST /v1/chat/completions   │
│                                                         │
│ 2. API Server -> Tokenizer: TokenizeMsg                 │
│    (ZMQ PUSH/PULL)                                      │
│                                                         │
│ 3. Tokenizer -> Scheduler: UserMsg                      │
│    (ZMQ PUSH/PULL, contains tokenized input_ids)        │
└─────────────────────────────────────────────────────────┘

Token Generation:
┌─────────────────────────────────────────────────────────┐
│ 4. Scheduler generates token, sends DetokenizeMsg       │
│    (ZMQ PUSH/PULL)                                      │
│                                                         │
│ 5. Detokenizer decodes token, sends to API Server       │
│    (ZMQ PUSH/PULL)                                      │
│                                                         │
│ 6. API Server streams text to User                      │
│    (HTTP SSE or WebSocket)                              │
└─────────────────────────────────────────────────────────┘

Multi-GPU (TP > 1):
┌─────────────────────────────────────────────────────────┐
│ Scheduler Rank 0 broadcasts to other ranks              │
│ (ZMQ PUB/SUB)                                           │
│                                                         │
│ Only Rank 0 sends results to detokenizer                │
└─────────────────────────────────────────────────────────┘
""")


def demo_zmq_patterns():
    """Explain ZMQ socket patterns used."""
    print("\n" + "=" * 60)
    print("8. ZMQ Socket Patterns")
    print("=" * 60)

    print("""
Mini-SGLang uses ZeroMQ for inter-process communication:

1. PUSH/PULL (Pipeline):
   - Tokenizer PUSH -> Scheduler PULL
   - One-to-one, load balanced
   - Used for request submission

2. PUB/SUB (Broadcast):
   - Scheduler Rank 0 PUB -> Other ranks SUB
   - One-to-many broadcast
   - Used for multi-GPU synchronization

ZMQ utilities in minisgl.utils:
  - ZmqPushQueue: Send messages
  - ZmqPullQueue: Receive messages
  - ZmqPubQueue: Broadcast messages
  - ZmqSubQueue: Subscribe to broadcasts

Each queue has:
  - encoder: Message -> bytes (via JSON + msgpack)
  - decoder: bytes -> Message
""")


if __name__ == "__main__":
    print("Mini-SGLang Messages Demo")
    print("Using REAL code from minisgl.message\n")

    demo_message_types()
    user_msg = demo_user_msg()
    detok_msg = demo_detokenize_msg()
    demo_serialization()
    demo_tensor_serialization()
    demo_batch_messages()
    demo_communication_flow()
    demo_zmq_patterns()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. Message types for different communication paths:
   - UserMsg: Tokenized request to scheduler
   - DetokenizeMsg: Token back to detokenizer
   - TokenizeMsg: Text to tokenizer
   - BatchXxxMsg: Efficient batching

2. Serialization supports:
   - Basic types (int, str, float, bool)
   - Dataclasses with __type__ tagging
   - 1D tensors via numpy bytes

3. Communication patterns:
   - PUSH/PULL for request pipeline
   - PUB/SUB for TP rank broadcast

4. ZMQ enables:
   - Fast inter-process communication
   - Language-agnostic protocol
   - No shared memory required
""")
