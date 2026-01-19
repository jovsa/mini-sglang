import torch
import numpy as np
import msgpack
from typing import Dict, Any, List
"""
PUZZLE 3.2: End-to-End Message Flow with ZMQ and CPU/GPU Transitions - [Advanced]

TASK:
Understand the complete message flow, ZMQ serialization mechanism, and how data moves
between CPU and GPU throughout the system.

GIVEN:
- Message classes from puzzle_3.1
- Understanding of ZMQ message passing
- Understanding of CPU/GPU tensor operations

================================================================================
ZMQ SERIALIZATION: How Messages Cross Process Boundaries
================================================================================

ZeroMQ (ZMQ) enables fast inter-process communication. Messages must be serialized
to bytes to cross process boundaries. Here's how it works:

```mermaid
flowchart LR
    A[UserMsg Object] -->|"serialize_type()"| B[Dict with __type__]
    B -->|"msgpack.packb()"| C[Raw Bytes]
    C -->|"ZMQ Socket"| D[Other Process]
    D -->|"msgpack.unpackb()"| E[Dict]
    E -->|"deserialize_type()"| F[UserMsg Object]
```

Key Points:
1. **Serialization**: Dataclass → Dict → Bytes (via msgpack)
2. **CPU Tensor Requirement**: GPU tensors cannot be serialized (process boundaries)
3. **Tensor Serialization**: CPU tensor → numpy → bytes → numpy → CPU tensor
4. **Raw Bytes Broadcasting**: Multi-GPU setups broadcast raw bytes to avoid re-serialization

Reference Implementation:
- `python/minisgl/message/utils.py:20-35` - serialize_type() converts objects to dicts
- `python/minisgl/message/utils.py:24-29` - Tensor serialization (CPU only, via numpy)
- `python/minisgl/utils/mp.py:24-26` - msgpack encoding for ZMQ
- `python/minisgl/message/backend.py:14-15` - UserMsg encoder method

================================================================================
CPU/GPU DATA MOVEMENT: Where Tensors Live and Move
================================================================================

Tensors move between CPU and GPU at specific points in the system:

```mermaid
flowchart TD
    A[UserMsg: CPU tensor] -->|"ZMQ message"| B[Scheduler receives]
    B -->|"Create Req"| C[Req: CPU tensor]
    C -->|"pin_memory + copy_"| D[token_pool: GPU tensor]
    D -->|"Load for batch"| E[Batch.input_ids: GPU]
    E -->|"Model forward"| F[Output: GPU tensor]
    F -->|"to('cpu')"| G[DetokenizeMsg: CPU tensor]
```

Device Transitions:
1. **CPU → CPU**: UserMsg → Req (stays on CPU for storage)
2. **CPU → GPU**: Req.input_ids → token_pool (via pin_memory + non_blocking copy)
3. **GPU → GPU**: token_pool → Batch.input_ids (loaded from GPU memory)
4. **GPU → CPU**: Model output → DetokenizeMsg (for detokenization)

Reference Implementation:
- `python/minisgl/scheduler/prefill.py:78-79` - CPU→GPU transfer with pin_memory
- `python/minisgl/scheduler/scheduler.py:211-212` - Loading from GPU token_pool
- `python/minisgl/engine/engine.py:199-200` - GPU→CPU output transfer

================================================================================
MULTI-RANK BROADCASTING: Efficient Message Distribution
================================================================================

In multi-GPU setups, Rank 0 receives messages and broadcasts to other ranks:

```mermaid
flowchart TD
    A[Tokenizer sends UserMsg] -->|"ZMQ"| B[Rank 0 receives]
    B -->|"get_raw()"| C[Extract raw bytes]
    C -->|"put_raw()"| D[Broadcast to ranks 1-N]
    D -->|"decode()"| E[All ranks decode UserMsg]
```

Why Raw Bytes?
- Avoids re-serialization overhead (serialize once, decode many times)
- Ensures all ranks see identical messages
- More efficient than re-encoding for each rank

Reference Implementation:
- `python/minisgl/scheduler/io.py:88-106` - Rank 0 broadcast logic
- `python/minisgl/scheduler/io.py:92-94` - get_raw() and put_raw() usage
- `python/minisgl/scheduler/io.py:109-122` - Other ranks receiving
- `python/minisgl/utils/mp.py:70-74` - get_raw() and decode() methods

================================================================================
EXERCISE 1: Trace ZMQ Serialization - [Interactive]
================================================================================

TASK: Implement `trace_zmq_serialization()` to understand how messages are serialized
for ZMQ transmission.

This exercise demonstrates the complete serialization pipeline:
1. UserMsg object → Dict (via serialize_type pattern)
2. Dict → Bytes (via msgpack)
3. Bytes → Dict (via msgpack)
4. Dict → UserMsg object (via deserialize_type pattern)

INSTRUCTIONS:
1. Implement `trace_zmq_serialization(user_msg: UserMsg) -> dict`
2. The function should:
   - Serialize UserMsg to dict (following serialize_type pattern)
   - Encode dict to bytes (using msgpack.packb)
   - Decode bytes back to dict (using msgpack.unpackb)
   - Deserialize dict back to UserMsg (following deserialize_type pattern)
   - Verify CPU tensor requirement throughout
   - Return dictionary with all intermediate steps

HINT:
- See `python/minisgl/message/utils.py:20-35` for serialize_type implementation
- See `python/minisgl/message/utils.py:52-69` for deserialize_type implementation
- See `python/minisgl/utils/mp.py:24-26` for msgpack encoding pattern
- Tensor serialization: `tensor.numpy().tobytes()` for CPU tensors only
- Tensor deserialization: `np.frombuffer()` → `torch.from_numpy()`
- Cross-reference: Verify against actual serialization in `python/minisgl/message/backend.py:14-15`

LEARNING GOAL: Understand why CPU tensors are required for ZMQ serialization
"""


def _serialize_any(value: Any) -> Any:
    """
    Helper function to serialize any value recursively.
    Matches the pattern from python/minisgl/message/utils.py:9-17
    """
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    elif isinstance(value, torch.Tensor):
        # Tensor serialization - CPU only!
        assert value.is_cpu, "Tensors must be on CPU for serialization"
        assert value.dim() == 1, "We can only serialize 1D tensors"
        return {
            "__type__": "Tensor",
            "buffer": value.numpy().tobytes(),
            "dtype": str(value.dtype)
        }
    else:
        # For dataclass objects, use serialize_type pattern
        return _serialize_type(value)


def _serialize_type(obj: Any) -> Dict:
    """
    Serialize an object to a dictionary.
    Matches the pattern from python/minisgl/message/utils.py:20-35
    """
    serialized = {}
    serialized["__type__"] = obj.__class__.__name__
    for k, v in obj.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, type], data: Any) -> Any:
    """
    Helper function to deserialize any value recursively.
    Matches the pattern from python/minisgl/message/utils.py:38-49
    """
    if isinstance(data, dict):
        if "__type__" in data:
            return _deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def _deserialize_type(cls_map: Dict[str, type], data: Dict) -> Any:
    """
    Deserialize a dictionary back to an object.
    Matches the pattern from python/minisgl/message/utils.py:52-69
    """
    type_name = data["__type__"]

    # Handle Tensor deserialization
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        # Always deserialize to CPU tensor
        return torch.from_numpy(np_tensor.copy())

    # Handle regular dataclass objects
    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)


def trace_zmq_serialization(user_msg: "UserMsg") -> dict:
    """
    Trace the complete ZMQ serialization/deserialization pipeline.

    This function demonstrates:
    1. How UserMsg is serialized to a dictionary
    2. How the dictionary is encoded to bytes (for ZMQ)
    3. How bytes are decoded back to a dictionary
    4. How the dictionary is deserialized back to UserMsg
    5. Why CPU tensors are required

    Args:
        user_msg: The UserMsg to serialize/deserialize

    Returns:
        Dictionary with all serialization steps and verification results

    TODO: Implement this function
    """
    # Step 1: Verify input has CPU tensor (required for serialization)
    assert user_msg.input_ids.is_cpu, "input_ids must be on CPU for ZMQ serialization"

    # Step 2: Serialize UserMsg to dict
    #   - Use _serialize_type() to convert UserMsg to dict
    #   - This matches the pattern in python/minisgl/message/utils.py:20-35
    serialized_dict = _serialize_type(user_msg)

    # Step 3: Encode dict to bytes (simulating msgpack encoding)
    #   - Use msgpack.packb() to encode the dictionary
    #   - This matches the pattern in python/minisgl/utils/mp.py:24-26
    encoded_bytes = msgpack.packb(serialized_dict, use_bin_type=True)

    # Step 4: Decode bytes back to dict (simulating msgpack decoding)
    #   - Use msgpack.unpackb() to decode the bytes
    decoded_dict = msgpack.unpackb(encoded_bytes, raw=False)

    # Step 5: Deserialize dict back to UserMsg
    #   - Use _deserialize_type() to reconstruct UserMsg
    #   - This matches the pattern in python/minisgl/message/utils.py:52-69
    deserialized_msg = _deserialize_type(cls_map, decoded_dict)

    # Step 6: Verify deserialized message
    #   - Check UID matches
    #   - Check tensor is on CPU (deserialization always creates CPU tensors)
    #   - Check tensor values match
    uid_matches = deserialized_msg.uid == user_msg.uid
    tensor_is_cpu = deserialized_msg.input_ids.is_cpu
    tensor_values_match = torch.equal(deserialized_msg.input_ids, user_msg.input_ids)
    sampling_params_match = (
        deserialized_msg.sampling_params.max_tokens == user_msg.sampling_params.max_tokens
    )

    # Step 7: Return comprehensive results
    return {
        "original_msg": user_msg,
        "serialized_dict": serialized_dict,
        "encoded_bytes": encoded_bytes,
        "decoded_dict": decoded_dict,
        "deserialized_msg": deserialized_msg,
        "cpu_tensor_required": True,  # CPU tensor is required for serialization
        "serialization_successful": uid_matches and tensor_is_cpu and tensor_values_match and sampling_params_match,
        "uid_matches": uid_matches,
        "tensor_is_cpu": tensor_is_cpu,
        "tensor_values_match": tensor_values_match
    }


"""
================================================================================
EXERCISE 2: Trace CPU to GPU Movement - [Interactive]
================================================================================

TASK: Implement `trace_cpu_to_gpu_movement()` to track how tensors move between
CPU and GPU throughout the system.

This exercise demonstrates the device transitions:
1. UserMsg: CPU tensor (from ZMQ)
2. Req: CPU tensor (stored in scheduler)
3. token_pool: GPU tensor (moved via pin_memory)
4. Batch.input_ids: GPU tensor (loaded from token_pool)
5. Model output: GPU tensor
6. DetokenizeMsg: CPU tensor (moved back for detokenization)

INSTRUCTIONS:
1. Implement `trace_cpu_to_gpu_movement(user_msg: UserMsg) -> dict`
2. The function should:
   - Start with CPU tensor in UserMsg (verify `.is_cpu`)
   - Simulate Req creation (verify stays CPU)
   - Simulate move to GPU token_pool (create GPU tensor, verify `.device.type == "cuda"`)
   - Simulate loading from GPU (verify tensor is on GPU)
   - Simulate output move back to CPU (verify `.is_cpu` after `.to("cpu")`)
   - Return dictionary tracking device at each step

HINT:
- See `python/minisgl/scheduler/prefill.py:78-79` for CPU→GPU transfer with pin_memory
- See `python/minisgl/scheduler/scheduler.py:211-212` for loading from GPU token_pool
- See `python/minisgl/engine/engine.py:199-200` for GPU→CPU output transfer
- Use `.is_cpu` to check CPU placement
- Use `.device.type == "cuda"` to check GPU placement
- Use `torch.cuda.is_available()` to check if GPU is available
- Cross-reference: Verify device transitions match actual implementation

LEARNING GOAL: Understand where and why tensors move between CPU and GPU
"""


def trace_cpu_to_gpu_movement(user_msg: "UserMsg") -> dict:
    """
    Trace how tensors move between CPU and GPU throughout the system.

    This function simulates the device transitions:
    1. UserMsg: CPU tensor (from ZMQ message)
    2. Req: CPU tensor (stored in scheduler, not yet on GPU)
    3. token_pool: GPU tensor (moved via pin_memory + non_blocking copy)
    4. Batch.input_ids: GPU tensor (loaded from GPU token_pool)
    5. Model output: GPU tensor (from model forward pass)
    6. DetokenizeMsg: CPU tensor (moved back for detokenization)

    Args:
        user_msg: The UserMsg with CPU tensor

    Returns:
        Dictionary tracking device placement at each step

    TODO: Implement this function
    """
    # Step 1: Verify UserMsg has CPU tensor
    assert user_msg.input_ids.is_cpu, "UserMsg.input_ids must be on CPU (from ZMQ)"
    step1_device = "cpu"
    step1_tensor = user_msg.input_ids

    # Step 2: Simulate Req creation (stays on CPU)
    #   - In real system: Req.input_ids is CPU tensor (see python/minisgl/core.py:29)
    #   - Req stores CPU tensor until it's moved to token_pool
    req_input_ids = user_msg.input_ids  # Still CPU
    assert req_input_ids.is_cpu, "Req.input_ids must be on CPU"
    step2_device = "cpu"
    step2_tensor = req_input_ids

    # Step 3: Simulate move to GPU token_pool
    #   - In real system: token_pool is on GPU (see python/minisgl/scheduler/prefill.py:78-79)
    #   - Uses pin_memory() + copy_() with non_blocking=True
    step3_device = None
    step3_tensor = None
    if torch.cuda.is_available():
        # Simulate GPU token_pool storage
        # In real system: device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        gpu_token_pool = req_input_ids.cuda()  # Move to GPU
        assert not gpu_token_pool.is_cpu, "token_pool should be on GPU"
        assert gpu_token_pool.device.type == "cuda", "token_pool device should be CUDA"
        step3_device = "cuda"
        step3_tensor = gpu_token_pool
    else:
        step3_device = "cpu"  # No GPU available, stays on CPU
        step3_tensor = req_input_ids

    # Step 4: Simulate loading from GPU token_pool for Batch
    #   - In real system: input.batch.input_ids = self.token_pool.view(-1)[input.load_indices]
    #   - This loads from GPU token_pool, so tensor is on GPU
    step4_device = step3_device  # Same device as token_pool
    step4_tensor = step3_tensor  # Loaded from token_pool

    # Step 5: Simulate model output (on GPU)
    #   - In real system: Model forward pass produces GPU tensors
    step5_device = step4_device  # Model output on same device as input
    if step5_device == "cuda" and torch.cuda.is_available():
        step5_tensor = step4_tensor  # Simulate GPU output (in real system, this is logits)
    else:
        step5_tensor = step4_tensor  # CPU fallback

    # Step 6: Simulate move back to CPU for detokenization
    #   - In real system: next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
    #   - See python/minisgl/engine/engine.py:199-200
    if step5_device == "cuda" and torch.cuda.is_available():
        cpu_output = step5_tensor.cpu()  # Move back to CPU
        assert cpu_output.is_cpu, "DetokenizeMsg should have CPU tensor"
        step6_device = "cpu"
        step6_tensor = cpu_output
    else:
        step6_device = "cpu"
        step6_tensor = step5_tensor

    # Step 7: Return device tracking results
    return {
        "step1_usermsg": {
            "device": step1_device,
            "is_cpu": step1_tensor.is_cpu,
            "tensor_shape": tuple(step1_tensor.shape)
        },
        "step2_req": {
            "device": step2_device,
            "is_cpu": step2_tensor.is_cpu,
            "tensor_shape": tuple(step2_tensor.shape)
        },
        "step3_token_pool": {
            "device": step3_device,
            "is_cpu": step3_tensor.is_cpu if step3_tensor is not None else None,
            "tensor_shape": tuple(step3_tensor.shape) if step3_tensor is not None else None
        },
        "step4_batch_input": {
            "device": step4_device,
            "is_cpu": step4_tensor.is_cpu if step4_tensor is not None else None,
            "tensor_shape": tuple(step4_tensor.shape) if step4_tensor is not None else None
        },
        "step5_model_output": {
            "device": step5_device,
            "is_cpu": step5_tensor.is_cpu if step5_tensor is not None else None,
            "tensor_shape": tuple(step5_tensor.shape) if step5_tensor is not None else None
        },
        "step6_detokenize": {
            "device": step6_device,
            "is_cpu": step6_tensor.is_cpu,
            "tensor_shape": tuple(step6_tensor.shape)
        },
        "gpu_available": torch.cuda.is_available(),
        "device_transitions": f"{step1_device} → {step2_device} → {step3_device} → {step4_device} → {step5_device} → {step6_device}"
    }


"""
================================================================================
EXERCISE 3: Trace Multi-Rank Broadcast - [Interactive]
================================================================================

TASK: Implement `trace_multi_rank_broadcast()` to understand how messages are
broadcast to multiple GPU ranks efficiently.

This exercise demonstrates:
1. Rank 0 receives UserMsg from tokenizer
2. Rank 0 extracts raw bytes (avoiding re-serialization)
3. Raw bytes are broadcast to other ranks
4. All ranks decode the same bytes to UserMsg
5. All ranks see identical messages

INSTRUCTIONS:
1. Implement `trace_multi_rank_broadcast(user_msg: UserMsg, num_ranks: int) -> dict`
2. The function should:
   - Simulate Rank 0 receiving UserMsg
   - Extract raw bytes (simulate get_raw())
   - Broadcast bytes to other ranks (simulated)
   - Decode bytes on all ranks (simulate decode())
   - Verify all ranks see identical UserMsg
   - Verify CPU tensor requirement after decode
   - Return dictionary with broadcast results

HINT:
- See `python/minisgl/scheduler/io.py:88-106` for Rank 0 broadcast logic
- See `python/minisgl/scheduler/io.py:92-94` for get_raw() and put_raw() usage
- See `python/minisgl/scheduler/io.py:109-122` for other ranks receiving
- See `python/minisgl/utils/mp.py:70-74` for get_raw() and decode() methods
- Raw bytes = msgpack encoded message (already serialized)
- Broadcasting raw bytes avoids re-serialization overhead
- Cross-reference: Verify against actual broadcast implementation

LEARNING GOAL: Understand efficient multi-GPU message distribution
"""


def trace_multi_rank_broadcast(user_msg: "UserMsg", num_ranks: int = 4) -> dict:
    """
    Trace how messages are broadcast to multiple GPU ranks.

    This function simulates:
    1. Rank 0 receives UserMsg from tokenizer
    2. Rank 0 extracts raw bytes (get_raw())
    3. Raw bytes are broadcast to ranks 1, 2, 3, ...
    4. All ranks decode bytes to UserMsg (decode())
    5. All ranks verify they see identical message

    Args:
        user_msg: The UserMsg received by Rank 0
        num_ranks: Number of ranks to simulate (default: 4)

    Returns:
        Dictionary with broadcast results and verification

    TODO: Implement this function
    """
    # Step 1: Rank 0 receives UserMsg
    #   - In real system: self._recv_from_tokenizer.get() (see python/minisgl/scheduler/io.py:83)
    rank0_msg = user_msg
    assert rank0_msg.input_ids.is_cpu, "UserMsg must have CPU tensor"

    # Step 2: Rank 0 extracts raw bytes (simulate get_raw())
    #   - In real system: raw = self._recv_from_tokenizer.get_raw() (see python/minisgl/scheduler/io.py:92)
    #   - This gets the already-serialized bytes (msgpack encoded)
    serialized_dict = _serialize_type(rank0_msg)
    raw_bytes = msgpack.packb(serialized_dict, use_bin_type=True)
    cls_map = {"UserMsg": type(rank0_msg), "SamplingParams": type(rank0_msg.sampling_params)}

    # Step 3: Broadcast raw bytes to other ranks (simulated)
    #   - In real system: self._send_into_ranks.put_raw(raw) (see python/minisgl/scheduler/io.py:93)
    #   - Broadcasting raw bytes avoids re-serialization overhead
    broadcast_bytes = raw_bytes  # Same bytes sent to all ranks

    # Step 4: All ranks decode bytes to UserMsg (simulate decode())
    #   - In real system: self._recv_from_rank0.decode(raw) (see python/minisgl/scheduler/io.py:121)
    cls_map = {"UserMsg": type(rank0_msg), "SamplingParams": type(rank0_msg.sampling_params)}
    decoded_msgs = []
    for rank in range(num_ranks):
        # Each rank decodes the same raw bytes
        decoded_dict = msgpack.unpackb(broadcast_bytes, raw=False)
        decoded_msg = _deserialize_type(cls_map, decoded_dict)
        decoded_msgs.append(decoded_msg)

    # Step 5: Verify all ranks see identical message
    #   - Check UID matches across all ranks
    #   - Check tensor values match
    #   - Check all tensors are on CPU (deserialization creates CPU tensors)
    all_uids_match = all(msg.uid == rank0_msg.uid for msg in decoded_msgs)
    all_tensors_match = all(torch.equal(msg.input_ids, rank0_msg.input_ids) for msg in decoded_msgs)
    all_tensors_cpu = all(msg.input_ids.is_cpu for msg in decoded_msgs)
    all_sampling_params_match = all(
        msg.sampling_params.max_tokens == rank0_msg.sampling_params.max_tokens
        for msg in decoded_msgs
    )

    # Step 6: Return broadcast results
    return {
        "rank0_received": {
            "uid": rank0_msg.uid,
            "tensor_shape": tuple(rank0_msg.input_ids.shape),
            "is_cpu": rank0_msg.input_ids.is_cpu
        },
        "raw_bytes_size": len(raw_bytes),
        "num_ranks": num_ranks,
        "all_ranks_decoded": len(decoded_msgs) == num_ranks,
        "all_uids_match": all_uids_match,
        "all_tensors_match": all_tensors_match,
        "all_tensors_cpu": all_tensors_cpu,
        "all_sampling_params_match": all_sampling_params_match,
        "broadcast_successful": all_uids_match and all_tensors_match and all_tensors_cpu and all_sampling_params_match,
        "rank_messages": [
            {
                "rank": i,
                "uid": msg.uid,
                "tensor_shape": tuple(msg.input_ids.shape),
                "is_cpu": msg.input_ids.is_cpu
            }
            for i, msg in enumerate(decoded_msgs)
        ]
    }


"""
================================================================================
EXERCISE 4: Trace Complete Request Lifecycle - [Comprehensive Interactive]
================================================================================

TASK: Implement `trace_complete_request_lifecycle()` to understand the complete
end-to-end flow from HTTP request to HTTP response.

This exercise ties together all concepts:
- HTTP request → TokenizeMsg
- TokenizeMsg → UserMsg (tokenization)
- UserMsg → Req (scheduler)
- Req → Batch → Engine (GPU computation)
- Engine → DetokenizeMsg → UserReply
- UserReply → HTTP response
- Track UID through all steps
- Track CPU/GPU transitions at each step

INSTRUCTIONS:
1. Implement `trace_complete_request_lifecycle(uid: int, text: str, max_tokens: int) -> dict`
2. The function should:
   - Simulate complete request lifecycle through all components
   - Track UID through all steps
   - Track device (CPU/GPU) at each step
   - Verify data transformations
   - Return comprehensive dictionary with all steps

HINT:
- See `python/minisgl/server/api_server.py:245-277` for HTTP endpoint
- See `python/minisgl/tokenizer/server.py:84-98` for tokenization
- See `python/minisgl/scheduler/scheduler.py:155-175` for UserMsg processing
- See `python/minisgl/scheduler/scheduler.py:180-201` for batch preparation
- See `python/minisgl/engine/engine.py:188-203` for forward pass
- Use functions from puzzle_3.1: create_tokenize_msg, create_user_msg, _simulate_tokenization
- Cross-reference: Verify each step matches actual implementation

LEARNING GOAL: Complete understanding of end-to-end message flow and device transitions
"""


def _simulate_tokenization(text: str) -> torch.Tensor:
    """
    Simulate tokenization by creating a simple tensor.
    In real system, this would use actual tokenizer.
    """
    num_tokens = max(1, len(text) // 3)
    return torch.tensor(list(range(1, num_tokens + 1)), dtype=torch.int32)


def trace_complete_request_lifecycle(uid: int, text: str, max_tokens: int) -> dict:
    """
    Trace a complete request through the entire system.

    This function simulates the full lifecycle:
    1. HTTP request → API Server creates TokenizeMsg
    2. TokenizeMsg → Tokenizer creates UserMsg
    3. UserMsg → Scheduler creates Req
    4. Req → Batch → Engine (GPU computation)
    5. Engine → DetokenizeMsg → Detokenizer creates UserReply
    6. UserReply → API Server streams HTTP response

    Args:
        uid: Unique request ID
        text: Input text to process
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with complete lifecycle steps and device tracking

    TODO: Implement this function
    """
    # Import helper functions from puzzle_3.1
    import sys
    import importlib.util
    from pathlib import Path

    puzzle_dir = Path(__file__).parent
    puzzle_3_1_file = puzzle_dir / "puzzle_3.1.py"
    spec = importlib.util.spec_from_file_location("puzzle_3_1", puzzle_3_1_file)
    puzzle_3_1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(puzzle_3_1)
    create_tokenize_msg = puzzle_3_1.create_tokenize_msg
    create_user_msg = puzzle_3_1.create_user_msg

    # Step 1: HTTP Request → API Server creates TokenizeMsg
    #   - See python/minisgl/server/api_server.py:245-277
    tokenize_msg = create_tokenize_msg(uid=uid, text=text, max_tokens=max_tokens)
    step1_device = "cpu"  # TokenizeMsg has text (str), no tensor yet

    # Step 2: TokenizeMsg → Tokenizer creates UserMsg
    #   - See python/minisgl/tokenizer/server.py:84-98
    input_ids = _simulate_tokenization(text)
    user_msg = create_user_msg(uid=uid, input_ids=input_ids, max_tokens=max_tokens)
    assert user_msg.input_ids.is_cpu, "UserMsg.input_ids must be on CPU (from ZMQ)"
    step2_device = "cpu"

    # Step 3: UserMsg → Scheduler creates Req
    #   - See python/minisgl/scheduler/scheduler.py:155-175
    #   - Req.input_ids stays on CPU (see python/minisgl/core.py:29)
    req_input_ids = user_msg.input_ids  # Still CPU
    assert req_input_ids.is_cpu, "Req.input_ids must be on CPU"
    step3_device = "cpu"

    # Step 4: Req → Batch → Engine (GPU computation)
    #   - See python/minisgl/scheduler/scheduler.py:180-201 for batch preparation
    #   - See python/minisgl/scheduler/prefill.py:78-79 for CPU→GPU transfer
    #   - See python/minisgl/engine/engine.py:188-203 for forward pass
    step4_device = "cpu"
    step4_tensor = req_input_ids
    if torch.cuda.is_available():
        # Simulate move to GPU token_pool and batch
        batch_input_ids = req_input_ids.cuda()
        assert not batch_input_ids.is_cpu, "Batch.input_ids should be on GPU"
        step4_device = "cuda"
        step4_tensor = batch_input_ids
    else:
        step4_device = "cpu"
        step4_tensor = req_input_ids

    # Step 5: Engine → DetokenizeMsg (model output)
    #   - See python/minisgl/engine/engine.py:199-200
    #   - Model output is on GPU, moved to CPU for detokenization
    if step4_device == "cuda" and torch.cuda.is_available():
        # Simulate model output (on GPU)
        model_output_gpu = step4_tensor  # In real system, this is logits/sampled tokens
        # Move to CPU for detokenization
        model_output_cpu = model_output_gpu.cpu()
        assert model_output_cpu.is_cpu, "DetokenizeMsg should have CPU tensor"
        step5_device = "cpu"
        step5_tensor = model_output_cpu
    else:
        step5_device = "cpu"
        step5_tensor = step4_tensor

    # Step 6: DetokenizeMsg → Detokenizer creates UserReply
    #   - See python/minisgl/tokenizer/server.py:65-82
    #   - UserReply has text (str), no tensor
    step6_device = "cpu"  # UserReply has incremental_output (str)

    # Step 7: UserReply → API Server streams HTTP response
    #   - See python/minisgl/server/api_server.py:151-156
    step7_device = "cpu"  # HTTP response is text

    # Step 8: Verify UID tracking through all steps
    uid_tracked = (
        tokenize_msg.uid == uid and
        user_msg.uid == uid and
        tokenize_msg.uid == user_msg.uid
    )

    # Step 9: Verify device transitions
    device_transitions = f"{step1_device} → {step2_device} → {step3_device} → {step4_device} → {step5_device} → {step6_device} → {step7_device}"

    # Step 10: Return comprehensive lifecycle results
    return {
        "step1_http_request": {
            "component": "API Server",
            "message_type": "TokenizeMsg",
            "device": step1_device,
            "uid": tokenize_msg.uid,
            "has_tensor": False
        },
        "step2_tokenization": {
            "component": "Tokenizer Worker",
            "message_type": "UserMsg",
            "device": step2_device,
            "uid": user_msg.uid,
            "has_tensor": True,
            "tensor_is_cpu": user_msg.input_ids.is_cpu,
            "tensor_shape": tuple(user_msg.input_ids.shape)
        },
        "step3_scheduler": {
            "component": "Scheduler",
            "data_type": "Req",
            "device": step3_device,
            "uid": uid,
            "has_tensor": True,
            "tensor_is_cpu": req_input_ids.is_cpu,
            "tensor_shape": tuple(req_input_ids.shape)
        },
        "step4_batch_engine": {
            "component": "Engine",
            "data_type": "Batch",
            "device": step4_device,
            "uid": uid,
            "has_tensor": True,
            "tensor_is_cpu": step4_tensor.is_cpu,
            "tensor_shape": tuple(step4_tensor.shape) if step4_tensor is not None else None
        },
        "step5_detokenize": {
            "component": "Detokenizer Worker",
            "message_type": "DetokenizeMsg",
            "device": step5_device,
            "uid": uid,
            "has_tensor": True,
            "tensor_is_cpu": step5_tensor.is_cpu,
            "tensor_shape": tuple(step5_tensor.shape) if step5_tensor is not None else None
        },
        "step6_user_reply": {
            "component": "Detokenizer Worker",
            "message_type": "UserReply",
            "device": step6_device,
            "uid": uid,
            "has_tensor": False
        },
        "step7_http_response": {
            "component": "API Server",
            "message_type": "HTTP Response",
            "device": step7_device,
            "uid": uid,
            "has_tensor": False
        },
        "uid_tracked": uid_tracked,
        "device_transitions": device_transitions,
        "gpu_available": torch.cuda.is_available(),
        "lifecycle_complete": True
    }


"""
================================================================================
CHALLENGE
================================================================================

- Understand ZMQ serialization mechanism
- Track CPU/GPU tensor movements
- Understand multi-rank broadcasting
- Trace complete request lifecycle

HINT:
- Reference: `python/minisgl/message/utils.py` for serialization logic
- Reference: `python/minisgl/utils/mp.py` for ZMQ queue implementations
- Reference: `python/minisgl/scheduler/io.py` for multi-rank message handling
- Reference: `python/minisgl/scheduler/prefill.py:78-79` for CPU→GPU transfer
- Reference: `python/minisgl/scheduler/scheduler.py:211-212` for GPU token loading
- Reference: `python/minisgl/engine/engine.py:199-200` for GPU→CPU output transfer
- Reference: `python/minisgl/server/api_server.py:245-277` for HTTP endpoint
- Reference: `python/minisgl/tokenizer/server.py:84-98` for tokenization

QUESTIONS:
1. Why can't GPU tensors be serialized for ZMQ?
2. How does raw byte broadcasting improve efficiency?
3. Where exactly does data move from CPU to GPU?
4. Why is pin_memory used for CPU→GPU transfers?
5. How does the system ensure all ranks see identical messages?

TEST:
Run: pytest learning/puzzles/03_messages/test_3.2.py -v
"""
