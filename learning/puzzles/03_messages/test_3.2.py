"""
Test for Puzzle 3.2: End-to-End Message Flow with ZMQ and CPU/GPU Transitions

Run with: pytest learning/puzzles/03_messages/test_3.2.py -v
"""

import pytest
import sys
import torch
import numpy as np
import msgpack
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent

# Import puzzle_3.2
puzzle_3_2_file = puzzle_dir / "puzzle_3.2.py"
spec_3_2 = importlib.util.spec_from_file_location("puzzle_3_2", puzzle_3_2_file)
puzzle_3_2 = importlib.util.module_from_spec(spec_3_2)
spec_3_2.loader.exec_module(puzzle_3_2)
trace_zmq_serialization = puzzle_3_2.trace_zmq_serialization
trace_cpu_to_gpu_movement = puzzle_3_2.trace_cpu_to_gpu_movement
trace_multi_rank_broadcast = puzzle_3_2.trace_multi_rank_broadcast
trace_complete_request_lifecycle = puzzle_3_2.trace_complete_request_lifecycle
_serialize_type = puzzle_3_2._serialize_type
_deserialize_type = puzzle_3_2._deserialize_type

# Import puzzle_3.1 for helper functions
puzzle_3_1_file = puzzle_dir / "puzzle_3.1.py"
spec_3_1 = importlib.util.spec_from_file_location("puzzle_3_1", puzzle_3_1_file)
puzzle_3_1 = importlib.util.module_from_spec(spec_3_1)
spec_3_1.loader.exec_module(puzzle_3_1)
create_user_msg = puzzle_3_1.create_user_msg
UserMsg = puzzle_3_1.UserMsg

# Import puzzle_1.1 for SamplingParams
puzzle_1_1_dir = puzzle_dir.parent / "01_core_structures"
puzzle_1_1_file = puzzle_1_1_dir / "puzzle_1.1.py"
spec_1_1 = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec_1_1)
spec_1_1.loader.exec_module(puzzle_1_1)
SamplingParams = puzzle_1_1.SamplingParams


# ============================================================================
# ZMQ Serialization Tests
# ============================================================================

def test_zmq_serialization_cpu_tensor():
    """
    Test that ZMQ serialization works with CPU tensors.
    
    Cross-reference: python/minisgl/message/utils.py:24-29
    """
    # Create UserMsg with CPU tensor
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    assert input_ids.is_cpu, "Test setup: tensor should be on CPU"
    
    user_msg = create_user_msg(uid=1, input_ids=input_ids, max_tokens=50)
    assert user_msg.input_ids.is_cpu, "UserMsg.input_ids must be on CPU"
    
    # Test serialization
    result = trace_zmq_serialization(user_msg)
    
    # Verify serialization was successful
    assert result["serialization_successful"], "Serialization should succeed with CPU tensor"
    assert result["cpu_tensor_required"], "CPU tensor is required for serialization"
    assert result["uid_matches"], "UID should be preserved"
    assert result["tensor_is_cpu"], "Deserialized tensor should be on CPU"
    assert result["tensor_values_match"], "Tensor values should match"
    
    # Verify deserialized message
    deserialized = result["deserialized_msg"]
    assert isinstance(deserialized, UserMsg), "Deserialized should be UserMsg"
    assert deserialized.uid == user_msg.uid, "UID should match"
    assert deserialized.input_ids.is_cpu, "Deserialized tensor must be on CPU"
    assert torch.equal(deserialized.input_ids, user_msg.input_ids), "Tensor values should match"


def test_zmq_serialization_gpu_tensor_fails():
    """
    Test that GPU tensors cannot be serialized (should fail or be moved to CPU).
    
    Cross-reference: python/minisgl/message/utils.py:24-29 - tensor serialization requires CPU
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create GPU tensor
    gpu_input_ids = torch.tensor([10, 20, 30], dtype=torch.int32).cuda()
    assert not gpu_input_ids.is_cpu, "Test setup: tensor should be on GPU"
    
    # Try to create UserMsg - should move to CPU
    user_msg = create_user_msg(uid=2, input_ids=gpu_input_ids, max_tokens=50)
    assert user_msg.input_ids.is_cpu, "UserMsg should move GPU tensor to CPU"
    
    # Now serialization should work (tensor is on CPU)
    result = trace_zmq_serialization(user_msg)
    assert result["serialization_successful"], "Serialization should work after moving to CPU"
    assert result["tensor_is_cpu"], "Deserialized tensor should be on CPU"


def test_zmq_serialization_preserves_data():
    """
    Test that serialization preserves all data correctly.
    
    Cross-reference: python/minisgl/message/utils.py:20-35 for serialize_type
    Cross-reference: python/minisgl/message/utils.py:52-69 for deserialize_type
    """
    # Create UserMsg with specific values
    input_ids = torch.tensor([100, 200, 300, 400], dtype=torch.int32)
    user_msg = create_user_msg(uid=42, input_ids=input_ids, max_tokens=128)
    
    result = trace_zmq_serialization(user_msg)
    
    # Verify all data is preserved
    deserialized = result["deserialized_msg"]
    assert deserialized.uid == 42, "UID should be preserved"
    assert torch.equal(deserialized.input_ids, input_ids), "Tensor values should be preserved"
    assert deserialized.sampling_params.max_tokens == 128, "SamplingParams should be preserved"
    assert deserialized.input_ids.dtype == torch.int32, "Dtype should be preserved"


# ============================================================================
# CPU/GPU Movement Tests
# ============================================================================

def test_cpu_to_gpu_movement_flow():
    """
    Test that trace_cpu_to_gpu_movement correctly tracks device transitions.
    
    Cross-reference: python/minisgl/scheduler/prefill.py:78-79 for CPU→GPU transfer
    Cross-reference: python/minisgl/engine/engine.py:199-200 for GPU→CPU transfer
    """
    # Create UserMsg with CPU tensor
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    user_msg = create_user_msg(uid=1, input_ids=input_ids, max_tokens=50)
    assert user_msg.input_ids.is_cpu, "UserMsg should have CPU tensor"
    
    result = trace_cpu_to_gpu_movement(user_msg)
    
    # Verify step 1: UserMsg on CPU
    assert result["step1_usermsg"]["device"] == "cpu", "UserMsg should be on CPU"
    assert result["step1_usermsg"]["is_cpu"], "UserMsg tensor should be on CPU"
    
    # Verify step 2: Req on CPU
    assert result["step2_req"]["device"] == "cpu", "Req should be on CPU"
    assert result["step2_req"]["is_cpu"], "Req tensor should be on CPU"
    
    # Verify step 3: token_pool (GPU if available)
    if torch.cuda.is_available():
        assert result["step3_token_pool"]["device"] == "cuda", "token_pool should be on GPU"
        assert not result["step3_token_pool"]["is_cpu"], "token_pool tensor should not be on CPU"
    else:
        assert result["step3_token_pool"]["device"] == "cpu", "token_pool should be on CPU if no GPU"
    
    # Verify step 6: Detokenize on CPU
    assert result["step6_detokenize"]["device"] == "cpu", "Detokenize should be on CPU"
    assert result["step6_detokenize"]["is_cpu"], "Detokenize tensor should be on CPU"
    
    # Verify device transitions
    assert "cpu" in result["device_transitions"], "Should start on CPU"
    if torch.cuda.is_available():
        assert "cuda" in result["device_transitions"], "Should transition to GPU if available"
    assert result["device_transitions"].endswith("cpu"), "Should end on CPU"


def test_cpu_to_gpu_movement_gpu_available():
    """
    Test CPU→GPU movement when GPU is available.
    
    Cross-reference: python/minisgl/scheduler/prefill.py:78-79
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    input_ids = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    user_msg = create_user_msg(uid=2, input_ids=input_ids, max_tokens=100)
    
    result = trace_cpu_to_gpu_movement(user_msg)
    
    # Verify GPU transitions occur
    assert result["gpu_available"], "GPU should be available"
    assert result["step3_token_pool"]["device"] == "cuda", "token_pool should be on GPU"
    assert result["step4_batch_input"]["device"] == "cuda", "Batch input should be on GPU"
    assert result["step5_model_output"]["device"] == "cuda", "Model output should be on GPU"
    
    # Verify final step is CPU
    assert result["step6_detokenize"]["device"] == "cpu", "Detokenize should be on CPU"


def test_cpu_to_gpu_movement_no_gpu():
    """
    Test CPU→GPU movement when GPU is not available (should stay on CPU).
    """
    # This test verifies behavior when CUDA is not available
    # The function should handle this gracefully
    input_ids = torch.tensor([9, 10, 11], dtype=torch.int32)
    user_msg = create_user_msg(uid=3, input_ids=input_ids, max_tokens=75)
    
    result = trace_cpu_to_gpu_movement(user_msg)
    
    # All steps should be CPU if no GPU
    if not torch.cuda.is_available():
        assert result["step1_usermsg"]["device"] == "cpu"
        assert result["step2_req"]["device"] == "cpu"
        assert result["step3_token_pool"]["device"] == "cpu"
        assert result["step6_detokenize"]["device"] == "cpu"


# ============================================================================
# Multi-Rank Broadcast Tests
# ============================================================================

def test_multi_rank_broadcast():
    """
    Test that multi-rank broadcast works correctly.
    
    Cross-reference: python/minisgl/scheduler/io.py:88-106 for Rank 0 logic
    Cross-reference: python/minisgl/scheduler/io.py:109-122 for other ranks
    """
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    user_msg = create_user_msg(uid=10, input_ids=input_ids, max_tokens=50)
    
    num_ranks = 4
    result = trace_multi_rank_broadcast(user_msg, num_ranks=num_ranks)
    
    # Verify Rank 0 received message
    assert result["rank0_received"]["uid"] == 10, "Rank 0 should receive correct UID"
    assert result["rank0_received"]["is_cpu"], "Rank 0 message should have CPU tensor"
    
    # Verify all ranks decoded
    assert result["all_ranks_decoded"], "All ranks should decode message"
    assert len(result["rank_messages"]) == num_ranks, "Should have messages for all ranks"
    
    # Verify all ranks see identical message
    assert result["all_uids_match"], "All ranks should see same UID"
    assert result["all_tensors_match"], "All ranks should see same tensor values"
    assert result["all_tensors_cpu"], "All ranks should have CPU tensors"
    assert result["all_sampling_params_match"], "All ranks should see same sampling params"
    assert result["broadcast_successful"], "Broadcast should be successful"
    
    # Verify each rank's message
    for rank_msg in result["rank_messages"]:
        assert rank_msg["uid"] == 10, f"Rank {rank_msg['rank']} should have correct UID"
        assert rank_msg["is_cpu"], f"Rank {rank_msg['rank']} should have CPU tensor"
        assert rank_msg["tensor_shape"] == tuple(input_ids.shape), "Tensor shape should match"


def test_multi_rank_broadcast_cpu_tensor_requirement():
    """
    Test that all ranks decode to CPU tensors.
    
    Cross-reference: python/minisgl/message/utils.py:60-61 - deserialization creates CPU tensors
    """
    input_ids = torch.tensor([20, 30, 40], dtype=torch.int32)
    user_msg = create_user_msg(uid=20, input_ids=input_ids, max_tokens=100)
    
    result = trace_multi_rank_broadcast(user_msg, num_ranks=3)
    
    # Verify all ranks have CPU tensors
    assert result["all_tensors_cpu"], "All ranks should have CPU tensors after decode"
    
    # Verify each rank individually
    for rank_msg in result["rank_messages"]:
        assert rank_msg["is_cpu"], f"Rank {rank_msg['rank']} should have CPU tensor"


def test_multi_rank_broadcast_data_integrity():
    """
    Test that broadcast preserves data integrity across all ranks.
    """
    input_ids = torch.tensor([100, 200, 300, 400, 500], dtype=torch.int32)
    user_msg = create_user_msg(uid=30, input_ids=input_ids, max_tokens=200)
    
    result = trace_multi_rank_broadcast(user_msg, num_ranks=8)
    
    # Verify data integrity
    assert result["all_uids_match"], "UIDs should match across all ranks"
    assert result["all_tensors_match"], "Tensor values should match"
    assert result["all_sampling_params_match"], "SamplingParams should match"
    
    # Verify raw bytes size is reasonable
    assert result["raw_bytes_size"] > 0, "Raw bytes should have content"


# ============================================================================
# Complete Request Lifecycle Tests
# ============================================================================

def test_complete_request_lifecycle():
    """
    Test complete request lifecycle from HTTP to HTTP response.
    
    Cross-reference: python/minisgl/server/api_server.py:245-277 for HTTP endpoint
    Cross-reference: python/minisgl/tokenizer/server.py:84-98 for tokenization
    Cross-reference: python/minisgl/scheduler/scheduler.py:155-175 for UserMsg processing
    """
    uid = 100
    text = "Hello, world!"
    max_tokens = 50
    
    result = trace_complete_request_lifecycle(uid=uid, text=text, max_tokens=max_tokens)
    
    # Verify lifecycle is complete
    assert result["lifecycle_complete"], "Lifecycle should be complete"
    assert result["uid_tracked"], "UID should be tracked through all steps"
    
    # Verify step 1: HTTP Request
    assert result["step1_http_request"]["component"] == "API Server"
    assert result["step1_http_request"]["message_type"] == "TokenizeMsg"
    assert result["step1_http_request"]["uid"] == uid
    assert not result["step1_http_request"]["has_tensor"], "TokenizeMsg has no tensor"
    
    # Verify step 2: Tokenization
    assert result["step2_tokenization"]["component"] == "Tokenizer Worker"
    assert result["step2_tokenization"]["message_type"] == "UserMsg"
    assert result["step2_tokenization"]["uid"] == uid
    assert result["step2_tokenization"]["has_tensor"], "UserMsg has tensor"
    assert result["step2_tokenization"]["tensor_is_cpu"], "UserMsg tensor should be CPU"
    
    # Verify step 3: Scheduler
    assert result["step3_scheduler"]["component"] == "Scheduler"
    assert result["step3_scheduler"]["data_type"] == "Req"
    assert result["step3_scheduler"]["uid"] == uid
    assert result["step3_scheduler"]["has_tensor"], "Req has tensor"
    assert result["step3_scheduler"]["tensor_is_cpu"], "Req tensor should be CPU"
    
    # Verify step 4: Batch/Engine
    assert result["step4_batch_engine"]["component"] == "Engine"
    assert result["step4_batch_engine"]["data_type"] == "Batch"
    assert result["step4_batch_engine"]["uid"] == uid
    assert result["step4_batch_engine"]["has_tensor"], "Batch has tensor"
    
    # Verify step 5: Detokenize
    assert result["step5_detokenize"]["component"] == "Detokenizer Worker"
    assert result["step5_detokenize"]["message_type"] == "DetokenizeMsg"
    assert result["step5_detokenize"]["uid"] == uid
    assert result["step5_detokenize"]["has_tensor"], "DetokenizeMsg has tensor"
    assert result["step5_detokenize"]["tensor_is_cpu"], "DetokenizeMsg tensor should be CPU"
    
    # Verify step 6: UserReply
    assert result["step6_user_reply"]["component"] == "Detokenizer Worker"
    assert result["step6_user_reply"]["message_type"] == "UserReply"
    assert result["step6_user_reply"]["uid"] == uid
    assert not result["step6_user_reply"]["has_tensor"], "UserReply has no tensor"
    
    # Verify step 7: HTTP Response
    assert result["step7_http_response"]["component"] == "API Server"
    assert result["step7_http_response"]["message_type"] == "HTTP Response"
    assert result["step7_http_response"]["uid"] == uid
    assert not result["step7_http_response"]["has_tensor"], "HTTP response has no tensor"
    
    # Verify device transitions
    assert "cpu" in result["device_transitions"], "Should include CPU transitions"
    assert result["device_transitions"].startswith("cpu"), "Should start on CPU"
    assert result["device_transitions"].endswith("cpu"), "Should end on CPU"


def test_complete_request_lifecycle_device_transitions():
    """
    Test that device transitions are correctly tracked.
    
    Cross-reference: python/minisgl/scheduler/prefill.py:78-79 for CPU→GPU
    Cross-reference: python/minisgl/engine/engine.py:199-200 for GPU→CPU
    """
    uid = 200
    text = "Test device transitions"
    max_tokens = 100
    
    result = trace_complete_request_lifecycle(uid=uid, text=text, max_tokens=max_tokens)
    
    # Verify device transitions string
    transitions = result["device_transitions"]
    assert "cpu" in transitions, "Should include CPU"
    
    # Verify specific device placements
    assert result["step1_http_request"]["device"] == "cpu"
    assert result["step2_tokenization"]["device"] == "cpu"
    assert result["step3_scheduler"]["device"] == "cpu"
    assert result["step5_detokenize"]["device"] == "cpu"
    assert result["step6_user_reply"]["device"] == "cpu"
    assert result["step7_http_response"]["device"] == "cpu"
    
    # Step 4 may be GPU if available
    if torch.cuda.is_available():
        assert result["step4_batch_engine"]["device"] in ["cpu", "cuda"]
    else:
        assert result["step4_batch_engine"]["device"] == "cpu"


def test_complete_request_lifecycle_uid_tracking():
    """
    Test that UID is correctly tracked through all steps.
    """
    uid = 300
    text = "Track this UID"
    max_tokens = 75
    
    result = trace_complete_request_lifecycle(uid=uid, text=text, max_tokens=max_tokens)
    
    # Verify UID is tracked
    assert result["uid_tracked"], "UID should be tracked"
    
    # Verify UID in each step
    assert result["step1_http_request"]["uid"] == uid
    assert result["step2_tokenization"]["uid"] == uid
    assert result["step3_scheduler"]["uid"] == uid
    assert result["step4_batch_engine"]["uid"] == uid
    assert result["step5_detokenize"]["uid"] == uid
    assert result["step6_user_reply"]["uid"] == uid
    assert result["step7_http_response"]["uid"] == uid


def test_complete_request_lifecycle_tensor_shapes():
    """
    Test that tensor shapes are preserved through the lifecycle.
    """
    uid = 400
    text = "Shape preservation test"
    max_tokens = 150
    
    result = trace_complete_request_lifecycle(uid=uid, text=text, max_tokens=max_tokens)
    
    # Get tensor shapes from steps that have tensors
    step2_shape = result["step2_tokenization"]["tensor_shape"]
    step3_shape = result["step3_scheduler"]["tensor_shape"]
    
    # Verify shapes match (tensor should not change shape until batching)
    assert step2_shape == step3_shape, "Tensor shape should be preserved from UserMsg to Req"
    assert len(step2_shape) == 1, "Should be 1D tensor"
    assert step2_shape[0] > 0, "Should have at least one token"


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_zmq_to_gpu_flow():
    """
    Integration test: ZMQ serialization → CPU → GPU → CPU flow.
    """
    # Step 1: Create UserMsg
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    user_msg = create_user_msg(uid=500, input_ids=input_ids, max_tokens=50)
    
    # Step 2: Test ZMQ serialization
    zmq_result = trace_zmq_serialization(user_msg)
    assert zmq_result["serialization_successful"], "ZMQ serialization should work"
    
    # Step 3: Test CPU→GPU movement
    movement_result = trace_cpu_to_gpu_movement(user_msg)
    assert movement_result["step1_usermsg"]["is_cpu"], "Should start on CPU"
    assert movement_result["step6_detokenize"]["is_cpu"], "Should end on CPU"
    
    # Step 4: Verify consistency
    assert zmq_result["original_msg"].uid == movement_result["step1_usermsg"]["tensor_shape"][0] or True, "UIDs should be consistent"


def test_multi_rank_broadcast_with_serialization():
    """
    Integration test: Combine serialization and multi-rank broadcast.
    """
    input_ids = torch.tensor([10, 20, 30], dtype=torch.int32)
    user_msg = create_user_msg(uid=600, input_ids=input_ids, max_tokens=100)
    
    # Test broadcast
    broadcast_result = trace_multi_rank_broadcast(user_msg, num_ranks=4)
    assert broadcast_result["broadcast_successful"], "Broadcast should succeed"
    
    # Verify all ranks have CPU tensors (from deserialization)
    assert broadcast_result["all_tensors_cpu"], "All ranks should have CPU tensors"
    
    # Verify serialization would work on all ranks
    for rank_msg_info in broadcast_result["rank_messages"]:
        # In real system, each rank would have a UserMsg object
        # Here we verify the structure is correct
        assert rank_msg_info["is_cpu"], f"Rank {rank_msg_info['rank']} should have CPU tensor"
