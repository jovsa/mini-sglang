"""
Test for Puzzle 3.1: Create Message Classes

Run with: pytest learning/puzzles/03_messages/test_3.1.py -v
"""

import pytest
import sys
import torch
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent

# Import puzzle_3.1 using importlib (module name has dot)
puzzle_3_1_file = puzzle_dir / "puzzle_3.1.py"
spec_3_1 = importlib.util.spec_from_file_location("puzzle_3_1", puzzle_3_1_file)
puzzle_3_1 = importlib.util.module_from_spec(spec_3_1)
spec_3_1.loader.exec_module(puzzle_3_1)
create_tokenize_msg = puzzle_3_1.create_tokenize_msg
create_user_msg = puzzle_3_1.create_user_msg
TokenizeMsg = puzzle_3_1.TokenizeMsg
UserMsg = puzzle_3_1.UserMsg
trace_message_flow = puzzle_3_1.trace_message_flow

# Import puzzle_1.1 using importlib (module name has dot)
puzzle_1_1_dir = puzzle_dir.parent / "01_core_structures"
puzzle_1_1_file = puzzle_1_1_dir / "puzzle_1.1.py"
spec_1_1 = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec_1_1)
spec_1_1.loader.exec_module(puzzle_1_1)
SamplingParams = puzzle_1_1.SamplingParams


def test_create_tokenize_msg():
    """Test TokenizeMsg creation"""
    msg = create_tokenize_msg(uid=1, text="Hello", max_tokens=50)

    assert msg is not None
    assert msg.uid == 1
    assert msg.text == "Hello"
    assert msg.sampling_params is not None
    assert msg.sampling_params.max_tokens == 50


def test_create_user_msg():
    """Test UserMsg creation"""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    msg = create_user_msg(uid=2, input_ids=input_ids, max_tokens=100)

    assert msg is not None
    assert msg.uid == 2
    assert torch.equal(msg.input_ids, input_ids)
    assert msg.input_ids.is_cpu, "input_ids should be on CPU"
    assert msg.sampling_params.max_tokens == 100


def test_tokenize_msg_fields():
    """Test TokenizeMsg has all required fields"""
    msg = create_tokenize_msg(uid=10, text="Test", max_tokens=20)

    assert hasattr(msg, 'uid')
    assert hasattr(msg, 'text')
    assert hasattr(msg, 'sampling_params')
    assert isinstance(msg.uid, int)
    assert isinstance(msg.text, str)
    # Check that sampling_params is a SamplingParams-like object (has required attributes)
    assert hasattr(msg.sampling_params, 'max_tokens')
    assert hasattr(msg.sampling_params, 'temperature')
    assert hasattr(msg.sampling_params, 'is_greedy')
    assert type(msg.sampling_params).__name__ == 'SamplingParams'


def test_user_msg_fields():
    """Test UserMsg has all required fields"""
    input_ids = torch.tensor([5, 6, 7], dtype=torch.int32)
    msg = create_user_msg(uid=20, input_ids=input_ids)

    assert hasattr(msg, 'uid')
    assert hasattr(msg, 'input_ids')
    assert hasattr(msg, 'sampling_params')
    assert isinstance(msg.uid, int)
    assert isinstance(msg.input_ids, torch.Tensor)
    # Check that sampling_params is a SamplingParams-like object (has required attributes)
    assert hasattr(msg.sampling_params, 'max_tokens')
    assert hasattr(msg.sampling_params, 'temperature')
    assert hasattr(msg.sampling_params, 'is_greedy')
    assert type(msg.sampling_params).__name__ == 'SamplingParams'


def test_user_msg_cpu_tensor():
    """Test that UserMsg ensures CPU tensor"""
    # Even if input is on GPU, should be moved to CPU
    input_ids = torch.tensor([1, 2], dtype=torch.int32)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    msg = create_user_msg(uid=1, input_ids=input_ids)
    assert msg.input_ids.is_cpu, "input_ids must be on CPU"


# ============================================================================
# Workflow and Data Flow Tests
# ============================================================================

def _simulate_tokenization(text: str) -> torch.Tensor:
    """
    Simulate tokenization by creating a simple tensor.
    In real system, this would use actual tokenizer.
    For testing, we just create a tensor with length based on text.
    """
    # Simple simulation: create tensor with length proportional to text length
    # In reality, tokenization would convert text to actual token IDs
    num_tokens = max(1, len(text) // 3)  # Rough approximation
    return torch.tensor(list(range(1, num_tokens + 1)), dtype=torch.int32)


def test_message_flow_tokenize_to_user():
    """
    Test complete message flow: TokenizeMsg → tokenization → UserMsg

    This simulates the workflow:
    1. API Server creates TokenizeMsg
    2. Tokenizer receives TokenizeMsg and tokenizes text
    3. Tokenizer creates UserMsg with tokenized input
    4. Verify all data is preserved through the flow
    """
    # Step 1: Create TokenizeMsg (simulating API Server)
    original_text = "Hello, how are you?"
    original_uid = 42
    original_max_tokens = 100
    original_temperature = 0.7
    original_top_k = 10

    tokenize_msg = create_tokenize_msg(
        uid=original_uid,
        text=original_text,
        max_tokens=original_max_tokens
    )

    # Verify TokenizeMsg structure
    assert tokenize_msg.uid == original_uid
    assert tokenize_msg.text == original_text
    assert tokenize_msg.sampling_params.max_tokens == original_max_tokens

    # Step 2: Simulate tokenization (simulating Tokenizer Worker)
    # In real system, tokenizer would convert text to token IDs
    input_ids = _simulate_tokenization(original_text)

    # Step 3: Create UserMsg (simulating Tokenizer creating UserMsg)
    user_msg = create_user_msg(
        uid=original_uid,  # Same UID as TokenizeMsg
        input_ids=input_ids,
        max_tokens=original_max_tokens
    )

    # Step 4: Verify data flow - UID preserved
    assert user_msg.uid == original_uid, "UID must be preserved through message flow"
    assert user_msg.uid == tokenize_msg.uid, "UID must match between TokenizeMsg and UserMsg"

    # Step 5: Verify data flow - SamplingParams preserved
    assert user_msg.sampling_params.max_tokens == original_max_tokens
    assert user_msg.sampling_params.max_tokens == tokenize_msg.sampling_params.max_tokens

    # Step 6: Verify transformation - text converted to tokens
    assert isinstance(user_msg.input_ids, torch.Tensor)
    assert user_msg.input_ids.is_cpu, "input_ids must be CPU tensor"
    assert len(user_msg.input_ids) > 0, "Tokenization should produce at least one token"


def test_sampling_params_preserved_through_flow():
    """
    Test that SamplingParams are preserved unchanged through message flow.

    This verifies that all sampling parameters flow correctly from
    TokenizeMsg to UserMsg without modification.
    """
    # Create TokenizeMsg with specific sampling parameters
    custom_params = {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "ignore_eos": True,
        "max_tokens": 256
    }

    # Create TokenizeMsg - note: create_tokenize_msg uses max_tokens parameter
    # We'll need to check if we can set other params directly
    tokenize_msg = create_tokenize_msg(
        uid=1,
        text="Test text",
        max_tokens=custom_params["max_tokens"]
    )

    # Store original params
    original_params = tokenize_msg.sampling_params

    # Simulate tokenization and create UserMsg
    input_ids = _simulate_tokenization("Test text")
    user_msg = create_user_msg(
        uid=1,
        input_ids=input_ids,
        max_tokens=custom_params["max_tokens"]
    )

    # Verify all SamplingParams fields are preserved
    assert user_msg.sampling_params.max_tokens == original_params.max_tokens
    assert user_msg.sampling_params.temperature == original_params.temperature
    assert user_msg.sampling_params.top_k == original_params.top_k
    assert user_msg.sampling_params.top_p == original_params.top_p
    assert user_msg.sampling_params.ignore_eos == original_params.ignore_eos

    # Verify is_greedy property is preserved
    assert user_msg.sampling_params.is_greedy == original_params.is_greedy


def test_multiple_messages_different_uids():
    """
    Test handling multiple messages with different UIDs.

    This verifies that the message system correctly tracks multiple
    concurrent requests using unique UIDs.
    """
    messages = []
    uids = [101, 202, 303, 404]
    texts = ["First request", "Second request", "Third request", "Fourth request"]

    # Create multiple TokenizeMsgs with different UIDs
    for uid, text in zip(uids, texts):
        tokenize_msg = create_tokenize_msg(uid=uid, text=text, max_tokens=50)
        messages.append((uid, text, tokenize_msg))

    # Verify all UIDs are distinct
    assert len(set(uids)) == len(uids), "All UIDs should be unique"

    # Convert each to UserMsg and verify UIDs preserved
    user_messages = []
    for uid, text, tokenize_msg in messages:
        input_ids = _simulate_tokenization(text)
        user_msg = create_user_msg(uid=uid, input_ids=input_ids, max_tokens=50)
        user_messages.append((uid, user_msg))

        # Verify UID matches
        assert user_msg.uid == uid, f"UID {uid} should be preserved"
        assert user_msg.uid == tokenize_msg.uid, "UID should match between messages"

    # Verify all UserMsg UIDs are still distinct
    user_uids = [uid for uid, _ in user_messages]
    assert len(set(user_uids)) == len(user_uids), "UIDs should remain distinct in UserMsgs"
    assert set(user_uids) == set(uids), "All original UIDs should be present"


def test_message_uid_tracking():
    """
    Test that UID is used to track requests through the message chain.

    This verifies that the same UID appears in both TokenizeMsg and UserMsg,
    enabling request tracking across process boundaries.
    """
    # Create a request with specific UID
    request_uid = 999
    text = "Track this request"

    # Step 1: API Server creates TokenizeMsg with UID
    tokenize_msg = create_tokenize_msg(uid=request_uid, text=text, max_tokens=75)
    assert tokenize_msg.uid == request_uid

    # Step 2: Tokenizer receives TokenizeMsg and creates UserMsg
    # The UID must be preserved to track the request
    input_ids = _simulate_tokenization(text)
    user_msg = create_user_msg(uid=request_uid, input_ids=input_ids, max_tokens=75)

    # Step 3: Verify UID tracking - same UID in both messages
    assert user_msg.uid == request_uid, "UserMsg must have same UID as TokenizeMsg"
    assert user_msg.uid == tokenize_msg.uid, "UID must be consistent across message chain"

    # Step 4: Verify UID can be used to correlate messages
    # In real system, scheduler would use UID to track request state
    assert user_msg.uid == tokenize_msg.uid == request_uid, "UID should be consistent"


def test_cpu_tensor_requirement_enforced():
    """
    Test that UserMsg creation enforces CPU tensor requirement.

    This verifies the behavior that GPU tensors are moved to CPU,
    which is required for ZMQ serialization across process boundaries.
    """
    # Test 1: CPU tensor should work directly
    cpu_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    assert cpu_tensor.is_cpu, "Test setup: tensor should be on CPU"

    user_msg = create_user_msg(uid=1, input_ids=cpu_tensor, max_tokens=50)
    assert user_msg.input_ids.is_cpu, "CPU tensor should remain on CPU"
    assert torch.equal(user_msg.input_ids, cpu_tensor), "Tensor values should be preserved"

    # Test 2: GPU tensor should be moved to CPU (if CUDA available)
    if torch.cuda.is_available():
        gpu_tensor = torch.tensor([10, 20, 30], dtype=torch.int32).cuda()
        assert not gpu_tensor.is_cpu, "Test setup: tensor should be on GPU"

        user_msg_gpu = create_user_msg(uid=2, input_ids=gpu_tensor, max_tokens=50)
        assert user_msg_gpu.input_ids.is_cpu, "GPU tensor must be moved to CPU"
        # Verify values are preserved after moving to CPU
        assert torch.equal(user_msg_gpu.input_ids.cpu(), gpu_tensor.cpu()), \
            "Tensor values should be preserved after moving to CPU"

    # Test 3: Verify behavior is consistent
    # Even if tensor starts on CPU, it should remain on CPU
    another_cpu_tensor = torch.tensor([100, 200], dtype=torch.int32)
    user_msg_2 = create_user_msg(uid=3, input_ids=another_cpu_tensor, max_tokens=50)
    assert user_msg_2.input_ids.is_cpu, "CPU tensor requirement must be enforced"


def test_trace_message_flow():
    """
    Test trace_message_flow function that traces complete message flow.
    
    This verifies that trace_message_flow correctly:
    1. Extracts data from TokenizeMsg
    2. Simulates tokenization
    3. Creates UserMsg
    4. Verifies data preservation
    5. Returns complete flow information
    """
    # Step 1: Create a TokenizeMsg
    original_uid = 123
    original_text = "Hello, world!"
    original_max_tokens = 200
    original_temperature = 0.8
    
    tokenize_msg = create_tokenize_msg(
        uid=original_uid,
        text=original_text,
        max_tokens=original_max_tokens
    )
    
    # Step 2: Call trace_message_flow
    flow_result = trace_message_flow(tokenize_msg)
    
    # Step 3: Verify return structure
    assert isinstance(flow_result, dict), "trace_message_flow should return a dictionary"
    
    # Verify all expected keys are present
    expected_keys = [
        "step1_tokenize_msg",
        "step2_tokenized",
        "step3_user_msg",
        "uid_preserved",
        "sampling_params_preserved"
    ]
    for key in expected_keys:
        assert key in flow_result, f"flow_result should contain '{key}'"
    
    # Step 4: Verify step1_tokenize_msg
    assert flow_result["step1_tokenize_msg"] == tokenize_msg
    assert flow_result["step1_tokenize_msg"].uid == original_uid
    assert flow_result["step1_tokenize_msg"].text == original_text
    
    # Step 5: Verify step2_tokenized (tokenized input)
    input_ids = flow_result["step2_tokenized"]
    assert isinstance(input_ids, torch.Tensor), "step2_tokenized should be a torch.Tensor"
    assert len(input_ids) > 0, "Tokenization should produce at least one token"
    
    # Step 6: Verify step3_user_msg
    user_msg = flow_result["step3_user_msg"]
    assert isinstance(user_msg, UserMsg), "step3_user_msg should be a UserMsg"
    assert user_msg.uid == original_uid, "UID should be preserved"
    assert torch.equal(user_msg.input_ids, input_ids), "input_ids should match tokenized result"
    assert user_msg.input_ids.is_cpu, "input_ids should be on CPU"
    
    # Step 7: Verify data preservation flags
    assert flow_result["uid_preserved"] == True, "UID should be preserved through flow"
    assert flow_result["sampling_params_preserved"] == True, "SamplingParams should be preserved through flow"
    
    # Step 8: Verify SamplingParams preservation in detail
    assert user_msg.sampling_params.max_tokens == original_max_tokens
    assert user_msg.sampling_params.max_tokens == tokenize_msg.sampling_params.max_tokens
    assert user_msg.sampling_params.temperature == tokenize_msg.sampling_params.temperature
