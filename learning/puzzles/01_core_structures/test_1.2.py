"""
Test for Puzzle 1.2: Fix the Req Creation

Run with: pytest learning/puzzles/01_core_structures/test_1.2.py -v
"""

import pytest
import sys
import torch
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle files with dots in names using importlib
puzzle_1_2_file = puzzle_dir / "puzzle_1.2.py"
spec_1_2 = importlib.util.spec_from_file_location("puzzle_1_2", puzzle_1_2_file)
puzzle_1_2 = importlib.util.module_from_spec(spec_1_2)
spec_1_2.loader.exec_module(puzzle_1_2)
create_req = puzzle_1_2.create_req
Req = puzzle_1_2.Req
MockCacheHandle = puzzle_1_2.MockCacheHandle

puzzle_1_1_file = puzzle_dir / "puzzle_1.1.py"
spec_1_1 = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec_1_1)
spec_1_1.loader.exec_module(puzzle_1_1)
create_sampling_params = puzzle_1_1.create_sampling_params


def test_create_req_basic():
    """Test basic request creation"""
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    req = create_req(input_ids, output_len=10, uid=1)

    assert req is not None
    assert req.device_len == 5, "device_len should equal input length"
    assert req.max_device_len == 15, "max_device_len should be input_len + output_len"
    assert req.cached_len == 0
    assert req.output_len == 10
    assert req.uid == 1


def test_req_properties():
    """Test Req properties"""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    req = create_req(input_ids, output_len=5, cached_len=1)

    assert req.device_len == 3
    assert req.max_device_len == 8  # 3 + 5
    assert req.remain_len == 5  # 8 - 3
    assert req.extend_len == 2  # 3 - 1
    assert req.can_decode() == True


def test_req_validation_cached_len():
    """Test that cached_len < device_len is enforced"""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)

    # This should work
    req = create_req(input_ids, output_len=5, cached_len=2)
    assert req.cached_len < req.device_len

    # This should raise an assertion error
    with pytest.raises(AssertionError):
        create_req(input_ids, output_len=5, cached_len=3)  # cached_len >= device_len


def test_req_validation_cpu_tensor():
    """Test that input_ids must be on CPU"""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)

    # CPU tensor should work
    req = create_req(input_ids, output_len=5)
    assert req.input_ids.is_cpu

    # If input_ids is on GPU, it should be moved to CPU
    if torch.cuda.is_available():
        gpu_input = input_ids.cuda()
        req2 = create_req(gpu_input, output_len=5)
        assert req2.input_ids.is_cpu, "input_ids should be moved to CPU"


def test_req_can_decode():
    """Test can_decode property"""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)

    # Request with remaining tokens
    req1 = create_req(input_ids, output_len=5)
    assert req1.can_decode() == True

    # Request at max length
    req2 = create_req(input_ids, output_len=0)
    assert req2.can_decode() == False  # No output tokens remaining


def test_req_edge_cases():
    """Test edge cases"""
    # Empty input (edge case)
    input_ids = torch.tensor([], dtype=torch.int32)
    req = create_req(input_ids, output_len=10, cached_len=0)
    assert req.device_len == 0
    assert req.max_device_len == 10

    # Single token input
    input_ids = torch.tensor([1], dtype=torch.int32)
    req = create_req(input_ids, output_len=100)
    assert req.device_len == 1
    assert req.max_device_len == 101


def test_req_validation_max_device_len():
    """Test that device_len <= max_device_len"""
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)

    # Normal case
    req = create_req(input_ids, output_len=10)
    assert req.device_len <= req.max_device_len

    # Edge case: output_len = 0
    req2 = create_req(input_ids, output_len=0)
    assert req2.device_len == req2.max_device_len  # Should be equal
