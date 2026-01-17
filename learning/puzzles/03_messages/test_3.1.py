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
    assert isinstance(msg.sampling_params, SamplingParams)


def test_user_msg_fields():
    """Test UserMsg has all required fields"""
    input_ids = torch.tensor([5, 6, 7], dtype=torch.int32)
    msg = create_user_msg(uid=20, input_ids=input_ids)

    assert hasattr(msg, 'uid')
    assert hasattr(msg, 'input_ids')
    assert hasattr(msg, 'sampling_params')
    assert isinstance(msg.uid, int)
    assert isinstance(msg.input_ids, torch.Tensor)
    assert isinstance(msg.sampling_params, SamplingParams)


def test_user_msg_cpu_tensor():
    """Test that UserMsg ensures CPU tensor"""
    # Even if input is on GPU, should be moved to CPU
    input_ids = torch.tensor([1, 2], dtype=torch.int32)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    msg = create_user_msg(uid=1, input_ids=input_ids)
    assert msg.input_ids.is_cpu, "input_ids must be on CPU"
