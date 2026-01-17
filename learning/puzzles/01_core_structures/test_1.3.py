"""
Test for Puzzle 1.3: Implement Request State Transitions

Run with: pytest learning/puzzles/01_core_structures/test_1.3.py -v
"""

import pytest
import sys
import torch
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle file with dot in name using importlib
puzzle_file = puzzle_dir / "puzzle_1.3.py"
spec = importlib.util.spec_from_file_location("puzzle_1_3", puzzle_file)
puzzle_1_3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_1_3)
create_extended_req = puzzle_1_3.create_extended_req
ExtendedReq = puzzle_1_3.ExtendedReq


def test_complete_one_basic():
    """Test complete_one() basic functionality"""
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    req = create_extended_req(input_ids, output_len=10, cached_len=0)

    initial_device_len = req.device_len
    initial_cached_len = req.cached_len

    req.complete_one()

    # cached_len should be updated to previous device_len
    assert req.cached_len == initial_device_len
    # device_len should be incremented by 1
    assert req.device_len == initial_device_len + 1


def test_complete_one_multiple_times():
    """Test calling complete_one() multiple times"""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    req = create_extended_req(input_ids, output_len=5, cached_len=0)

    # Complete first token
    req.complete_one()
    assert req.cached_len == 3  # Was device_len
    assert req.device_len == 4  # Incremented

    # Complete second token
    req.complete_one()
    assert req.cached_len == 4  # Was device_len
    assert req.device_len == 5  # Incremented


def test_append_host_basic():
    """Test append_host() basic functionality"""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    req = create_extended_req(input_ids, output_len=5, cached_len=3)

    initial_len = len(req.input_ids)

    new_token = torch.tensor([99], dtype=torch.int32)
    req.append_host(new_token)

    # input_ids should have new token appended
    assert len(req.input_ids) == initial_len + 1
    assert req.input_ids[-1].item() == 99
    # Note: append_host does NOT update device_len - that's done by complete_one()


def test_append_host_multiple_times():
    """Test calling append_host() multiple times"""
    input_ids = torch.tensor([1, 2], dtype=torch.int32)
    req = create_extended_req(input_ids, output_len=5, cached_len=2)

    req.append_host(torch.tensor([10], dtype=torch.int32))
    req.append_host(torch.tensor([20], dtype=torch.int32))

    assert len(req.input_ids) == 4
    assert req.input_ids.tolist() == [1, 2, 10, 20]
    # Note: append_host does NOT update device_len


def test_complete_one_then_append_host():
    """Test the sequence: complete_one then append_host (decode phase)"""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    req = create_extended_req(input_ids, output_len=5, cached_len=3)

    # During decode: complete current position, then append new token
    req.complete_one()  # Mark position 3 as cached, move to 4
    assert req.cached_len == 3
    assert req.device_len == 4

    req.append_host(torch.tensor([99], dtype=torch.int32))  # Add generated token to host
    assert len(req.input_ids) == 4
    # Note: device_len stays at 4 (append_host only updates input_ids, not device_len)


def test_state_transitions():
    """Test complete request lifecycle state transitions"""
    input_ids = torch.tensor([1, 2], dtype=torch.int32)
    req = create_extended_req(input_ids, output_len=3, cached_len=0)

    # Initial state: prefill phase
    assert req.cached_len == 0
    assert req.device_len == 2
    assert req.remain_len == 3

    # Process first token in prefill
    req.complete_one()
    assert req.cached_len == 2
    assert req.device_len == 3

    # Process second token in prefill
    req.complete_one()
    assert req.cached_len == 3
    assert req.device_len == 4

    # Now in decode phase: generate and append tokens
    req.append_host(torch.tensor([10], dtype=torch.int32))
    assert req.device_len == 4
    assert len(req.input_ids) == 3

    req.complete_one()
    assert req.cached_len == 4
    assert req.device_len == 5
