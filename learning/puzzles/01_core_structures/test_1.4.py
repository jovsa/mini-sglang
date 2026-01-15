"""
Test for Puzzle 1.4: Batch Creation Challenge

Run with: pytest learning/puzzles/01_core_structures/test_1.4.py -v
"""

import pytest
import sys
import torch
from pathlib import Path

puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

from puzzle_1_4 import create_prefill_batch, create_decode_batch, Batch
from puzzle_1_2 import create_req


def test_create_prefill_batch():
    """Test creating a prefill batch"""
    req1 = create_req(torch.tensor([1, 2, 3], dtype=torch.int32), output_len=5)
    req2 = create_req(torch.tensor([4, 5], dtype=torch.int32), output_len=3)

    batch = create_prefill_batch([req1, req2])

    assert batch is not None
    assert batch.phase == "prefill"
    assert batch.is_prefill == True
    assert batch.is_decode == False
    assert batch.size == 2
    assert len(batch.reqs) == 2


def test_create_decode_batch():
    """Test creating a decode batch"""
    req1 = create_req(torch.tensor([1, 2, 3], dtype=torch.int32), output_len=5)
    req2 = create_req(torch.tensor([4, 5], dtype=torch.int32), output_len=3)

    batch = create_decode_batch([req1, req2])

    assert batch is not None
    assert batch.phase == "decode"
    assert batch.is_prefill == False
    assert batch.is_decode == True
    assert batch.size == 2


def test_batch_size_property():
    """Test Batch.size property"""
    req1 = create_req(torch.tensor([1], dtype=torch.int32), output_len=5)
    req2 = create_req(torch.tensor([2], dtype=torch.int32), output_len=3)
    req3 = create_req(torch.tensor([3], dtype=torch.int32), output_len=10)

    batch = create_prefill_batch([req1, req2, req3])
    assert batch.size == 3


def test_batch_padded_size():
    """Test Batch.padded_size property"""
    req1 = create_req(torch.tensor([1], dtype=torch.int32), output_len=5)
    req2 = create_req(torch.tensor([2], dtype=torch.int32), output_len=3)

    batch = create_prefill_batch([req1, req2])
    # Initially, padded_size should equal size (no padding yet)
    assert batch.padded_size == batch.size


def test_batch_with_different_lengths():
    """Test batch with requests of different lengths"""
    req1 = create_req(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), output_len=5)
    req2 = create_req(torch.tensor([6, 7], dtype=torch.int32), output_len=3)
    req3 = create_req(torch.tensor([8], dtype=torch.int32), output_len=10)

    batch = create_prefill_batch([req1, req2, req3])
    assert batch.size == 3
    assert len(batch.reqs) == 3


def test_empty_batch():
    """Test creating batch with no requests (edge case)"""
    batch = create_prefill_batch([])
    assert batch.size == 0
    assert batch.phase == "prefill"


def test_single_request_batch():
    """Test batch with single request"""
    req = create_req(torch.tensor([1, 2, 3], dtype=torch.int32), output_len=5)
    batch = create_prefill_batch([req])
    assert batch.size == 1
    assert batch.reqs[0] == req
