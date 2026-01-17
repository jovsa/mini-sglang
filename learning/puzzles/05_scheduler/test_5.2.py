"""
Test for Puzzle 5.2: DecodeManager

Run with: pytest learning/puzzles/05_scheduler/test_5.2.py -v
"""

import pytest
import sys
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle file with dot in name using importlib
puzzle_file = puzzle_dir / "puzzle_5.2.py"
spec = importlib.util.spec_from_file_location("puzzle_5_2", puzzle_file)
puzzle_5_2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_5_2)
SimpleDecodeManager = puzzle_5_2.SimpleDecodeManager
create_decode_manager = puzzle_5_2.create_decode_manager
create_mock_req = puzzle_5_2.create_mock_req


def test_initial_state():
    """Test initial state of decode manager"""
    manager = create_decode_manager()
    assert manager.runnable is False
    assert manager.inflight_tokens == 0


def test_add_single_req():
    """Test adding a single request"""
    manager = create_decode_manager()
    req = create_mock_req(uid=1, remain_len=10)
    manager.add_reqs([req])

    assert manager.runnable is True
    assert manager.inflight_tokens == 10


def test_add_multiple_reqs():
    """Test adding multiple requests"""
    manager = create_decode_manager()
    req1 = create_mock_req(uid=1, remain_len=10)
    req2 = create_mock_req(uid=2, remain_len=5)
    manager.add_reqs([req1, req2])

    assert manager.inflight_tokens == 15


def test_add_req_with_zero_remain():
    """Test that requests with remain_len=0 are not added"""
    manager = create_decode_manager()
    req = create_mock_req(uid=1, remain_len=0)
    manager.add_reqs([req])

    assert manager.runnable is False
    assert manager.inflight_tokens == 0


def test_add_mixed_reqs():
    """Test adding mix of valid and invalid requests"""
    manager = create_decode_manager()
    req1 = create_mock_req(uid=1, remain_len=10)
    req2 = create_mock_req(uid=2, remain_len=0)  # Won't be added
    req3 = create_mock_req(uid=3, remain_len=5)
    manager.add_reqs([req1, req2, req3])

    assert manager.inflight_tokens == 15  # Only req1 and req3


def test_remove_req():
    """Test removing a request"""
    manager = create_decode_manager()
    req = create_mock_req(uid=1, remain_len=10)
    manager.add_reqs([req])
    manager.remove_req(req)

    assert manager.runnable is False
    assert manager.inflight_tokens == 0


def test_remove_nonexistent_req():
    """Test removing a request that doesn't exist (should not error)"""
    manager = create_decode_manager()
    req = create_mock_req(uid=1, remain_len=10)
    # Should not raise an error
    manager.remove_req(req)
    assert manager.runnable is False


def test_runnable_becomes_false():
    """Test runnable becomes False when all requests removed"""
    manager = create_decode_manager()
    req1 = create_mock_req(uid=1, remain_len=10)
    req2 = create_mock_req(uid=2, remain_len=5)

    manager.add_reqs([req1, req2])
    assert manager.runnable is True

    manager.remove_req(req1)
    assert manager.runnable is True  # Still has req2

    manager.remove_req(req2)
    assert manager.runnable is False  # No more requests


def test_add_empty_list():
    """Test adding empty list of requests"""
    manager = create_decode_manager()
    manager.add_reqs([])
    assert manager.runnable is False


def test_inflight_tokens_updates():
    """Test inflight_tokens updates correctly"""
    manager = create_decode_manager()

    req1 = create_mock_req(uid=1, remain_len=10)
    manager.add_reqs([req1])
    assert manager.inflight_tokens == 10

    req2 = create_mock_req(uid=2, remain_len=20)
    manager.add_reqs([req2])
    assert manager.inflight_tokens == 30

    manager.remove_req(req1)
    assert manager.inflight_tokens == 20


def test_add_same_req_twice():
    """Test adding same request twice (set behavior)"""
    manager = create_decode_manager()
    req = create_mock_req(uid=1, remain_len=10)

    manager.add_reqs([req])
    manager.add_reqs([req])

    # Should only count once (set behavior)
    assert manager.inflight_tokens == 10
