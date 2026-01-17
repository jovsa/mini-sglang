"""
Test for Puzzle 4.1: Cache Handle Basics

Run with: pytest learning/puzzles/04_kvcache/test_4.1.py -v
"""

import pytest
import sys
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle file with dot in name using importlib
puzzle_file = puzzle_dir / "puzzle_4.1.py"
spec = importlib.util.spec_from_file_location("puzzle_4_1", puzzle_file)
puzzle_4_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_4_1)
SizeInfo = puzzle_4_1.SizeInfo
SimpleCacheHandle = puzzle_4_1.SimpleCacheHandle
create_size_info = puzzle_4_1.create_size_info
create_cache_handle = puzzle_4_1.create_cache_handle


def test_size_info_creation():
    """Test SizeInfo creation"""
    size_info = create_size_info(evictable=100, protected=50)
    assert size_info.evictable_size == 100
    assert size_info.protected_size == 50


def test_size_info_total_size():
    """Test SizeInfo total_size property"""
    size_info = create_size_info(evictable=100, protected=50)
    assert size_info.total_size == 150


def test_size_info_zero_values():
    """Test SizeInfo with zero values"""
    size_info = create_size_info(evictable=0, protected=0)
    assert size_info.total_size == 0


def test_size_info_only_evictable():
    """Test SizeInfo with only evictable entries"""
    size_info = create_size_info(evictable=200, protected=0)
    assert size_info.evictable_size == 200
    assert size_info.protected_size == 0
    assert size_info.total_size == 200


def test_size_info_only_protected():
    """Test SizeInfo with only protected entries"""
    size_info = create_size_info(evictable=0, protected=75)
    assert size_info.evictable_size == 0
    assert size_info.protected_size == 75
    assert size_info.total_size == 75


def test_cache_handle_creation():
    """Test SimpleCacheHandle creation"""
    handle = create_cache_handle(cached_len=10)
    assert handle.cached_len == 10


def test_cache_handle_zero_cached():
    """Test SimpleCacheHandle with zero cached tokens"""
    handle = create_cache_handle(cached_len=0)
    assert handle.cached_len == 0


def test_cache_handle_large_cached():
    """Test SimpleCacheHandle with large cached length"""
    handle = create_cache_handle(cached_len=1024)
    assert handle.cached_len == 1024


def test_size_info_is_namedtuple():
    """Test that SizeInfo behaves like a NamedTuple"""
    size_info = create_size_info(evictable=10, protected=5)
    # Should be iterable
    evictable, protected = size_info
    assert evictable == 10
    assert protected == 5


def test_cache_handle_is_frozen():
    """Test that SimpleCacheHandle is immutable (frozen dataclass)"""
    handle = create_cache_handle(cached_len=10)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        handle.cached_len = 20
