"""
Test for Puzzle 5.1: TableManager Basics

Run with: pytest learning/puzzles/05_scheduler/test_5.1.py -v
"""

import pytest
import sys
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle file with dot in name using importlib
puzzle_file = puzzle_dir / "puzzle_5.1.py"
spec = importlib.util.spec_from_file_location("puzzle_5_1", puzzle_file)
puzzle_5_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_5_1)
SimpleTableManager = puzzle_5_1.SimpleTableManager
create_table_manager = puzzle_5_1.create_table_manager


def test_initial_available_size():
    """Test that all slots are initially available"""
    manager = create_table_manager(10)
    assert manager.available_size == 10


def test_allocate_decreases_available():
    """Test that allocate decreases available slots"""
    manager = create_table_manager(5)
    manager.allocate()
    assert manager.available_size == 4


def test_allocate_returns_valid_slot():
    """Test that allocate returns a valid slot index"""
    manager = create_table_manager(5)
    slot = manager.allocate()
    assert 0 <= slot < 5


def test_allocate_returns_unique_slots():
    """Test that allocate returns unique slots"""
    manager = create_table_manager(5)
    slots = [manager.allocate() for _ in range(5)]
    assert len(set(slots)) == 5  # All unique


def test_free_increases_available():
    """Test that free increases available slots"""
    manager = create_table_manager(5)
    slot = manager.allocate()
    assert manager.available_size == 4
    manager.free(slot)
    assert manager.available_size == 5


def test_allocate_all_then_free_all():
    """Test allocating all slots then freeing them"""
    manager = create_table_manager(3)

    # Allocate all
    slots = [manager.allocate() for _ in range(3)]
    assert manager.available_size == 0

    # Free all
    for slot in slots:
        manager.free(slot)
    assert manager.available_size == 3


def test_allocate_when_empty_raises():
    """Test that allocating when no slots available raises error"""
    manager = create_table_manager(2)
    manager.allocate()
    manager.allocate()

    with pytest.raises(IndexError):
        manager.allocate()


def test_free_then_reallocate():
    """Test that freed slot can be reallocated"""
    manager = create_table_manager(2)
    slot1 = manager.allocate()
    slot2 = manager.allocate()

    manager.free(slot1)
    slot3 = manager.allocate()
    assert slot3 == slot1  # Should get the freed slot back


def test_available_size_with_zero_max():
    """Test manager with zero max requests"""
    manager = create_table_manager(0)
    assert manager.available_size == 0


def test_multiple_free_and_allocate():
    """Test multiple free and allocate operations"""
    manager = create_table_manager(5)

    # Allocate 3
    s1 = manager.allocate()
    s2 = manager.allocate()
    s3 = manager.allocate()
    assert manager.available_size == 2

    # Free middle one
    manager.free(s2)
    assert manager.available_size == 3

    # Allocate again
    s4 = manager.allocate()
    assert s4 == s2  # Should get s2 back
    assert manager.available_size == 2
