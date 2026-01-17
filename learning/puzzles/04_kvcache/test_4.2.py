"""
Test for Puzzle 4.2: RadixTree Node Operations

Run with: pytest learning/puzzles/04_kvcache/test_4.2.py -v
"""

import pytest
import sys
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle file with dot in name using importlib
puzzle_file = puzzle_dir / "puzzle_4.2.py"
spec = importlib.util.spec_from_file_location("puzzle_4_2", puzzle_file)
puzzle_4_2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_4_2)
SimpleRadixNode = puzzle_4_2.SimpleRadixNode
create_radix_tree = puzzle_4_2.create_radix_tree
add_prefix_to_tree = puzzle_4_2.add_prefix_to_tree


def test_create_radix_tree():
    """Test that create_radix_tree returns a proper root node"""
    root = create_radix_tree()
    assert root is not None
    assert root.is_root() is True
    assert root.ref_count == 1  # Root is always protected


def test_root_is_initially_leaf():
    """Test that a new root is also a leaf (no children)"""
    root = create_radix_tree()
    assert root.is_leaf() is True


def test_node_is_root():
    """Test is_root() method"""
    root = create_radix_tree()
    child = SimpleRadixNode()
    root.add_child(1, child)

    assert root.is_root() is True
    assert child.is_root() is False


def test_node_is_leaf():
    """Test is_leaf() method"""
    root = create_radix_tree()
    child = SimpleRadixNode()
    root.add_child(1, child)

    assert root.is_leaf() is False  # Has child
    assert child.is_leaf() is True  # No children


def test_add_child():
    """Test add_child() method"""
    root = create_radix_tree()
    child = SimpleRadixNode()
    root.add_child(42, child)

    assert 42 in root.children
    assert root.children[42] is child
    assert child._parent is root


def test_get_child_exists():
    """Test get_child() when child exists"""
    root = create_radix_tree()
    child = SimpleRadixNode()
    root.add_child(10, child)

    found = root.get_child(10)
    assert found is child


def test_get_child_not_exists():
    """Test get_child() when child doesn't exist"""
    root = create_radix_tree()
    found = root.get_child(999)
    assert found is None


def test_add_prefix_to_tree_single_token():
    """Test adding a single token prefix"""
    root = create_radix_tree()
    leaf = add_prefix_to_tree(root, [5])

    assert root.is_leaf() is False
    assert leaf.is_leaf() is True
    assert root.get_child(5) is leaf


def test_add_prefix_to_tree_multiple_tokens():
    """Test adding a multi-token prefix"""
    root = create_radix_tree()
    leaf = add_prefix_to_tree(root, [1, 2, 3])

    # Check path exists
    node1 = root.get_child(1)
    assert node1 is not None
    assert node1.is_leaf() is False

    node2 = node1.get_child(2)
    assert node2 is not None
    assert node2.is_leaf() is False

    node3 = node2.get_child(3)
    assert node3 is not None
    assert node3.is_leaf() is True
    assert node3 is leaf


def test_add_prefix_empty():
    """Test adding empty prefix returns root"""
    root = create_radix_tree()
    result = add_prefix_to_tree(root, [])
    assert result is root


def test_multiple_children():
    """Test adding multiple children to same node"""
    root = create_radix_tree()
    child1 = SimpleRadixNode()
    child2 = SimpleRadixNode()

    root.add_child(1, child1)
    root.add_child(2, child2)

    assert root.get_child(1) is child1
    assert root.get_child(2) is child2
    assert len(root.children) == 2


def test_unique_node_ids():
    """Test that each node gets a unique ID"""
    nodes = [SimpleRadixNode() for _ in range(5)]
    ids = [n.node_id for n in nodes]
    assert len(set(ids)) == 5  # All unique
