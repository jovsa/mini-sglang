"""
PUZZLE 4.2: RadixTree Node Operations - [Medium]

TASK:
Implement a simplified RadixTreeNode for KV cache prefix matching.
The radix tree enables efficient prefix sharing between requests.

GIVEN:
- Understanding of tree data structures
- How prefix caching works in LLM inference

CHALLENGE:
- Implement is_root() - check if node has no parent
- Implement is_leaf() - check if node has no children
- Implement add_child() - add a child node
- Track ref_count for cache eviction decisions

HINT:
- Reference: `python/minisgl/kvcache/radix_manager.py:13-79` for RadixTreeNode
- A node is root if _parent is None
- A node is leaf if children dict is empty
- ref_count tracks how many requests are using this node

QUESTIONS:
1. Why is prefix matching important for LLM inference?
2. What does ref_count == 0 mean for a node?
3. When would a node be split in a radix tree?

TEST:
Run: pytest learning/puzzles/04_kvcache/test_4.2.py -v
"""

from typing import Dict, Optional


class SimpleRadixNode:
    """
    A simplified radix tree node for prefix caching.

    Attributes:
        children: Dict mapping token IDs to child nodes
        _parent: Reference to parent node (None for root)
        ref_count: Number of active references (0 = can be evicted)
        node_id: Unique identifier for this node
    """
    _counter: int = 0

    def __init__(self) -> None:
        self.children: Dict[int, "SimpleRadixNode"] = {}
        self._parent: Optional["SimpleRadixNode"] = None
        self.ref_count: int = 0
        self.node_id = SimpleRadixNode._counter
        SimpleRadixNode._counter += 1

    def is_root(self) -> bool:
        """
        Check if this node is the root of the tree.

        Returns:
            True if this node has no parent, False otherwise

        HINT: Check if _parent is None
        """
        # YOUR CODE HERE
        pass

    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf (has no children).

        Returns:
            True if this node has no children, False otherwise

        HINT: Check if children dict is empty
        """
        # YOUR CODE HERE
        pass

    def add_child(self, token_id: int, child: "SimpleRadixNode") -> None:
        """
        Add a child node for the given token ID.

        Args:
            token_id: The token ID that leads to this child
            child: The child node to add

        Should:
        1. Set child's _parent to self
        2. Add child to self.children dict with token_id as key

        HINT: Look at set_parent in radix_manager.py:35-37
        """
        # YOUR CODE HERE
        pass

    def get_child(self, token_id: int) -> Optional["SimpleRadixNode"]:
        """
        Get the child node for a given token ID.

        Args:
            token_id: The token ID to look up

        Returns:
            The child node, or None if not found
        """
        # YOUR CODE HERE
        pass


def create_radix_tree() -> SimpleRadixNode:
    """
    Create a new radix tree with just a root node.

    The root node should have ref_count = 1 (always protected).

    Returns:
        The root node of the tree
    """
    # YOUR CODE HERE
    pass


def add_prefix_to_tree(root: SimpleRadixNode, token_ids: list) -> SimpleRadixNode:
    """
    Add a sequence of tokens as a path in the tree.

    Starting from root, create child nodes for each token in the sequence.
    Return the final (deepest) node created.

    Args:
        root: The root of the radix tree
        token_ids: List of token IDs to add as a path

    Returns:
        The final node in the path

    Example:
        add_prefix_to_tree(root, [1, 2, 3]) creates:
        root -> node(1) -> node(2) -> node(3)
        Returns the node for token 3
    """
    # YOUR CODE HERE
    pass


# Test your implementation
if __name__ == "__main__":
    # Create a tree
    root = create_radix_tree()
    print(f"Root: is_root={root.is_root()}, is_leaf={root.is_leaf()}, ref_count={root.ref_count}")

    # Add a prefix path
    leaf = add_prefix_to_tree(root, [1, 2, 3])
    print(f"After adding [1,2,3]:")
    print(f"  Root: is_leaf={root.is_leaf()} (expected False)")
    print(f"  Leaf: is_leaf={leaf.is_leaf()} (expected True)")
    print(f"  Leaf: is_root={leaf.is_root()} (expected False)")

    # Check the path
    node = root.get_child(1)
    print(f"  Node at token 1 exists: {node is not None}")
