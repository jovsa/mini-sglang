"""
PUZZLE 4.2: RadixTree Node Operations - [Medium]

TASK:
Implement a simplified RadixTreeNode for KV cache prefix matching.
The radix tree enables efficient prefix sharing between requests.

GIVEN:
- Understanding of tree data structures
- How prefix caching works in LLM inference

================================================================================
DATA FLOW: Prefix Matching Through Radix Tree
================================================================================

The radix tree enables multiple requests to share common prefixes:

```mermaid
flowchart TD
    A[UserMsg arrives] -->|"input_ids: [1,2,3,4,5]"| B[CacheManager.match_req]
    B -->|"Traverse radix tree"| C[RadixTree]
    C -->|"Match prefix [1,2,3]"| D[Found node at depth 3]
    D -->|"cached_len=3"| E[Create CacheHandle]
    E -->|"cached_len=3"| F[Create Req]
    F -->|"Only process [4,5]"| G[Save computation]
    G -->|"New request with [1,2,3,6,7]"| H[Match same prefix]
    H -->|"Share cache for [1,2,3]"| I[Reuse KV cache]
```

Step-by-step flow:
1. **CacheManager.match_req()** (`python/minisgl/kvcache/radix_manager.py:82-120`):
   - Takes `input_ids` from UserMsg
   - Traverses radix tree starting from root
   - Matches longest common prefix
   - Returns `CacheHandle` with `cached_len` = length of matched prefix

2. **Radix Tree Traversal**:
   - Start at root node
   - For each token in `input_ids`, follow child node
   - Stop when no matching child found
   - Return deepest matching node

3. **Prefix Sharing**:
   - Multiple requests with same prefix share the same cache node
   - `ref_count` tracks how many requests use this node
   - Node is protected (can't evict) as long as `ref_count > 0`

4. **Cache Efficiency**:
   - If prefix matches, only new tokens need processing
   - KV cache for prefix is reused across requests
   - Saves computation and memory

**Example**:
- Request 1: `[1, 2, 3, 4, 5]` → Matches prefix `[1, 2, 3]` (cached_len=3)
- Request 2: `[1, 2, 3, 6, 7]` → Matches same prefix `[1, 2, 3]` (cached_len=3)
- Both requests share KV cache for tokens `[1, 2, 3]`
- Only process new tokens: Request 1 processes `[4, 5]`, Request 2 processes `[6, 7]`

================================================================================
COMPONENT CONNECTIONS: Radix Tree to Cache Memory
================================================================================

```mermaid
graph TB
    subgraph "Request Layer"
        Req1[Req 1: 1,2,3,4,5]
        Req2[Req 2: 1,2,3,6,7]
    end
    
    subgraph "Radix Tree"
        Root[Root Node]
        N1[Node: token=1]
        N2[Node: token=2]
        N3[Node: token=3]
        N4[Node: token=4]
        N6[Node: token=6]
    end
    
    subgraph "Cache Memory"
        Cache1[Cache for 1,2,3]
        Cache2[Cache for 4]
        Cache3[Cache for 6]
    end
    
    Req1 -->|"Match prefix"| Root
    Req2 -->|"Match prefix"| Root
    Root -->|"child[1]"| N1
    N1 -->|"child[2]"| N2
    N2 -->|"child[3]"| N3
    N3 -->|"Points to"| Cache1
    N3 -->|"child[4]"| N4
    N3 -->|"child[6]"| N6
    N4 -->|"Points to"| Cache2
    N6 -->|"Points to"| Cache3
    Cache1 -->|"Shared by"| Req1
    Cache1 -->|"Shared by"| Req2
```

Key Connections:
- **Tree Structure**: Each node represents a token position in the sequence
- **Shared Prefix**: Multiple requests share nodes for common prefixes
- **Cache Mapping**: Each node maps to physical cache memory pages
- **Reference Counting**: `ref_count` tracks how many requests use each node

================================================================================
TECHNICAL DECISIONS: Why Radix Tree vs Simple Prefix Matching
================================================================================

**Decision**: Use radix tree for prefix matching instead of simple prefix comparison.

**Why?**
1. **Efficiency**:
   - **Simple matching**: O(n*m) - compare each request against all cached prefixes
   - **Radix tree**: O(n) - single tree traversal to find longest match
   - Much faster for many requests with overlapping prefixes

2. **Memory Efficiency**:
   - Radix tree compresses common prefixes
   - Multiple requests share the same tree nodes
   - Reduces memory overhead compared to storing all prefixes separately

3. **Scalability**:
   - Tree structure scales well with number of unique prefixes
   - Adding new prefixes is O(depth) - typically much less than total cache size
   - Efficient for systems with many concurrent requests

**Alternative Considered**: Simple prefix matching (compare against all cached sequences).
   - **Problem**: O(n*m) complexity - slow with many requests
   - **Problem**: No structure for efficient lookup
   - **Chosen**: Radix tree for better performance and scalability

**Why Prefix Matching is Important**:
1. **Common Scenarios**: Many requests share common prefixes
   - System prompts (same for all requests)
   - Few-shot examples (repeated across requests)
   - Conversation history (shared context)

2. **Performance**: Reusing cached KV saves computation
   - Attention over cached tokens is already computed
   - Only need to compute attention for new tokens
   - Significant speedup for long shared prefixes

3. **Memory**: Sharing cache reduces memory usage
   - One cache entry serves multiple requests
   - Enables more concurrent requests with same memory

**ref_count Mechanism**:
- Each node tracks `ref_count` (number of requests using it)
- When request starts: Increment ref_count for all nodes in matched path
- When request completes: Decrement ref_count for all nodes in path
- When ref_count reaches 0: Node becomes evictable (can be freed)

**When Node is Split**:
- Radix tree nodes can be split when requests diverge
- Example: `[1,2,3,4]` and `[1,2,3,5]` share prefix `[1,2,3]`
- Node at position 3 has two children: one for token 4, one for token 5
- This enables fine-grained prefix sharing

================================================================================
CHALLENGE
================================================================================

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
4. **NEW**: Trace through the code: How does `CacheManager.match_req()` traverse the radix tree?
   (Hint: Read `python/minisgl/kvcache/radix_manager.py:82-120`)
5. **NEW**: What happens when two requests have different prefixes? How does the tree handle this?
   (Hint: Tree branches at the first differing token)

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
