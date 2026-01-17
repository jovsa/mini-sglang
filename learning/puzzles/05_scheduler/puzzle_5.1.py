"""
PUZZLE 5.1: TableManager Basics - [Easy]

TASK:
Implement a simplified TableManager that manages request slots.
The TableManager tracks which slots are free and allocates them to requests.

GIVEN:
- Understanding of resource allocation
- Fixed number of maximum concurrent requests

CHALLENGE:
- Implement available_size property
- Implement allocate() to get a free slot
- Implement free() to return a slot

HINT:
- Reference: `python/minisgl/scheduler/table.py:4-19` for TableManager
- Use a list to track free slots
- allocate() pops from the free list
- free() appends back to the free list

QUESTIONS:
1. Why do we need to limit max_running_reqs?
2. What happens if we try to allocate when no slots are free?
3. Why use a list instead of a set for free slots?

TEST:
Run: pytest learning/puzzles/05_scheduler/test_5.1.py -v
"""

from typing import List


class SimpleTableManager:
    """
    Manages request slots for the scheduler.

    Each request needs a slot (table index) to track its KV cache.
    This manager tracks which slots are available.
    """

    def __init__(self, max_running_reqs: int) -> None:
        """
        Initialize the table manager.

        Args:
            max_running_reqs: Maximum number of concurrent requests

        TODO: Initialize _free_slots as a list of slot indices [0, 1, 2, ..., max_running_reqs-1]
        """
        self._max_running_reqs = max_running_reqs
        # YOUR CODE HERE - initialize _free_slots
        self._free_slots: List[int] = []

    @property
    def available_size(self) -> int:
        """
        Return the number of available slots.

        Returns:
            Number of free slots that can be allocated
        """
        # YOUR CODE HERE
        pass

    def allocate(self) -> int:
        """
        Allocate a slot for a new request.

        Returns:
            The allocated slot index

        Raises:
            IndexError: If no slots are available

        HINT: Use list.pop() to get and remove the last element
        """
        # YOUR CODE HERE
        pass

    def free(self, slot: int) -> None:
        """
        Free a slot when a request completes.

        Args:
            slot: The slot index to free

        HINT: Use list.append() to add the slot back
        """
        # YOUR CODE HERE
        pass


def create_table_manager(max_reqs: int) -> SimpleTableManager:
    """
    Create a SimpleTableManager.

    Args:
        max_reqs: Maximum number of concurrent requests

    Returns:
        SimpleTableManager object
    """
    return SimpleTableManager(max_reqs)


# Test your implementation
if __name__ == "__main__":
    # Create a manager with 5 slots
    manager = create_table_manager(5)
    print(f"Initial available: {manager.available_size} (expected 5)")

    # Allocate some slots
    slot1 = manager.allocate()
    slot2 = manager.allocate()
    print(f"Allocated slots: {slot1}, {slot2}")
    print(f"Available after allocate: {manager.available_size} (expected 3)")

    # Free a slot
    manager.free(slot1)
    print(f"Available after free: {manager.available_size} (expected 4)")
