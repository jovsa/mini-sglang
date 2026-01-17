"""
PUZZLE 6.1: Context Management - [Easy]

TASK:
Implement a simplified Context class for managing global state during inference.
The Context provides access to the current batch and configuration.

GIVEN:
- Understanding of context managers in Python
- Global state pattern for accessing configuration

CHALLENGE:
- Implement the batch property that asserts _batch is not None
- Implement forward_batch context manager
- Implement set_global_ctx and get_global_ctx functions

HINT:
- Reference: `python/minisgl/core.py:97-129` for Context
- Use @contextmanager decorator for forward_batch
- Global context should only be set once (assert it's None before setting)
- The batch property should raise AssertionError if no batch is active

QUESTIONS:
1. Why use a context manager for forward_batch?
2. Why prevent nested forward_batch calls?
3. What's the benefit of a global context vs passing context everywhere?

TEST:
Run: pytest learning/puzzles/06_context/test_6.1.py -v
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MockBatch:
    """Mock batch for testing."""
    size: int
    phase: str


@dataclass
class SimpleContext:
    """
    Context for managing global state during inference.

    Provides access to the current batch being processed.
    """
    page_size: int
    _batch: Optional[MockBatch] = field(default=None, init=False)

    @property
    def batch(self) -> MockBatch:
        """
        Get the current active batch.

        Returns:
            The current batch

        Raises:
            AssertionError: If no batch is currently active

        HINT: Use assert to check _batch is not None
        """
        # YOUR CODE HERE
        pass

    @contextmanager
    def forward_batch(self, batch: MockBatch):
        """
        Context manager for processing a batch.

        Sets _batch for the duration of the context, then clears it.
        Prevents nested forward_batch calls.

        Args:
            batch: The batch to process

        Usage:
            with ctx.forward_batch(batch):
                # batch is available via ctx.batch
                process(ctx.batch)
            # batch is no longer available

        HINT:
        1. Assert _batch is None (no nested calls)
        2. Set _batch = batch
        3. yield
        4. Set _batch = None in finally block
        """
        # YOUR CODE HERE
        pass


# Global context
_GLOBAL_CTX: Optional[SimpleContext] = None


def set_global_ctx(ctx: SimpleContext) -> None:
    """
    Set the global context.

    Can only be called once (when _GLOBAL_CTX is None).

    Args:
        ctx: The context to set as global

    Raises:
        AssertionError: If global context is already set
    """
    global _GLOBAL_CTX
    # YOUR CODE HERE
    pass


def get_global_ctx() -> SimpleContext:
    """
    Get the global context.

    Returns:
        The global context

    Raises:
        AssertionError: If global context is not set
    """
    # YOUR CODE HERE
    pass


def reset_global_ctx() -> None:
    """Reset the global context (for testing)."""
    global _GLOBAL_CTX
    _GLOBAL_CTX = None


# Test your implementation
if __name__ == "__main__":
    reset_global_ctx()

    # Create and set context
    ctx = SimpleContext(page_size=16)
    set_global_ctx(ctx)
    print(f"Global context set with page_size={get_global_ctx().page_size}")

    # Test forward_batch
    batch = MockBatch(size=4, phase="prefill")
    with ctx.forward_batch(batch):
        print(f"Inside forward_batch: batch.size={ctx.batch.size}")

    # Batch should be None outside context
    try:
        _ = ctx.batch
        print("ERROR: Should have raised AssertionError")
    except AssertionError:
        print("Correctly raised AssertionError when no batch active")

    reset_global_ctx()
