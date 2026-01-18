"""
PUZZLE 6.1: Context Management - [Easy]

TASK:
Implement a simplified Context class for managing global state during inference.
The Context provides access to the current batch and configuration.

GIVEN:
- Understanding of context managers in Python
- Global state pattern for accessing configuration

================================================================================
DATA FLOW: Context Through Engine to Layers
================================================================================

Context flows from Engine through the deep call stack to all layers:

```mermaid
flowchart TD
    A[Engine.forward_batch] -->|"ctx.forward_batch(batch)"| B[Context Manager]
    B -->|"Sets _batch"| C[Global Context Active]
    C -->|"Model.forward"| D[TransformerLayer]
    D -->|"get_global_ctx()"| E[Access ctx.batch]
    E -->|"Attention.forward"| F[get_global_ctx()]
    F -->|"Access ctx.batch"| G[Use batch info]
    G -->|"Complete forward"| H[Context Manager Exit]
    H -->|"Clears _batch"| I[Context Inactive]
```

Step-by-step flow:
1. **Engine.forward_batch()** (`python/minisgl/engine/engine.py`):
   - Creates Batch object
   - Enters context: `with ctx.forward_batch(batch):`
   - Sets `_batch = batch` in Context

2. **Model Forward Pass**:
   - Calls `model.forward(input_ids)`
   - Deep call stack: Model → TransformerLayer → Attention → ...

3. **Layer Access** (`python/minisgl/layers/attention.py`):
   - Layers call `get_global_ctx()` to access context
   - Access `ctx.batch` to get current batch information
   - Use batch info for attention metadata, padding, etc.

4. **Context Exit**:
   - When forward pass completes, context manager exits
   - Sets `_batch = None`
   - Context is ready for next batch

**Key Pattern**: Context is set once at the top (Engine), accessed many times deep in the stack (Layers).

================================================================================
COMPONENT CONNECTIONS: Global Context Through Call Stack
================================================================================

```mermaid
graph TB
    subgraph "Engine Layer"
        Engine[Engine]
        Ctx[Context]
        Batch[Batch]
    end
    
    subgraph "Model Layer"
        Model[Model]
        TransformerLayer[TransformerLayer]
    end
    
    subgraph "Attention Layer"
        Attention[Attention]
        AttnBackend[AttentionBackend]
    end
    
    subgraph "Global Access"
        GetCtx[get_global_ctx]
        BatchProp[ctx.batch]
    end
    
    Engine -->|"Creates"| Batch
    Engine -->|"Uses"| Ctx
    Ctx -->|"forward_batch context"| Batch
    Engine -->|"Calls"| Model
    Model -->|"Calls"| TransformerLayer
    TransformerLayer -->|"Calls"| Attention
    Attention -->|"Calls"| GetCtx
    GetCtx -->|"Returns"| Ctx
    Ctx -->|"Provides"| BatchProp
    BatchProp -->|"Used by"| AttnBackend
```

Key Connections:
- **Engine**: Sets context at the top of call stack
- **Layers**: Access context deep in call stack (no need to pass through all layers)
- **Global Context**: Thread-local storage (safe for single-threaded GPU ops)

================================================================================
TECHNICAL DECISIONS: Why Global Context vs Passing Everywhere
================================================================================

**Decision**: Use global context instead of passing context through every function call.

**Why?**
1. **Deep Call Stack**:
   - Call depth: Engine → Model → TransformerLayer → Attention → AttentionBackend
   - Passing context through all layers is verbose and error-prone
   - Every function signature would need `ctx` parameter

2. **Thread Safety**:
   - GPU operations are single-threaded (one batch at a time)
   - Thread-local storage is safe (no race conditions)
   - Context is only active during forward pass (isolated)

3. **API Cleanliness**:
   - Layers have cleaner signatures (no context parameter)
   - Easier to use layers independently
   - Less boilerplate code

**Alternative Considered**: Pass context as parameter through all layers.
   - **Problem**: Verbose - every function needs `ctx` parameter
   - **Problem**: Easy to forget to pass context
   - **Problem**: Breaks encapsulation (layers don't need to know about context)
   - **Chosen**: Global context for cleaner API

**Real Impact**: This design enables:
- Clean layer APIs (no context parameter needed)
- Easy to use layers independently
- Less code complexity

**Why Context Manager?**
1. **Automatic Cleanup**: Ensures `_batch` is cleared even if exception occurs
2. **Nested Prevention**: Assertion prevents nested forward_batch calls (would be confusing)
3. **Clear Scope**: Makes it obvious when context is active

**Why Prevent Nested forward_batch?**
- Only one batch should be processed at a time
- Nested calls would cause confusion (which batch is active?)
- Assertion catches bugs early

**Why Global Context Only Set Once?**
- Context is initialized once when Engine is created
- Global context should not change during execution
- Prevents accidental overwrites

================================================================================
CHALLENGE
================================================================================

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
4. **NEW**: Trace through the code: How does Attention layer access the context?
   (Hint: Read `python/minisgl/layers/attention.py` - look for `get_global_ctx()`)
5. **NEW**: Why is global context safe for GPU operations?
   (Hint: GPU operations are single-threaded - one batch processes at a time)

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
