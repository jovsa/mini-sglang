"""
Test for Puzzle 6.1: Context Management

Run with: pytest learning/puzzles/06_context/test_6.1.py -v
"""

import pytest
import sys
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

# Import puzzle file with dot in name using importlib
puzzle_file = puzzle_dir / "puzzle_6.1.py"
spec = importlib.util.spec_from_file_location("puzzle_6_1", puzzle_file)
puzzle_6_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_6_1)
SimpleContext = puzzle_6_1.SimpleContext
MockBatch = puzzle_6_1.MockBatch
set_global_ctx = puzzle_6_1.set_global_ctx
get_global_ctx = puzzle_6_1.get_global_ctx
reset_global_ctx = puzzle_6_1.reset_global_ctx


@pytest.fixture(autouse=True)
def reset_ctx():
    """Reset global context before each test."""
    reset_global_ctx()
    yield
    reset_global_ctx()


def test_context_creation():
    """Test SimpleContext creation"""
    ctx = SimpleContext(page_size=16)
    assert ctx.page_size == 16


def test_batch_raises_when_none():
    """Test that batch property raises when no batch is active"""
    ctx = SimpleContext(page_size=16)
    with pytest.raises(AssertionError):
        _ = ctx.batch


def test_forward_batch_sets_batch():
    """Test that forward_batch makes batch accessible"""
    ctx = SimpleContext(page_size=16)
    batch = MockBatch(size=4, phase="prefill")

    with ctx.forward_batch(batch):
        assert ctx.batch is batch
        assert ctx.batch.size == 4
        assert ctx.batch.phase == "prefill"


def test_forward_batch_clears_batch():
    """Test that batch is cleared after forward_batch exits"""
    ctx = SimpleContext(page_size=16)
    batch = MockBatch(size=4, phase="prefill")

    with ctx.forward_batch(batch):
        assert ctx.batch is batch

    with pytest.raises(AssertionError):
        _ = ctx.batch


def test_forward_batch_clears_on_exception():
    """Test that batch is cleared even if exception occurs"""
    ctx = SimpleContext(page_size=16)
    batch = MockBatch(size=4, phase="prefill")

    try:
        with ctx.forward_batch(batch):
            raise ValueError("Test error")
    except ValueError:
        pass

    with pytest.raises(AssertionError):
        _ = ctx.batch


def test_nested_forward_batch_raises():
    """Test that nested forward_batch raises AssertionError"""
    ctx = SimpleContext(page_size=16)
    batch1 = MockBatch(size=4, phase="prefill")
    batch2 = MockBatch(size=2, phase="decode")

    with pytest.raises(AssertionError):
        with ctx.forward_batch(batch1):
            with ctx.forward_batch(batch2):
                pass


def test_set_global_ctx():
    """Test setting global context"""
    ctx = SimpleContext(page_size=32)
    set_global_ctx(ctx)
    assert get_global_ctx() is ctx


def test_get_global_ctx_raises_when_not_set():
    """Test that get_global_ctx raises when not set"""
    with pytest.raises(AssertionError):
        get_global_ctx()


def test_set_global_ctx_twice_raises():
    """Test that setting global context twice raises"""
    ctx1 = SimpleContext(page_size=16)
    ctx2 = SimpleContext(page_size=32)

    set_global_ctx(ctx1)
    with pytest.raises(AssertionError):
        set_global_ctx(ctx2)


def test_global_ctx_page_size():
    """Test accessing page_size through global context"""
    ctx = SimpleContext(page_size=64)
    set_global_ctx(ctx)
    assert get_global_ctx().page_size == 64


def test_forward_batch_through_global_ctx():
    """Test forward_batch through global context"""
    ctx = SimpleContext(page_size=16)
    set_global_ctx(ctx)
    batch = MockBatch(size=8, phase="decode")

    global_ctx = get_global_ctx()
    with global_ctx.forward_batch(batch):
        assert global_ctx.batch.size == 8


def test_multiple_forward_batch_sequential():
    """Test multiple forward_batch calls in sequence"""
    ctx = SimpleContext(page_size=16)
    batch1 = MockBatch(size=4, phase="prefill")
    batch2 = MockBatch(size=2, phase="decode")

    with ctx.forward_batch(batch1):
        assert ctx.batch.size == 4

    with ctx.forward_batch(batch2):
        assert ctx.batch.size == 2
