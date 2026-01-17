"""
PUZZLE 5.2: DecodeManager - [Easy]

TASK:
Implement a simplified DecodeManager that tracks running requests in the decode phase.

GIVEN:
- Understanding of prefill vs decode phases
- Requests transition from prefill to decode after initial processing

CHALLENGE:
- Implement add_reqs() to add requests that can continue decoding
- Implement remove_req() to remove finished requests
- Implement inflight_tokens property to track total tokens being generated
- Implement runnable property to check if there are requests to process

HINT:
- Reference: `python/minisgl/scheduler/decode.py:10-30` for DecodeManager
- Use a Set to track running requests
- Only add requests where can_decode() returns True
- inflight_tokens is the sum of remain_len for all running requests

QUESTIONS:
1. Why use a Set instead of a List for running_reqs?
2. What does inflight_tokens tell us about system load?
3. When would can_decode() return False?

TEST:
Run: pytest learning/puzzles/05_scheduler/test_5.2.py -v
"""

from dataclasses import dataclass, field
from typing import Set, Iterable


@dataclass
class MockReq:
    """Mock request for testing."""
    uid: int
    remain_len: int

    def can_decode(self) -> bool:
        """Check if request can continue decoding."""
        return self.remain_len > 0


@dataclass
class SimpleDecodeManager:
    """
    Manages requests in the decode phase.

    Tracks which requests are currently running and need to generate more tokens.
    """
    running_reqs: Set[MockReq] = field(default_factory=set)

    def add_reqs(self, reqs: Iterable[MockReq]) -> None:
        """
        Add requests that can continue decoding.

        Only adds requests where can_decode() returns True.

        Args:
            reqs: Iterable of requests to potentially add

        HINT: Use a generator expression with update()
              Only add req if req.can_decode() is True
        """
        # YOUR CODE HERE
        pass

    def remove_req(self, req: MockReq) -> None:
        """
        Remove a finished request.

        Uses discard() to safely remove (no error if not present).

        Args:
            req: The request to remove
        """
        # YOUR CODE HERE
        pass

    @property
    def inflight_tokens(self) -> int:
        """
        Calculate total tokens being generated across all running requests.

        Returns:
            Sum of remain_len for all running requests
        """
        # YOUR CODE HERE
        pass

    @property
    def runnable(self) -> bool:
        """
        Check if there are any requests to process.

        Returns:
            True if running_reqs is not empty
        """
        # YOUR CODE HERE
        pass


def create_decode_manager() -> SimpleDecodeManager:
    """Create a new SimpleDecodeManager."""
    return SimpleDecodeManager()


def create_mock_req(uid: int, remain_len: int) -> MockReq:
    """Create a mock request for testing."""
    return MockReq(uid=uid, remain_len=remain_len)


# Test your implementation
if __name__ == "__main__":
    manager = create_decode_manager()
    print(f"Initial runnable: {manager.runnable} (expected False)")
    print(f"Initial inflight_tokens: {manager.inflight_tokens} (expected 0)")

    # Add some requests
    req1 = create_mock_req(uid=1, remain_len=10)
    req2 = create_mock_req(uid=2, remain_len=5)
    req3 = create_mock_req(uid=3, remain_len=0)  # Can't decode

    manager.add_reqs([req1, req2, req3])
    print(f"\nAfter adding 3 reqs (one with remain_len=0):")
    print(f"  runnable: {manager.runnable} (expected True)")
    print(f"  inflight_tokens: {manager.inflight_tokens} (expected 15)")

    # Remove a request
    manager.remove_req(req1)
    print(f"\nAfter removing req1:")
    print(f"  inflight_tokens: {manager.inflight_tokens} (expected 5)")
