"""
PUZZLE 3.1: Create Message Classes - [Easy]

TASK:
Implement TokenizeMsg and UserMsg classes following the message system patterns.

GIVEN:
- Base message class structure
- Understanding of message types

CHALLENGE:
- Decide what fields each message needs
- Follow the dataclass pattern
- Understand message serialization

HINT:
- Reference: `python/minisgl/message/backend.py:33-36` for UserMsg fields
- Reference: `python/minisgl/message/tokenizer.py:34-38` for TokenizeMsg fields
- Both messages need: uid, sampling_params
- TokenizeMsg has text (str), UserMsg has input_ids (torch.Tensor)

QUESTIONS:
1. What fields does UserMsg need? Why?
2. What's the difference between TokenizeMsg and UserMsg?
3. Why do messages need serialization?

TEST:
Run: pytest learning/puzzles/03_messages/test_3.1.py -v
"""

from dataclasses import dataclass
from typing import Optional
import torch
import sys
import importlib.util
from pathlib import Path

# Import puzzle_1.1 using importlib (module name has dot)
puzzle_dir = Path(__file__).parent.parent / "01_core_structures"
puzzle_1_1_file = puzzle_dir / "puzzle_1.1.py"
spec = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_1_1)
create_sampling_params = puzzle_1_1.create_sampling_params


@dataclass
class BaseMessage:
    """Base class for all messages."""
    pass


@dataclass
class TokenizeMsg(BaseMessage):
    """
    Message to request tokenization of text.

    TODO: Add the required fields:
    - uid: int - Unique request ID
    - text: str - Text to tokenize
    - sampling_params: object - Sampling parameters (use SamplingParams from puzzle_1_1)
    """
    # YOUR CODE HERE
    pass


@dataclass
class UserMsg(BaseMessage):
    """
    Message containing tokenized user request.

    TODO: Add the required fields:
    - uid: int - Unique request ID
    - input_ids: torch.Tensor - Tokenized input (CPU tensor)
    - sampling_params: object - Sampling parameters
    """
    # YOUR CODE HERE
    pass


def create_tokenize_msg(uid: int, text: str, max_tokens: int = 1024) -> TokenizeMsg:
    """
    Create a TokenizeMsg.

    TODO: Implement this function
    """
    # YOUR CODE HERE
    pass


def create_user_msg(uid: int, input_ids: torch.Tensor, max_tokens: int = 1024) -> UserMsg:
    """
    Create a UserMsg from tokenized input.

    TODO: Implement this function
    """
    # YOUR CODE HERE
    pass


# Test your implementation
if __name__ == "__main__":
    # Test TokenizeMsg
    msg1 = create_tokenize_msg(uid=1, text="Hello, world!", max_tokens=50)
    print(f"TokenizeMsg: uid={msg1.uid}, text='{msg1.text}'")

    # Test UserMsg
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    msg2 = create_user_msg(uid=2, input_ids=input_ids, max_tokens=100)
    print(f"UserMsg: uid={msg2.uid}, input_ids={msg2.input_ids.tolist()}")
