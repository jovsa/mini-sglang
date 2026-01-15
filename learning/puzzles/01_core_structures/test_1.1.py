"""
Test for Puzzle 1.1: Complete the SamplingParams

Run with: pytest learning/puzzles/01_core_structures/test_1.1.py -v
"""

import pytest
import sys
from pathlib import Path

# Add puzzle directory to path
puzzle_dir = Path(__file__).parent
sys.path.insert(0, str(puzzle_dir))

from puzzle_1_1 import create_sampling_params, SamplingParams


def test_create_sampling_params_basic():
    """Test basic SamplingParams creation"""
    params = create_sampling_params(temperature=0.0, top_k=1, top_p=1.0)
    assert params is not None, "create_sampling_params should return a SamplingParams object"
    assert params.temperature == 0.0
    assert params.top_k == 1
    assert params.top_p == 1.0
    assert params.ignore_eos == False
    assert params.max_tokens == 1024


def test_create_sampling_params_custom():
    """Test SamplingParams with custom values"""
    params = create_sampling_params(
        temperature=0.7,
        top_k=10,
        top_p=0.9,
        ignore_eos=True,
        max_tokens=512
    )
    assert params.temperature == 0.7
    assert params.top_k == 10
    assert params.top_p == 0.9
    assert params.ignore_eos == True
    assert params.max_tokens == 512


def test_is_greedy_temperature_zero():
    """Test greedy detection with temperature=0.0"""
    params = create_sampling_params(temperature=0.0, top_k=10, top_p=0.9)
    assert params.is_greedy == True, "temperature=0.0 should be greedy regardless of top_k/top_p"


def test_is_greedy_top_k_one_top_p_one():
    """Test greedy detection with top_k=1 and top_p=1.0"""
    params = create_sampling_params(temperature=0.7, top_k=1, top_p=1.0)
    assert params.is_greedy == True, "top_k=1 and top_p=1.0 should be greedy"


def test_is_greedy_sampling():
    """Test non-greedy (sampling) detection"""
    params = create_sampling_params(temperature=0.7, top_k=10, top_p=0.9)
    assert params.is_greedy == False, "temperature>0 and top_k>1 should not be greedy"


def test_is_greedy_edge_cases():
    """Test edge cases for greedy detection"""
    # Temperature 0 but top_p < 1 - should still be greedy (temperature takes precedence)
    params1 = create_sampling_params(temperature=0.0, top_p=0.5)
    assert params1.is_greedy == True, "temperature=0.0 should always be greedy"

    # top_k=1 but top_p < 1 - should not be greedy
    params2 = create_sampling_params(temperature=0.7, top_k=1, top_p=0.5)
    assert params2.is_greedy == False, "top_k=1 alone is not enough, need top_p=1.0"

    # top_p=1.0 but top_k > 1 - should not be greedy (unless temperature=0)
    params3 = create_sampling_params(temperature=0.7, top_k=10, top_p=1.0)
    assert params3.is_greedy == False, "top_p=1.0 alone is not enough, need top_k=1"


def test_is_greedy_negative_temperature():
    """Test with negative temperature (edge case)"""
    params = create_sampling_params(temperature=-1.0)
    assert params.is_greedy == True, "negative temperature should be treated as greedy"


def test_sampling_params_defaults():
    """Test default values"""
    params = create_sampling_params()
    assert params.temperature == 0.0
    assert params.top_k == -1
    assert params.top_p == 1.0
    assert params.ignore_eos == False
    assert params.max_tokens == 1024
    # Default should be greedy (temperature=0.0)
    assert params.is_greedy == True
