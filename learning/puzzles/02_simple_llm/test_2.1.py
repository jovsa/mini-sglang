"""
Test for Puzzle 2.1: Predict Sampling Behavior

Run with: pytest learning/puzzles/02_simple_llm/test_2.1.py -v
"""

import pytest
import sys
import importlib.util
from pathlib import Path

puzzle_dir = Path(__file__).parent

# Import puzzle_2.1 using importlib (module name has dot)
puzzle_2_1_file = puzzle_dir / "puzzle_2.1.py"
spec_2_1 = importlib.util.spec_from_file_location("puzzle_2_1", puzzle_2_1_file)
puzzle_2_1 = importlib.util.module_from_spec(spec_2_1)
spec_2_1.loader.exec_module(puzzle_2_1)
predict_sampling_behavior = puzzle_2_1.predict_sampling_behavior
compare_sampling_configs = puzzle_2_1.compare_sampling_configs

# Import puzzle_1.1 using importlib (module name has dot)
puzzle_1_1_dir = puzzle_dir.parent / "01_core_structures"
puzzle_1_1_file = puzzle_1_1_dir / "puzzle_1.1.py"
spec_1_1 = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec_1_1)
spec_1_1.loader.exec_module(puzzle_1_1)
create_sampling_params = puzzle_1_1.create_sampling_params


def test_predict_greedy():
    """Test prediction for greedy sampling"""
    params = create_sampling_params(temperature=0.0, top_k=1, top_p=1.0)
    behavior = predict_sampling_behavior(params)

    assert behavior["deterministic"] == True, "Greedy sampling should be deterministic"
    assert behavior["diversity"] == "low", "Greedy sampling has low diversity"


def test_predict_creative():
    """Test prediction for creative sampling"""
    params = create_sampling_params(temperature=1.0, top_k=50, top_p=0.9)
    behavior = predict_sampling_behavior(params)

    assert behavior["deterministic"] == False, "High temperature should not be deterministic"
    assert behavior["diversity"] == "high", "High temperature should have high diversity"


def test_predict_balanced():
    """Test prediction for balanced sampling"""
    params = create_sampling_params(temperature=0.7, top_k=10, top_p=0.95)
    behavior = predict_sampling_behavior(params)

    assert behavior["deterministic"] == False
    assert behavior["diversity"] in ["medium", "high"]


def test_vocab_size_limited():
    """Test vocab size prediction"""
    # top_k limits vocabulary
    params = create_sampling_params(temperature=0.7, top_k=5)
    behavior = predict_sampling_behavior(params)
    assert behavior["vocab_size"] == "limited"

    # top_p also limits vocabulary
    params2 = create_sampling_params(temperature=0.7, top_p=0.8)
    behavior2 = predict_sampling_behavior(params2)
    assert behavior2["vocab_size"] in ["limited", "very_limited"]


def test_compare_configs():
    """Test comparison of multiple configs"""
    results = compare_sampling_configs()

    assert "greedy" in results
    assert "creative" in results
    assert "balanced" in results
    assert "focused" in results

    # Greedy should be most deterministic
    assert results["greedy"]["deterministic"] == True
    # Creative should have high diversity
    assert results["creative"]["diversity"] == "high"
