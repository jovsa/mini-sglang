# Puzzle Placeholders

This directory contains puzzles organized by topic. Some puzzles are fully implemented, others are placeholders for future development.

## Implemented Puzzles

- **01_core_structures**: All 4 puzzles complete (1.1-1.4)
- **02_simple_llm**: Puzzle 2.1 complete
- **03_messages**: Puzzle 3.1 complete

## Placeholder Puzzles

The following puzzle sets are structured but need implementation:

- **04_scheduler_basic**: Scheduler puzzles
- **05_request_lifecycle**: Request flow puzzles
- **06_batch_processing**: Batching puzzles
- **07_kvcache_basics**: KV cache puzzles
- **08_attention_backends**: Attention puzzles
- **09_custom_sampling**: Sampling puzzles
- **10_minimal_server**: Server setup puzzles
- **11_debugging**: Debugging puzzles
- **12_performance**: Performance puzzles
- **13_add_message_type**: Contribution puzzle
- **14_extend_scheduler**: Contribution puzzle
- **15_add_model**: Contribution puzzle
- **16_modify_attention**: Contribution puzzle
- **17_testing**: Testing puzzles

## Adding New Puzzles

To add a new puzzle:

1. Create `puzzle_X.Y.py` in the appropriate directory
2. Follow the format from existing puzzles
3. Create corresponding `test_X.Y.py`
4. Create `solution_X.Y.py` in solutions directory
5. Update this file

See the plan document for detailed puzzle specifications.
