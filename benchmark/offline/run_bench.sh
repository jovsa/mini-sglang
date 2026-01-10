#!/bin/bash

# Script to run both mini-sglang and sglang benchmarks with aligned parameters
# Usage: ./run_bench.sh [options]

set -e

# Default values
NUM_SEQS=256
MAX_INPUT_LEN=1024
MAX_OUTPUT_LEN=1024
MODEL_PATH="Qwen/Qwen3-0.6B"
MAX_SEQ_LEN_OVERRIDE=4096
MAX_EXTEND_TOKENS=16384
CUDA_GRAPH_MAX_BS=256
SEED=0
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-seqs) NUM_SEQS="$2"; shift 2 ;;
        --max-input-len) MAX_INPUT_LEN="$2"; shift 2 ;;
        --max-output-len) MAX_OUTPUT_LEN="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --max-seq-len-override) MAX_SEQ_LEN_OVERRIDE="$2"; shift 2 ;;
        --max-extend-tokens) MAX_EXTEND_TOKENS="$2"; shift 2 ;;
        --cuda-graph-max-bs) CUDA_GRAPH_MAX_BS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --num-seqs NUM                 Number of sequences (default: $NUM_SEQS)"
            echo "  --max-input-len NUM           Maximum input length (default: $MAX_INPUT_LEN)"
            echo "  --max-output-len NUM           Maximum output length (default: $MAX_OUTPUT_LEN)"
            echo "  --model-path PATH              Model path (default: $MODEL_PATH)"
            echo "  --max-seq-len-override NUM     Max sequence length override (default: $MAX_SEQ_LEN_OVERRIDE)"
            echo "  --max-extend-tokens NUM        Max extend tokens (default: $MAX_EXTEND_TOKENS)"
            echo "  --cuda-graph-max-bs NUM        CUDA graph max batch size (default: $CUDA_GRAPH_MAX_BS)"
            echo "  --seed NUM                     Random seed (default: $SEED)"
            echo "  --help, -h                     Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage information"; exit 1 ;;
    esac
done

# Build common arguments
COMMON_ARGS=(--num-seqs "$NUM_SEQS" --max-input-len "$MAX_INPUT_LEN" --max-output-len "$MAX_OUTPUT_LEN" --model-path "$MODEL_PATH" --max-seq-len-override "$MAX_SEQ_LEN_OVERRIDE" --max-extend-tokens "$MAX_EXTEND_TOKENS" --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" --seed "$SEED")

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Print configuration
echo "=========================================="
echo "Benchmark Configuration:"
echo "=========================================="
echo "  Model:              $MODEL_PATH"
echo "  Num Sequences:      $NUM_SEQS"
echo "  Max Input Length:   $MAX_INPUT_LEN"
echo "  Max Output Length:  $MAX_OUTPUT_LEN"
echo "  Max Seq Len:        $MAX_SEQ_LEN_OVERRIDE"
echo "  Max Extend Tokens:  $MAX_EXTEND_TOKENS"
echo "  CUDA Graph Max BS:  $CUDA_GRAPH_MAX_BS"
echo "  Seed:               $SEED"
echo "=========================================="
echo ""

# Run mini-sglang benchmark
echo "Running Mini-SGLang benchmark..."
cd "$PROJECT_ROOT"
MINISGL_OUTPUT=$(python "$BENCH_DIR/bench.py" "${COMMON_ARGS[@]}" 2>&1)
MINISGL_LINE=$(echo "$MINISGL_OUTPUT" | grep "^MINISGL:")

# Run sglang benchmark
echo "Running SGLang benchmark..."
SGLANG_OUTPUT=$(python "$BENCH_DIR/bench_sglang.py" "${COMMON_ARGS[@]}" 2>&1)
SGLANG_LINE=$(echo "$SGLANG_OUTPUT" | grep "^SGLANG:")

# Print aligned results
echo ""
echo "=========================================="
echo "Benchmark Results:"
echo "=========================================="
if [ -n "$MINISGL_LINE" ]; then
    echo "$MINISGL_LINE"
else
    echo "MINISGL: (failed to parse output)"
    echo "$MINISGL_OUTPUT"
fi

if [ -n "$SGLANG_LINE" ]; then
    echo "$SGLANG_LINE"
else
    echo "SGLANG:  (failed to parse output)"
    echo "$SGLANG_OUTPUT"
fi
echo "=========================================="

# Extract and compare metrics if both succeeded
if [ -n "$MINISGL_LINE" ] && [ -n "$SGLANG_LINE" ]; then
    MINISGL_THROUGHPUT=$(echo "$MINISGL_LINE" | grep -oP 'Throughput: \K[0-9.]+')
    SGLANG_THROUGHPUT=$(echo "$SGLANG_LINE" | grep -oP 'Throughput: \K[0-9.]+')

    if [ -n "$MINISGL_THROUGHPUT" ] && [ -n "$SGLANG_THROUGHPUT" ]; then
        # Calculate speedup (SGLang vs Mini-SGLang)
        if command -v bc &> /dev/null; then
            SPEEDUP=$(echo "scale=2; $SGLANG_THROUGHPUT / $MINISGL_THROUGHPUT" | bc)
            echo ""
            echo "Speedup (SGLang / Mini-SGLang): ${SPEEDUP}x"
        fi
    fi
fi
