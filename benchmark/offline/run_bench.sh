#!/bin/bash

# Script to run both mini-sglang and sglang benchmarks with aligned parameters
#
# Usage Examples:
#   Run with a specific config:
#     ./run_bench.sh --configs "tiny:16:128:128"
#
#   Run with multiple specific configs:
#     ./run_bench.sh --configs "tiny:16:128:128" "small:64:512:512"
#
#   Run with all default configs:
#     ./run_bench.sh --multi-config
#
#   Run with verbose output:
#     ./run_bench.sh --configs "tiny:16:128:128" --verbose
#
# Config format: "name:num_seqs:max_input_len:max_output_len"

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
VERBOSE=false
USE_MULTI_CONFIG=false
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BENCH_DIR/../.." && pwd)"

# Default configurations for multi-config mode
# Format: "name:num_seqs:max_input_len:max_output_len"
#   - name: configuration name
#   - num_seqs: number of sequences
#   - max_input_len: maximum input length (tokens)
#   - max_output_len: maximum output length (tokens)
DEFAULT_CONFIGS=(
    "tiny:2:5:5"        # 16 seqs, 128 input, 128 output
    "small:64:512:512"       # 64 seqs, 512 input, 512 output
    "medium:256:1024:1024"   # 256 seqs, 1024 input, 1024 output
    "large:512:2048:2048"    # 512 seqs, 2048 input, 2048 output
)

# Parse command line arguments
CONFIGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v) VERBOSE=true; shift ;;
        --multi-config|--configs)
            USE_MULTI_CONFIG=true
            shift
            # Collect all configs until next option or end
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CONFIGS+=("$1")
                shift
            done
            ;;
        --help|-h)
            echo "Usage: $0 [--configs CONFIG1 CONFIG2 ...] [--multi-config] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --configs, --multi-config      Run with configurations (see DEFAULT_CONFIGS in script)"
            echo "  --verbose, -v                  Show detailed metrics"
            echo "  --help, -h                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --configs \"tiny:16:128:128\"               # Run a specific config"
            echo "  $0 --multi-config                    # Run ALL default configs"
            echo "  $0 --configs \"tiny:16:128:128\" --verbose   # Run with verbose output"
            exit 0
            ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage information"; exit 1 ;;
    esac
done

# If multi-config mode but no configs provided, use defaults
# If no config mode specified, default to all configs
if [ "$USE_MULTI_CONFIG" = false ]; then
    USE_MULTI_CONFIG=true
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
elif [ "$USE_MULTI_CONFIG" = true ] && [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Function to run a single benchmark configuration
run_single_bench() {
    local config_name="$1"
    local num_seqs="$2"
    local max_input_len="$3"
    local max_output_len="$4"

    # Build common arguments
    local common_args=(--num-seqs "$num_seqs" --max-input-len "$max_input_len" --max-output-len "$max_output_len" --model-path "$MODEL_PATH" --max-seq-len-override "$MAX_SEQ_LEN_OVERRIDE" --max-extend-tokens "$MAX_EXTEND_TOKENS" --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" --seed "$SEED")
    if [ "$VERBOSE" = true ]; then
        common_args+=(--verbose)
    fi

    # Print configuration
    echo ""
    echo "=========================================="
    if [ -n "$config_name" ]; then
        echo "Configuration: $config_name"
    else
        echo "Benchmark Configuration:"
    fi
    echo "=========================================="
    echo "  Model:              $MODEL_PATH"
    echo "  Num Sequences:      $num_seqs"
    echo "  Max Input Length:   $max_input_len"
    echo "  Max Output Length:  $max_output_len"
    echo "  Max Seq Len:        $MAX_SEQ_LEN_OVERRIDE"
    echo "  Max Extend Tokens:  $MAX_EXTEND_TOKENS"
    echo "  CUDA Graph Max BS:  $CUDA_GRAPH_MAX_BS"
    echo "  Seed:               $SEED"
    echo "=========================================="
    echo ""

    # Run mini-sglang benchmark
    echo "Running Mini-SGLang benchmark..."
    cd "$PROJECT_ROOT"
    local minisgl_output=$(python "$BENCH_DIR/bench.py" "${common_args[@]}" 2>&1)
    local minisgl_line=$(echo "$minisgl_output" | grep "^MINISGL:" || true)

    # Run sglang benchmark
    echo "Running SGLang benchmark..."
    local sglang_output=$(python "$BENCH_DIR/bench_sglang.py" "${common_args[@]}" 2>&1)
    local sglang_line=$(echo "$sglang_output" | grep "^SGLANG:" || true)

    # Print aligned results
    echo ""
    echo "=========================================="
    if [ -n "$config_name" ]; then
        echo "Results for $config_name:"
    else
        echo "Benchmark Results:"
    fi
    echo "=========================================="
    if [ -n "$minisgl_line" ]; then
        echo "$minisgl_line"
    else
        echo "MINISGL: (failed to parse output)"
        echo "$minisgl_output"
    fi

    if [ -n "$sglang_line" ]; then
        echo "$sglang_line"
    else
        echo "SGLANG:  (failed to parse output)"
        echo "$sglang_output"
    fi
    echo "=========================================="

    # Extract and compare metrics if both succeeded
    if [ -n "$minisgl_line" ] && [ -n "$sglang_line" ]; then
        local minisgl_throughput=$(echo "$minisgl_line" | grep -oP 'Throughput: \K[0-9.]+' || true)
        local sglang_throughput=$(echo "$sglang_line" | grep -oP 'Throughput: \K[0-9.]+' || true)

        if [ -n "$minisgl_throughput" ] && [ -n "$sglang_throughput" ]; then
            # Calculate speedup (SGLang vs Mini-SGLang)
            if command -v bc &> /dev/null; then
                local speedup=$(echo "scale=2; $sglang_throughput / $minisgl_throughput" | bc)
                echo ""
                echo "Speedup (SGLang / Mini-SGLang): ${speedup}x"
            fi
        fi

    fi

    # Return results for summary
    echo "$minisgl_line|$sglang_line|$config_name"
}

# Main execution
if [ "$USE_MULTI_CONFIG" = true ]; then
    # Multi-config mode
    echo "=========================================="
    echo "Running Multiple Benchmark Configurations"
    echo "=========================================="
    echo "Total configurations: ${#CONFIGS[@]}"
    echo ""

    RESULTS=()
    CONFIG_NAMES=()

    for config in "${CONFIGS[@]}"; do
        # Parse config: "name:num_seqs:max_input_len:max_output_len"
        IFS=':' read -r config_name config_num_seqs config_max_input_len config_max_output_len <<< "$config"

        if [ -z "$config_num_seqs" ] || [ -z "$config_max_input_len" ] || [ -z "$config_max_output_len" ]; then
            echo "Warning: Invalid config format '$config'. Expected 'name:num_seqs:max_input_len:max_output_len'. Skipping."
            continue
        fi

        CONFIG_NAMES+=("$config_name")
        result=$(run_single_bench "$config_name" "$config_num_seqs" "$config_max_input_len" "$config_max_output_len")
        RESULTS+=("$result")

        # Add separator between configs
        if [ "$config" != "${CONFIGS[-1]}" ]; then
            echo ""
            echo "=========================================="
            echo ""
        fi
    done

    # Print summary
    echo ""
    echo "=========================================="
    echo "Summary of All Configurations"
    echo "=========================================="
    printf "%-15s %-20s %-20s %-15s\n" "Config" "Mini-SGLang" "SGLang" "Speedup"
    echo "--------------------------------------------"

    for i in "${!RESULTS[@]}"; do
        IFS='|' read -r minisgl_line sglang_line config_name <<< "${RESULTS[$i]}"

        if [ -n "$minisgl_line" ] && [ -n "$sglang_line" ]; then
            minisgl_throughput=$(echo "$minisgl_line" | grep -oP 'Throughput: \K[0-9.]+' || echo "N/A")
            sglang_throughput=$(echo "$sglang_line" | grep -oP 'Throughput: \K[0-9.]+' || echo "N/A")

            speedup="N/A"
            if [ "$minisgl_throughput" != "N/A" ] && [ "$sglang_throughput" != "N/A" ] && command -v bc &> /dev/null; then
                speedup=$(echo "scale=2; $sglang_throughput / $minisgl_throughput" | bc)
            fi

            printf "%-15s %-20s %-20s %-15s\n" "${CONFIG_NAMES[$i]}" "${minisgl_throughput} tok/s" "${sglang_throughput} tok/s" "${speedup}x"
        else
            printf "%-15s %-20s %-20s %-15s\n" "${CONFIG_NAMES[$i]}" "FAILED" "FAILED" "N/A"
        fi
    done
    echo "=========================================="
fi
