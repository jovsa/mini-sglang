#!/bin/bash
# Run all comparison benchmarks: Mini-SGLang vs SGLang

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"

echo "=============================================="
echo "Mini-SGLang vs SGLang Comparison Benchmarks"
echo "=============================================="
echo "Model: $MODEL"
echo "Requests per test: $NUM_REQUESTS"
echo ""

# Check GPU availability
if ! python -c "import torch; assert torch.cuda.is_available()"; then
    echo "ERROR: No CUDA GPU available. Benchmarks require a GPU."
    exit 1
fi

# Check SGLang installation
echo "Checking SGLang installation..."
if python -c "import sglang; print(f'SGLang version: {getattr(sglang, \"__version__\", \"unknown\")}')"; then
    SGLANG_AVAILABLE=1
    echo "SGLang is installed. Will run comparison benchmarks."
else
    SGLANG_AVAILABLE=0
    echo "WARNING: SGLang is not installed. Running Mini-SGLang benchmarks only."
    echo "To install SGLang: pip install sglang"
fi
echo ""

# Create results directory
mkdir -p results

# Helper function
run_benchmark() {
    local name=$1
    local script=$2
    shift 2

    echo ""
    echo ">>> [$name]"
    echo "-------------------------------------------"

    if python "$script" "$@"; then
        echo "[$name] COMPLETED"
    else
        echo "[$name] FAILED"
    fi
}

# 1. Throughput Comparison
if [ "$SGLANG_AVAILABLE" -eq 1 ]; then
    run_benchmark "Throughput Comparison" 01_offline_throughput.py \
        --model "$MODEL" \
        --num-requests "$NUM_REQUESTS" \
        --input-lengths 256 \
        --output-lengths 128
else
    run_benchmark "Throughput (Mini-SGLang only)" 01_offline_throughput.py \
        --model "$MODEL" \
        --num-requests "$NUM_REQUESTS" \
        --input-lengths 256 \
        --output-lengths 128 \
        --minisgl-only
fi

# 2. Latency Comparison
if [ "$SGLANG_AVAILABLE" -eq 1 ]; then
    run_benchmark "Latency Comparison" 02_online_latency.py \
        --model "$MODEL" \
        --num-requests 20 \
        --input-lengths 256 \
        --output-lengths 64
else
    run_benchmark "Latency (Mini-SGLang only)" 02_online_latency.py \
        --model "$MODEL" \
        --num-requests 20 \
        --input-lengths 256 \
        --output-lengths 64 \
        --minisgl-only
fi

# 3. Memory Profiling (quick estimation only)
echo ""
echo ">>> [Memory Estimation]"
echo "-------------------------------------------"
python 03_memory_profile.py --model "$MODEL"
echo "[Memory Estimation] COMPLETED"

# Full memory profile (optional, uncomment to enable)
# if [ "$SGLANG_AVAILABLE" -eq 1 ]; then
#     run_benchmark "Memory Profile (Full)" 03_memory_profile.py \
#         --model "$MODEL" \
#         --full-profile
# else
#     run_benchmark "Memory Profile (Mini-SGLang)" 03_memory_profile.py \
#         --model "$MODEL" \
#         --full-profile \
#         --minisgl-only
# fi

# 4. Scaling (single GPU only by default)
run_benchmark "Scaling" 04_scaling.py \
    --model "$MODEL" \
    --tp-sizes 1 \
    --num-requests "$NUM_REQUESTS"

# 5. Feature Ablation vs SGLang Baseline
if [ "$SGLANG_AVAILABLE" -eq 1 ]; then
    run_benchmark "Ablation vs SGLang" 05_ablation.py \
        --model "$MODEL" \
        --num-requests 32 \
        --input-len 256 \
        --output-len 128
else
    run_benchmark "Ablation (no SGLang baseline)" 05_ablation.py \
        --model "$MODEL" \
        --num-requests 32 \
        --input-len 256 \
        --output-len 128 \
        --skip-sglang
fi

echo ""
echo "=============================================="
echo "All benchmarks complete!"
echo "=============================================="
echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Summary:"
ls -la results/*.json 2>/dev/null | tail -10 || echo "  (no JSON results found)"
echo ""
if [ "$SGLANG_AVAILABLE" -eq 1 ]; then
    echo "Comparison mode: Mini-SGLang vs SGLang"
else
    echo "Note: SGLang not installed. Install with 'pip install sglang' for comparisons."
fi
