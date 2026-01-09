# Mini-SGLang Learning Prototypes & Benchmarks

This directory contains learning prototypes and benchmarking tools for understanding Mini-SGLang.

## Directory Structure

```
prototypes/
├── learning/                      # Educational prototypes (no GPU required)
│   ├── 01_core_data_structures.py # Req, Batch, Context lifecycle
│   └── 02_radix_cache.py          # Radix Tree for KV cache
├── benchmarks/                    # Performance benchmarks (GPU required)
│   ├── common.py                  # Shared utilities
│   ├── 01_offline_throughput.py   # Throughput measurement
│   ├── 02_online_latency.py       # TTFT, TPOT, E2E latency
│   ├── 03_memory_profile.py       # GPU memory analysis
│   ├── 04_scaling.py              # Multi-GPU scaling
│   ├── 05_ablation.py             # Feature impact analysis
│   ├── run_all.sh                 # Run complete suite
│   └── results/                   # Benchmark outputs
└── README.md                      # This file
```

## Learning Prototypes

These prototypes help understand key Mini-SGLang concepts. They run without a GPU.

### 1. Core Data Structures

```bash
python learning/01_core_data_structures.py
```

Demonstrates:
- `Req`: Request lifecycle and state tracking
- `Batch`: Grouping requests for prefill/decode
- `Context`: Global state management
- Prefix caching concepts

### 2. Radix Cache

```bash
python learning/02_radix_cache.py
```

Demonstrates:
- Radix Tree structure and operations
- Prefix matching and sharing
- Node splitting for partial matches
- LRU eviction policy
- Multi-turn chat efficiency gains

## Benchmarks

These require a CUDA GPU. Install Mini-SGLang first:

```bash
cd /workspace/mini-sglang
pip install -e .
```

### Quick Start

Run all benchmarks with default settings:

```bash
cd benchmarks
chmod +x run_all.sh
./run_all.sh
```

Or with custom model:

```bash
MODEL="Qwen/Qwen3-14B" NUM_REQUESTS=128 ./run_all.sh
```

### Individual Benchmarks

#### 1. Offline Throughput

```bash
python 01_offline_throughput.py --model "Qwen/Qwen3-0.6B" --num-requests 64
```

Measures: tokens/sec, requests/sec at various batch sizes.

#### 2. Online Latency

```bash
python 02_online_latency.py --model "Qwen/Qwen3-0.6B" --num-requests 50
```

Measures: TTFT, TPOT, E2E latency with percentiles (p50, p90, p99).

#### 3. Memory Profiling

```bash
python 03_memory_profile.py --model "Qwen/Qwen3-0.6B" --full-profile
```

Measures: Model memory, KV cache size, peak usage.

#### 4. Multi-GPU Scaling

```bash
python 04_scaling.py --model "Qwen/Qwen3-0.6B" --tp-sizes 1 2 4
```

Measures: Throughput scaling efficiency across GPUs.

#### 5. Feature Ablation

```bash
python 05_ablation.py --model "Qwen/Qwen3-0.6B" --num-requests 32
```

Measures impact of:
- Radix Cache vs Naive Cache
- Overlap Scheduling
- CUDA Graph

## Results

Benchmark results are saved to `benchmarks/results/` in JSON and CSV format:

- `throughput_TIMESTAMP.json` - Raw throughput data
- `latency_TIMESTAMP.json` - Raw latency data
- `memory_TIMESTAMP.json` - Memory profiling data
- `scaling_TIMESTAMP.json` - Scaling data
- `ablation_TIMESTAMP.json` - Ablation study data

## Comparing with SGLang

To compare with the full SGLang framework:

```bash
# Install SGLang
pip install sglang[all]

# Run SGLang server
python -m sglang.launch_server --model "Qwen/Qwen3-0.6B" --port 8001

# Use the same benchmark client against SGLang
python 02_online_latency.py --port 8001 --no-start-server
```

## Metrics Explained

- **TTFT (Time to First Token)**: Latency until first token is generated
- **TPOT (Time Per Output Token)**: Average time between tokens during decoding
- **E2E (End-to-End)**: Total time from request to completion
- **Throughput**: Total output tokens generated per second
- **Scaling Efficiency**: Actual speedup vs ideal linear speedup
