#!/usr/bin/env python3
"""
Multi-GPU Scaling Benchmark
===========================

Measures throughput scaling across multiple GPUs with Tensor Parallelism.

Usage:
    python 04_scaling.py [--model MODEL] [--tp-sizes 1 2 4]
"""

import argparse
import subprocess
import sys
import json
import os
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    ScalingResult, save_results, check_gpu_available, 
    get_gpu_memory_info, print_comparison_table
)


def count_gpus() -> int:
    """Count available CUDA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True
        )
        return len(result.stdout.strip().split("\n"))
    except:
        return 0


def run_single_tp_test(
    model: str, 
    tp_size: int, 
    num_requests: int, 
    input_len: int, 
    output_len: int
) -> Optional[float]:
    """Run throughput test in a subprocess and return tokens/sec."""
    
    if tp_size > 1:
        # Multi-GPU TP requires torchrun or similar
        print(f"    Note: TP={tp_size} requires multi-process setup (torchrun)")
        print(f"    Skipping TP>{1} for now (single-GPU benchmark)")
        return None
    
    script = f'''
import sys
import json
import time
from random import randint, seed

# Reset TP info before importing LLM
import minisgl.distributed.info as info_module
info_module._TP_INFO = None

from minisgl.core import SamplingParams
from minisgl.llm import LLM

model = "{model}"
num_requests = {num_requests}
input_len = {input_len}
output_len = {output_len}

# Create LLM
llm = LLM(model, max_seq_len_override=2048, cuda_graph_max_bs=128)

# Generate test data
seed(42)
prompts = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_requests)]
params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
          for _ in range(num_requests)]

# Warmup
llm.generate(["warmup"], SamplingParams(max_tokens=10))

# Benchmark
start = time.time()
llm.generate(prompts, params)
elapsed = time.time() - start

total_tokens = num_requests * output_len
throughput = total_tokens / elapsed

print("RESULT:" + json.dumps({{"throughput": throughput}}))
'''
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                return data["throughput"]
        
        if result.returncode != 0:
            print(f"    Error: {result.stderr[-300:] if result.stderr else 'Unknown'}")
        return None
        
    except subprocess.TimeoutExpired:
        print("    Timed out")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU scaling benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model to benchmark")
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[1],
                       help="TP sizes to test (default: [1])")
    parser.add_argument("--num-requests", type=int, default=64, help="Number of requests")
    parser.add_argument("--input-len", type=int, default=256, help="Input length")
    parser.add_argument("--output-len", type=int, default=128, help="Output length")
    args = parser.parse_args()
    
    if not check_gpu_available():
        print("ERROR: No GPU available.")
        return
    
    num_gpus = count_gpus()
    
    print("=" * 60)
    print("Multi-GPU Scaling Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Available GPUs: {num_gpus}")
    print(f"TP sizes to test: {args.tp_sizes}")
    print()
    
    tp_sizes_tested: List[int] = []
    throughputs: List[float] = []
    
    for tp_size in args.tp_sizes:
        if tp_size > num_gpus:
            print(f"\nSkipping TP={tp_size} (only {num_gpus} GPUs available)")
            continue
        
        print(f"\nTesting TP={tp_size}:")
        throughput = run_single_tp_test(
            args.model, tp_size, args.num_requests, args.input_len, args.output_len
        )
        
        if throughput:
            tp_sizes_tested.append(tp_size)
            throughputs.append(throughput)
            print(f"  Throughput: {throughput:.2f} tok/s")
    
    # Print scaling summary
    if throughputs:
        print("\n" + "=" * 60)
        print("Scaling Results")
        print("=" * 60)
        
        baseline = throughputs[0] if throughputs else 0
        print(f"\n{'TP Size':<10} {'Throughput (tok/s)':<20} {'Scaling Efficiency':>20}")
        print("-" * 50)
        
        for tp_size, throughput in zip(tp_sizes_tested, throughputs):
            expected = baseline * tp_size
            efficiency = (throughput / expected) * 100 if expected > 0 else 0
            print(f"{tp_size:<10} {throughput:<20.2f} {efficiency:>19.1f}%")
        
        # Calculate overall scaling efficiency
        if len(throughputs) > 1:
            max_tp = max(tp_sizes_tested)
            max_throughput = throughputs[tp_sizes_tested.index(max_tp)]
            scaling_eff = (max_throughput / (baseline * max_tp)) * 100
        else:
            scaling_eff = 100.0
        
        result = ScalingResult(
            system="Mini-SGLang",
            model=args.model,
            tp_sizes=tp_sizes_tested,
            throughputs=throughputs,
            scaling_efficiency=scaling_eff,
        )
        save_results([result], "scaling")


if __name__ == "__main__":
    main()
