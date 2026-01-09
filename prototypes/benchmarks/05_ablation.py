#!/usr/bin/env python3
"""
Feature Ablation Benchmark: Mini-SGLang vs SGLang
==================================================

Measures the impact of individual Mini-SGLang optimizations relative to SGLang:
- Radix Cache vs Naive Cache
- Overlap Scheduling ON/OFF
- CUDA Graph ON/OFF

Shows SGLang as the reference baseline for comparison.

Usage:
    python 05_ablation.py [--model MODEL]
"""

import argparse
import os
import subprocess
import sys
import json
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    ThroughputResult, save_results, check_gpu_available,
    check_sglang_installed, get_sglang_version,
)


def run_sglang_baseline(
    model: str,
    num_requests: int,
    input_len: int,
    output_len: int,
) -> Optional[ThroughputResult]:
    """Run SGLang baseline in a subprocess."""
    script = f'''
import json
import time
from random import randint, seed

import sglang as sgl

model = "{model}"
num_requests = {num_requests}
input_len = {input_len}
output_len = {output_len}

# Create SGLang engine
llm = sgl.Engine(model_path=model, tp_size=1)

# Generate test data
seed(42)
prompts = [" ".join(str(randint(0, 9999)) for _ in range(input_len // 2)) for _ in range(num_requests)]
sampling_params = sgl.SamplingParams(temperature=0.6, max_new_tokens=output_len, ignore_eos=True)

# Warmup
llm.generate(["warmup"], sgl.SamplingParams(max_new_tokens=10))

# Benchmark
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

total_tokens = sum(len(o["meta_info"]["completion_tokens"]) if "meta_info" in o else output_len for o in outputs)
if total_tokens == 0:
    total_tokens = num_requests * output_len

llm.shutdown()

result = {{
    "tokens_per_sec": total_tokens / elapsed,
    "total_time_s": elapsed,
    "total_tokens": total_tokens,
}}
print("RESULT:" + json.dumps(result))
'''
    
    print("  Testing SGLang baseline...")
    
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
                return ThroughputResult(
                    system="SGLang (baseline)",
                    model=model,
                    batch_size=num_requests,
                    input_length=input_len,
                    output_length=output_len,
                    total_tokens=data["total_tokens"],
                    total_time_s=data["total_time_s"],
                    tokens_per_sec=data["tokens_per_sec"],
                    requests_per_sec=num_requests / data["total_time_s"],
                )
        
        if result.returncode != 0:
            print(f"    SGLang error: {result.stderr[-300:] if result.stderr else 'Unknown'}")
        return None
        
    except subprocess.TimeoutExpired:
        print("    SGLang timed out")
        return None
    except Exception as e:
        print(f"    SGLang error: {e}")
        return None


def run_minisgl_config(
    model: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    cache_type: str,
    disable_overlap: bool,
    cuda_graph_max_bs: int,
) -> Optional[ThroughputResult]:
    """Run a Mini-SGLang configuration in a subprocess."""
    
    config_name = f"Mini-SGLang (cache={cache_type}, overlap={'off' if disable_overlap else 'on'}, graph={cuda_graph_max_bs})"
    
    env_settings = ""
    if disable_overlap:
        env_settings = 'import os; os.environ["MINISGL_DISABLE_OVERLAP_SCHEDULING"] = "1"'

    script = f'''
import sys
import json
import time
from random import randint, seed

{env_settings}

import minisgl.distributed.info as info_module
info_module._TP_INFO = None

from minisgl.core import SamplingParams
from minisgl.llm import LLM

model = "{model}"
num_requests = {num_requests}
input_len = {input_len}
output_len = {output_len}

llm = LLM(
    model,
    max_seq_len_override=2048,
    cuda_graph_max_bs={cuda_graph_max_bs},
    cache_type="{cache_type}",
)

seed(42)
prompts = [[randint(0, 10000) for _ in range(input_len)] for _ in range(num_requests)]
params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
          for _ in range(num_requests)]

llm.generate(["warmup"], SamplingParams(max_tokens=10))

start = time.time()
llm.generate(prompts, params)
elapsed = time.time() - start

total_tokens = num_requests * output_len

result = {{
    "tokens_per_sec": total_tokens / elapsed,
    "total_time_s": elapsed,
    "total_tokens": total_tokens,
}}
print("RESULT:" + json.dumps(result))
'''
    
    print(f"  Testing {config_name}...")
    
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
                return ThroughputResult(
                    system=config_name,
                    model=model,
                    batch_size=num_requests,
                    input_length=input_len,
                    output_length=output_len,
                    total_tokens=data["total_tokens"],
                    total_time_s=data["total_time_s"],
                    tokens_per_sec=data["tokens_per_sec"],
                    requests_per_sec=num_requests / data["total_time_s"],
                )
        
        if result.returncode != 0:
            print(f"    Error: {result.stderr[-300:] if result.stderr else 'Unknown'}")
        return None
        
    except subprocess.TimeoutExpired:
        print("    Timed out")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def print_ablation_comparison(results: List[ThroughputResult]) -> None:
    """Print ablation comparison with SGLang baseline."""
    print("\n" + "=" * 80)
    print("Ablation Results vs SGLang Baseline")
    print("=" * 80)
    
    if not results:
        print("No results to display")
        return
    
    # Find SGLang baseline
    sglang_result = next((r for r in results if "SGLang" in r.system and "baseline" in r.system), None)
    baseline_tps = sglang_result.tokens_per_sec if sglang_result else results[0].tokens_per_sec
    
    print(f"\n{'Configuration':<60} {'tok/s':>10} {'vs SGLang':>12}")
    print("-" * 84)
    
    for r in results:
        diff = ((r.tokens_per_sec / baseline_tps) - 1) * 100 if baseline_tps > 0 else 0
        sign = "+" if diff >= 0 else ""
        
        if "SGLang" in r.system and "baseline" in r.system:
            diff_str = "baseline"
        else:
            diff_str = f"{sign}{diff:.1f}%"
        
        print(f"{r.system:<60} {r.tokens_per_sec:>10.2f} {diff_str:>12}")


def main():
    parser = argparse.ArgumentParser(description="Feature ablation: Mini-SGLang vs SGLang")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model to benchmark")
    parser.add_argument("--num-requests", type=int, default=32, help="Number of requests")
    parser.add_argument("--input-len", type=int, default=256, help="Input length")
    parser.add_argument("--output-len", type=int, default=128, help="Output length")
    parser.add_argument("--skip-sglang", action="store_true", help="Skip SGLang baseline")
    args = parser.parse_args()

    if not check_gpu_available():
        print("ERROR: No GPU available.")
        return

    sglang_available = check_sglang_installed()
    
    print("=" * 80)
    print("Feature Ablation Benchmark: Mini-SGLang vs SGLang")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Requests: {args.num_requests}")
    print(f"Input/Output: {args.input_len}/{args.output_len}")
    print(f"SGLang installed: {sglang_available}")
    print()

    results: List[ThroughputResult] = []
    
    # Run SGLang baseline first (if available)
    if sglang_available and not args.skip_sglang:
        print("Running SGLang baseline:")
        sglang_result = run_sglang_baseline(
            args.model, args.num_requests, args.input_len, args.output_len
        )
        if sglang_result:
            results.append(sglang_result)
            print(f"    Throughput: {sglang_result.tokens_per_sec:.2f} tok/s")
        else:
            print("    FAILED (continuing without SGLang baseline)")
    
    # Mini-SGLang configurations to test
    configs = [
        # Full optimizations (closest to SGLang)
        {"cache_type": "radix", "disable_overlap": False, "cuda_graph_max_bs": 256, "name": "all optimizations"},
        # Without Radix Cache
        {"cache_type": "naive", "disable_overlap": False, "cuda_graph_max_bs": 256, "name": "no radix cache"},
        # Without Overlap Scheduling
        {"cache_type": "radix", "disable_overlap": True, "cuda_graph_max_bs": 256, "name": "no overlap"},
        # Without CUDA Graphs
        {"cache_type": "radix", "disable_overlap": False, "cuda_graph_max_bs": 0, "name": "no cuda graph"},
        # Minimal (no optimizations)
        {"cache_type": "naive", "disable_overlap": True, "cuda_graph_max_bs": 0, "name": "minimal"},
    ]
    
    print(f"\nRunning Mini-SGLang configurations ({len(configs)} tests):")
    
    for i, config in enumerate(configs):
        print(f"\nTest {i+1}/{len(configs)}: {config['name']}")
        result = run_minisgl_config(
            model=args.model,
            num_requests=args.num_requests,
            input_len=args.input_len,
            output_len=args.output_len,
            cache_type=config["cache_type"],
            disable_overlap=config["disable_overlap"],
            cuda_graph_max_bs=config["cuda_graph_max_bs"],
        )
        if result:
            results.append(result)
            print(f"    Throughput: {result.tokens_per_sec:.2f} tok/s")
        else:
            print("    FAILED")
    
    # Print results
    if results:
        print_ablation_comparison(results)
        save_results(results, "ablation_comparison")


if __name__ == "__main__":
    main()
