#!/usr/bin/env python3
"""
Memory Profiling Benchmark: Mini-SGLang vs SGLang
==================================================

Profiles GPU memory usage during model loading and inference for both systems.

Usage:
    python 03_memory_profile.py [--model MODEL] [--full-profile]
"""

import argparse
import gc
import subprocess
import sys
import json
import time
from typing import Dict, List, Tuple, Optional
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    MemoryResult, save_results, check_gpu_available, get_gpu_memory_info,
    check_sglang_installed, get_sglang_version,
)


def get_torch_memory_stats(device: int = 0) -> Dict[str, float]:
    """Get PyTorch CUDA memory statistics."""
    import torch

    if not torch.cuda.is_available():
        return {}

    torch.cuda.synchronize(device)

    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
    }


def estimate_model_memory(model_path: str) -> Tuple[float, Dict]:
    """Estimate model memory from config (shared for both systems)."""
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_path)
    
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
    
    # Rough estimate: embed + layers + lm_head
    params = (
        vocab_size * hidden_size +  # embed
        num_layers * (
            4 * hidden_size * hidden_size +  # attention
            3 * hidden_size * intermediate_size +  # mlp
            4 * hidden_size  # norms
        ) +
        vocab_size * hidden_size  # lm_head
    )
    
    dtype_bytes = 2  # bfloat16
    estimated_gb = params * dtype_bytes / (1024**3)
    
    return estimated_gb, {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "params_b": params / 1e9,
    }


def profile_minisgl_memory(model_path: str, batch_size: int, seq_len: int) -> Optional[Dict]:
    """Profile Mini-SGLang memory usage in a subprocess."""
    script = f'''
import json
import gc
import torch
import minisgl.distributed.info as info_module
info_module._TP_INFO = None

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

from minisgl.core import SamplingParams
from minisgl.llm import LLM
from random import randint, seed

before = torch.cuda.memory_allocated() / (1024**3)

llm = LLM("{model_path}", max_seq_len_override=2048, cuda_graph_max_bs=64)

after_load = torch.cuda.memory_allocated() / (1024**3)

# Run inference
seed(42)
prompts = [[randint(0, 10000) for _ in range({seq_len})] for _ in range({batch_size})]
params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=64)
llm.generate(prompts, [params] * {batch_size})

torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / (1024**3)

result = {{
    "model_memory_gb": after_load - before,
    "peak_memory_gb": peak,
}}
print("RESULT:" + json.dumps(result))
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
                return json.loads(line[7:])
        
        if result.returncode != 0:
            print(f"  Mini-SGLang error: {result.stderr[-300:] if result.stderr else 'Unknown'}")
        return None
        
    except subprocess.TimeoutExpired:
        print("  Mini-SGLang timed out")
        return None
    except Exception as e:
        print(f"  Mini-SGLang error: {e}")
        return None


def profile_sglang_memory(model_path: str, batch_size: int, seq_len: int) -> Optional[Dict]:
    """Profile SGLang memory usage in a subprocess."""
    script = f'''
import json
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import sglang as sgl
from random import randint, seed

before = torch.cuda.memory_allocated() / (1024**3)

llm = sgl.Engine(model_path="{model_path}", tp_size=1)

after_load = torch.cuda.memory_allocated() / (1024**3)

# Run inference
seed(42)
prompts = [" ".join(str(randint(0, 9999)) for _ in range({seq_len} // 2)) for _ in range({batch_size})]

sampling_params = sgl.SamplingParams(temperature=0.6, max_new_tokens=64)
outputs = llm.generate(prompts, sampling_params)

torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / (1024**3)

llm.shutdown()

result = {{
    "model_memory_gb": after_load - before,
    "peak_memory_gb": peak,
}}
print("RESULT:" + json.dumps(result))
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
                return json.loads(line[7:])
        
        if result.returncode != 0:
            print(f"  SGLang error: {result.stderr[-300:] if result.stderr else 'Unknown'}")
        return None
        
    except subprocess.TimeoutExpired:
        print("  SGLang timed out")
        return None
    except Exception as e:
        print(f"  SGLang error: {e}")
        return None


def print_memory_comparison(results: List[MemoryResult]) -> None:
    """Print memory comparison table."""
    print("\n" + "=" * 70)
    print("Memory Comparison")
    print("=" * 70)
    
    if not results:
        print("No results to display")
        return
    
    baseline = next((r for r in results if r.system == "SGLang"), None)
    
    print(f"\n{'System':<15} {'Model(GB)':>10} {'Peak(GB)':>10} {'Util%':>8} {'vs SGLang':>12}")
    print("-" * 57)
    
    for r in results:
        if baseline and baseline.peak_memory_gb > 0:
            diff_pct = ((r.peak_memory_gb / baseline.peak_memory_gb) - 1) * 100
            sign = "+" if diff_pct >= 0 else ""
            baseline_str = "baseline" if r.system == "SGLang" else f"{sign}{diff_pct:.1f}%"
        else:
            baseline_str = "baseline" if r == results[0] else "N/A"
        
        print(f"{r.system:<15} {r.model_memory_gb:>10.2f} {r.peak_memory_gb:>10.2f} {r.utilization_pct:>7.1f}% {baseline_str:>12}")


def main():
    parser = argparse.ArgumentParser(description="Memory profiling: Mini-SGLang vs SGLang")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model to profile")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length for inference")
    parser.add_argument("--full-profile", action="store_true", help="Run full inference profile")
    parser.add_argument("--minisgl-only", action="store_true", help="Only test Mini-SGLang")
    parser.add_argument("--sglang-only", action="store_true", help="Only test SGLang")
    args = parser.parse_args()

    if not check_gpu_available():
        print("ERROR: No GPU available.")
        return

    sglang_available = check_sglang_installed()
    
    print("=" * 70)
    print("Memory Profiling: Mini-SGLang vs SGLang")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    
    gpu_info = get_gpu_memory_info()
    print(f"GPU Memory: {gpu_info['total_gb']:.2f} GB total, {gpu_info['free_gb']:.2f} GB free")
    print(f"SGLang installed: {sglang_available}")
    print()
    
    # Determine which systems to test
    systems = []
    if not args.sglang_only:
        systems.append("Mini-SGLang")
    if not args.minisgl_only and sglang_available:
        systems.append("SGLang")
    
    # Estimate model memory (shared calculation)
    print("1. Model Memory Estimation:")
    estimated_gb, model_info = estimate_model_memory(args.model)
    print(f"  Config: {model_info['num_layers']} layers, hidden={model_info['hidden_size']}")
    print(f"  Parameters: {model_info['params_b']:.2f}B")
    print(f"  Estimated memory (bf16): {estimated_gb:.2f} GB")
    
    if not args.full_profile:
        print("\n  (Use --full-profile to run actual inference profiling)")
        return
    
    print("\n2. Actual Memory Profiling:")
    results: List[MemoryResult] = []
    
    for system in systems:
        print(f"\n  Profiling {system}...")
        
        if system == "Mini-SGLang":
            profile = profile_minisgl_memory(args.model, args.batch_size, args.seq_len)
        else:
            profile = profile_sglang_memory(args.model, args.batch_size, args.seq_len)
        
        if profile:
            result = MemoryResult(
                system=system,
                model=args.model,
                model_memory_gb=profile["model_memory_gb"],
                kv_cache_gb=0,  # Not measured separately
                peak_memory_gb=profile["peak_memory_gb"],
                total_gpu_memory_gb=gpu_info["total_gb"],
                utilization_pct=profile["peak_memory_gb"] / gpu_info["total_gb"] * 100,
            )
            results.append(result)
            print(f"    Model: {result.model_memory_gb:.2f} GB")
            print(f"    Peak: {result.peak_memory_gb:.2f} GB")
            print(f"    Utilization: {result.utilization_pct:.1f}%")
        else:
            print(f"    FAILED")
    
    if results:
        print_memory_comparison(results)
        save_results(results, "memory_comparison")


if __name__ == "__main__":
    main()
