#!/usr/bin/env python3
"""
Common utilities for benchmarking Mini-SGLang and SGLang.
"""

import json
import time
import subprocess
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model: str = "Qwen/Qwen3-0.6B"
    tp_size: int = 1
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 128, 256])
    input_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024])
    output_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])
    num_requests: int = 100
    warmup_requests: int = 5
    port: int = 8000


@dataclass  
class LatencyMetrics:
    """Latency statistics in milliseconds."""
    avg: float
    p50: float
    p90: float
    p99: float
    max: float
    
    @classmethod
    def from_list(cls, values: List[float], scale: float = 1000.0) -> "LatencyMetrics":
        """Compute metrics from a list of values (scale to ms by default)."""
        if not values:
            return cls(0, 0, 0, 0, 0)
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return cls(
            avg=scale * sum(sorted_vals) / n,
            p50=scale * sorted_vals[int(n * 0.5)],
            p90=scale * sorted_vals[int(n * 0.9)],
            p99=scale * sorted_vals[int(n * 0.99)] if n > 100 else scale * sorted_vals[-1],
            max=scale * sorted_vals[-1]
        )


@dataclass
class ThroughputResult:
    """Results from a throughput benchmark."""
    system: str
    model: str
    batch_size: int
    input_length: int
    output_length: int
    total_tokens: int
    total_time_s: float
    tokens_per_sec: float
    requests_per_sec: float


@dataclass
class LatencyResult:
    """Results from a latency benchmark."""
    system: str
    model: str
    num_requests: int
    ttft: LatencyMetrics  # Time to First Token
    tpot: LatencyMetrics  # Time Per Output Token
    e2e: LatencyMetrics   # End-to-End latency


@dataclass
class MemoryResult:
    """Memory usage metrics."""
    system: str
    model: str
    model_memory_gb: float
    kv_cache_gb: float
    peak_memory_gb: float
    total_gpu_memory_gb: float
    utilization_pct: float


@dataclass
class ScalingResult:
    """Multi-GPU scaling results."""
    system: str
    model: str
    tp_sizes: List[int]
    throughputs: List[float]  # tokens/sec at each TP size
    scaling_efficiency: float  # vs ideal linear scaling


def save_results(results: List[Any], name: str) -> Path:
    """Save results to JSON and CSV files."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_path = RESULTS_DIR / f"{name}_{timestamp}"
    
    # Save JSON
    json_path = base_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in results], f, indent=2)
    
    # Save CSV
    csv_path = base_path.with_suffix(".csv")
    if results and hasattr(results[0], "__dataclass_fields__"):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].__dataclass_fields__.keys())
            writer.writeheader()
            for r in results:
                row = asdict(r)
                # Flatten nested dataclasses
                flat_row = {}
                for k, v in row.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            flat_row[f"{k}_{k2}"] = v2
                    else:
                        flat_row[k] = v
                writer.writerow(flat_row)
    
    print(f"Results saved to: {json_path}")
    return json_path


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory info using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Sum across all GPUs
            total, used, free = 0, 0, 0
            for line in lines:
                t, u, f = map(float, line.split(","))
                total += t
                used += u
                free += f
            return {
                "total_gb": total / 1024,
                "used_gb": used / 1024,
                "free_gb": free / 1024,
            }
    except Exception as e:
        print(f"Warning: Could not get GPU memory info: {e}")
    return {"total_gb": 0, "used_gb": 0, "free_gb": 0}


def is_server_running(port: int) -> bool:
    """Check if a server is running on the given port."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        if is_server_running(port):
            # Give it a moment to fully initialize
            time.sleep(2)
            return True
        time.sleep(1)
    return False


def start_minisgl_server(model: str, port: int, tp_size: int = 1, extra_args: List[str] = None) -> subprocess.Popen:
    """Start a Mini-SGLang server."""
    cmd = [
        sys.executable, "-m", "minisgl",
        "--model", model,
        "--port", str(port),
        "--tp", str(tp_size),
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Starting Mini-SGLang server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def start_sglang_server(model: str, port: int, tp_size: int = 1, extra_args: List[str] = None) -> subprocess.Popen:
    """Start an SGLang server."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model", model,
        "--port", str(port),
        "--tp", str(tp_size),
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Starting SGLang server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """Stop a server process."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def check_sglang_installed() -> bool:
    """Check if SGLang is installed."""
    try:
        import sglang
        return True
    except ImportError:
        return False


def get_sglang_version() -> Optional[str]:
    """Get the installed SGLang version."""
    try:
        import sglang
        return getattr(sglang, "__version__", "unknown")
    except ImportError:
        return None


async def run_server_throughput_benchmark(
    base_url: str,
    model: str,
    num_requests: int,
    input_length: int,
    output_length: int,
    system_name: str = "Unknown",
) -> Optional[ThroughputResult]:
    """Run throughput benchmark against any OpenAI-compatible server."""
    from openai import AsyncOpenAI
    import asyncio
    from random import randint, seed
    
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="dummy")
    
    # Generate random prompts (same seed for reproducibility)
    seed(42)
    prompts = []
    for _ in range(num_requests):
        # Create a prompt of approximately input_length tokens
        # Using random numbers as a simple way to control length
        prompt = " ".join(str(randint(0, 9999)) for _ in range(input_length // 2))
        prompts.append(prompt)
    
    # Warmup
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
    except Exception as e:
        print(f"  Warmup failed: {e}")
        await client.close()
        return None
    
    # Run benchmark
    start_time = time.time()
    
    async def run_one(prompt: str):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_length,
                temperature=0.6,
            )
            return response.usage.completion_tokens if response.usage else output_length
        except Exception as e:
            return 0
    
    tasks = [run_one(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    total_tokens = sum(results)
    
    await client.close()
    
    if total_tokens == 0:
        return None
    
    return ThroughputResult(
        system=system_name,
        model=model,
        batch_size=num_requests,
        input_length=input_length,
        output_length=output_length,
        total_tokens=total_tokens,
        total_time_s=elapsed,
        tokens_per_sec=total_tokens / elapsed,
        requests_per_sec=num_requests / elapsed,
    )


def print_throughput_comparison(results: List[ThroughputResult], baseline_system: str = "SGLang") -> None:
    """Print throughput comparison with baseline."""
    print("\n" + "=" * 70)
    print("Throughput Comparison")
    print("=" * 70)
    
    if not results:
        print("No results to display")
        return
    
    # Find baseline
    baseline = next((r for r in results if r.system == baseline_system), None)
    baseline_tps = baseline.tokens_per_sec if baseline else results[0].tokens_per_sec
    
    print(f"\n{'System':<20} {'tok/s':>12} {'req/s':>10} {'vs baseline':>12}")
    print("-" * 56)
    
    for r in results:
        diff_pct = ((r.tokens_per_sec / baseline_tps) - 1) * 100 if baseline_tps > 0 else 0
        sign = "+" if diff_pct >= 0 else ""
        baseline_str = "baseline" if r.system == baseline_system else f"{sign}{diff_pct:.1f}%"
        print(f"{r.system:<20} {r.tokens_per_sec:>12.2f} {r.requests_per_sec:>10.2f} {baseline_str:>12}")


def print_latency_comparison(results: List[LatencyResult], baseline_system: str = "SGLang") -> None:
    """Print latency comparison with baseline."""
    print("\n" + "=" * 70)
    print("Latency Comparison")
    print("=" * 70)
    
    if not results:
        print("No results to display")
        return
    
    # Find baseline
    baseline = next((r for r in results if r.system == baseline_system), None)
    
    print(f"\n{'System':<20} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'E2E(s)':>10} {'vs baseline':>12}")
    print("-" * 64)
    
    for r in results:
        if baseline and baseline.ttft.avg > 0:
            diff_pct = ((r.ttft.avg / baseline.ttft.avg) - 1) * 100
            sign = "+" if diff_pct >= 0 else ""
            baseline_str = "baseline" if r.system == baseline_system else f"{sign}{diff_pct:.1f}%"
        else:
            baseline_str = "baseline" if r == results[0] else "N/A"
        print(f"{r.system:<20} {r.ttft.avg:>10.2f} {r.tpot.avg:>10.2f} {r.e2e.avg:>10.2f} {baseline_str:>12}")


def print_comparison_table(results: List[Any], title: str) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    
    if not results:
        print("No results to display")
        return
    
    # Get fields from first result
    if hasattr(results[0], "__dataclass_fields__"):
        fields = list(results[0].__dataclass_fields__.keys())
    else:
        fields = list(results[0].keys())
    
    # Print header
    header = " | ".join(f"{f:>12}" for f in fields)
    print(header)
    print("-" * len(header))
    
    # Print rows
    for r in results:
        if hasattr(r, "__dataclass_fields__"):
            values = [getattr(r, f) for f in fields]
        else:
            values = [r[f] for f in fields]
        
        formatted = []
        for v in values:
            if isinstance(v, float):
                formatted.append(f"{v:>12.2f}")
            elif isinstance(v, LatencyMetrics):
                formatted.append(f"{v.avg:>12.2f}")
            else:
                formatted.append(f"{str(v):>12}")
        print(" | ".join(formatted))


if __name__ == "__main__":
    # Quick test
    print("Benchmark common utilities loaded.")
    print(f"GPU available: {check_gpu_available()}")
    print(f"GPU memory: {get_gpu_memory_info()}")
    print(f"Results directory: {RESULTS_DIR}")
