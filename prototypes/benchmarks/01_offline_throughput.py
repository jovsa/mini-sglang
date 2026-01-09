#!/usr/bin/env python3
"""
Offline Throughput Benchmark: Mini-SGLang vs SGLang
====================================================

Measures generation throughput (tokens/sec) for both Mini-SGLang and SGLang.
Both systems run as HTTP servers and are benchmarked via OpenAI-compatible API.

Usage:
    python 01_offline_throughput.py [--model MODEL] [--num-requests N]
"""

import argparse
import asyncio
import time
from typing import List, Optional

from common import (
    ThroughputResult, save_results,
    print_throughput_comparison, check_gpu_available, get_gpu_memory_info,
    check_sglang_installed, get_sglang_version,
    start_minisgl_server, start_sglang_server, stop_server,
    wait_for_server, is_server_running,
    run_server_throughput_benchmark,
)

# Port assignments (use 8010/8012 to avoid conflicts with nginx on 8001)
MINISGL_PORT = 8010
SGLANG_PORT = 8012


async def run_throughput_test(
    model: str,
    num_requests: int,
    input_length: int,
    output_length: int,
    systems: List[str],
) -> List[ThroughputResult]:
    """Run throughput test against specified systems."""
    results = []

    for system in systems:
        if system == "Mini-SGLang":
            port = MINISGL_PORT
        elif system == "SGLang":
            port = SGLANG_PORT
        else:
            continue

        print(f"  Testing {system}...")
        result = await run_server_throughput_benchmark(
            base_url=f"http://127.0.0.1:{port}",
            model=model,
            num_requests=num_requests,
            input_length=input_length,
            output_length=output_length,
            system_name=system,
        )

        if result:
            results.append(result)
            print(f"    {result.tokens_per_sec:.2f} tok/s")
        else:
            print(f"    FAILED")

    return results


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark: Mini-SGLang vs SGLang")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model to benchmark")
    parser.add_argument("--num-requests", type=int, default=32, help="Number of requests")
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[256],
                       help="Input sequence lengths to test")
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[128],
                       help="Output sequence lengths to test")
    parser.add_argument("--minisgl-only", action="store_true", help="Only test Mini-SGLang")
    parser.add_argument("--sglang-only", action="store_true", help="Only test SGLang")
    args = parser.parse_args()

    if not check_gpu_available():
        print("ERROR: No GPU available. This benchmark requires a CUDA GPU.")
        return

    # Check SGLang installation
    sglang_available = check_sglang_installed()
    sglang_version = get_sglang_version() if sglang_available else None

    print("=" * 70)
    print("Throughput Benchmark: Mini-SGLang vs SGLang")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Requests per test: {args.num_requests}")
    print(f"Input lengths: {args.input_lengths}")
    print(f"Output lengths: {args.output_lengths}")
    print(f"GPU Memory: {get_gpu_memory_info()}")
    print(f"SGLang installed: {sglang_available} (v{sglang_version})" if sglang_available else "SGLang: not installed")
    print()

    # Determine which systems to test
    systems = []
    if not args.sglang_only:
        systems.append("Mini-SGLang")
    if not args.minisgl_only and sglang_available:
        systems.append("SGLang")
    elif args.sglang_only and not sglang_available:
        print("ERROR: --sglang-only specified but SGLang is not installed")
        return

    if not systems:
        print("ERROR: No systems to test")
        return

    print(f"Testing systems: {', '.join(systems)}")

    # Start servers
    procs = {}
    try:
        if "Mini-SGLang" in systems:
            print(f"\nStarting Mini-SGLang server on port {MINISGL_PORT}...")
            procs["Mini-SGLang"] = start_minisgl_server(args.model, MINISGL_PORT)
            if not wait_for_server(MINISGL_PORT, timeout=300):
                print("ERROR: Mini-SGLang server failed to start")
                return
            print("Mini-SGLang server ready!")

        if "SGLang" in systems:
            print(f"\nStarting SGLang server on port {SGLANG_PORT}...")
            procs["SGLang"] = start_sglang_server(args.model, SGLANG_PORT)
            if not wait_for_server(SGLANG_PORT, timeout=300):
                print("ERROR: SGLang server failed to start")
                return
            print("SGLang server ready!")

        # Run benchmarks
        all_results: List[ThroughputResult] = []

        for input_len in args.input_lengths:
            for output_len in args.output_lengths:
                print(f"\n--- Test: input={input_len}, output={output_len} ---")

                results = asyncio.run(run_throughput_test(
                    model=args.model,
                    num_requests=args.num_requests,
                    input_length=input_len,
                    output_length=output_len,
                    systems=systems,
                ))
                all_results.extend(results)

        # Print comparison
        if all_results:
            print_throughput_comparison(all_results, baseline_system="SGLang")
            save_results(all_results, "throughput_comparison")

    finally:
        # Stop servers
        print("\nStopping servers...")
        for name, proc in procs.items():
            print(f"  Stopping {name}...")
            stop_server(proc)
        print("Done.")


if __name__ == "__main__":
    main()
