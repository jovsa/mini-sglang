#!/usr/bin/env python3
"""
Online Latency Benchmark: Mini-SGLang vs SGLang
================================================

Measures latency metrics (TTFT, TPOT, E2E) using streaming API.
Both systems run as HTTP servers and are benchmarked via OpenAI-compatible API.

Usage:
    python 02_online_latency.py [--model MODEL] [--num-requests N]
"""

import argparse
import asyncio
import time
from typing import List, Optional

from common import (
    LatencyResult, LatencyMetrics, save_results, 
    print_latency_comparison, check_gpu_available,
    check_sglang_installed, get_sglang_version,
    start_minisgl_server, start_sglang_server, stop_server,
    wait_for_server, is_server_running,
)

# Port assignments (use 8010/8012 to avoid conflicts with nginx on 8001)
MINISGL_PORT = 8010
SGLANG_PORT = 8012


async def run_latency_benchmark(
    port: int,
    model: str,
    num_requests: int,
    input_length: int,
    output_length: int,
    system_name: str,
) -> Optional[LatencyResult]:
    """Run latency benchmark using streaming API."""
    from openai import AsyncOpenAI
    from random import randint, seed
    
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="dummy")
    
    # Generate prompts
    seed(42)
    prompts = []
    for _ in range(num_requests):
        prompt = " ".join(str(randint(0, 9999)) for _ in range(input_length // 2))
        prompts.append(prompt)
    
    ttft_values = []
    tpot_values = []
    e2e_values = []
    
    async def run_one_streaming(prompt: str):
        """Run a single streaming request and collect timing."""
        try:
            start = time.time()
            first_token_time = None
            last_token_time = start
            token_count = 0
            
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_length,
                temperature=0.6,
                stream=True,
            )
            
            async for chunk in stream:
                now = time.time()
                if chunk.choices and chunk.choices[0].delta.content:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        tpot_values.append(now - last_token_time)
                    last_token_time = now
            
            end = time.time()
            
            if first_token_time:
                ttft_values.append(first_token_time - start)
                e2e_values.append(end - start)
            
            return token_count
        except Exception as e:
            print(f"    Request error: {e}")
            return 0
    
    # Warmup
    try:
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )
    except Exception as e:
        print(f"    Warmup failed: {e}")
        await client.close()
        return None
    
    # Run benchmark
    print(f"    Running {num_requests} streaming requests...")
    for i, prompt in enumerate(prompts):
        await run_one_streaming(prompt)
        if (i + 1) % 10 == 0:
            print(f"      Progress: {i + 1}/{num_requests}")
    
    await client.close()
    
    if not ttft_values:
        return None
    
    return LatencyResult(
        system=system_name,
        model=model,
        num_requests=num_requests,
        ttft=LatencyMetrics.from_list(ttft_values),  # ms
        tpot=LatencyMetrics.from_list(tpot_values),  # ms
        e2e=LatencyMetrics.from_list(e2e_values, scale=1.0),  # seconds
    )


async def run_latency_tests(
    model: str,
    num_requests: int,
    input_length: int,
    output_length: int,
    systems: List[str],
) -> List[LatencyResult]:
    """Run latency tests against specified systems."""
    results = []
    
    for system in systems:
        if system == "Mini-SGLang":
            port = MINISGL_PORT
        elif system == "SGLang":
            port = SGLANG_PORT
        else:
            continue
        
        print(f"  Testing {system}...")
        result = await run_latency_benchmark(
            port=port,
            model=model,
            num_requests=num_requests,
            input_length=input_length,
            output_length=output_length,
            system_name=system,
        )
        
        if result:
            results.append(result)
            print(f"    TTFT: {result.ttft.avg:.2f}ms, TPOT: {result.tpot.avg:.2f}ms")
        else:
            print(f"    FAILED")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark: Mini-SGLang vs SGLang")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model to benchmark")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of requests")
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[256],
                       help="Input sequence lengths to test")
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[64],
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
    print("Latency Benchmark: Mini-SGLang vs SGLang")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Requests per test: {args.num_requests}")
    print(f"Input lengths: {args.input_lengths}")
    print(f"Output lengths: {args.output_lengths}")
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
        all_results: List[LatencyResult] = []
        
        for input_len in args.input_lengths:
            for output_len in args.output_lengths:
                print(f"\n--- Test: input={input_len}, output={output_len} ---")
                
                results = asyncio.run(run_latency_tests(
                    model=args.model,
                    num_requests=args.num_requests,
                    input_length=input_len,
                    output_length=output_len,
                    systems=systems,
                ))
                all_results.extend(results)
        
        # Print comparison
        if all_results:
            print_latency_comparison(all_results, baseline_system="SGLang")
            save_results(all_results, "latency_comparison")
        
    finally:
        # Stop servers
        print("\nStopping servers...")
        for name, proc in procs.items():
            print(f"  Stopping {name}...")
            stop_server(proc)
        print("Done.")


if __name__ == "__main__":
    main()
