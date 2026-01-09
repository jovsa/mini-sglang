#!/usr/bin/env python3
"""
Learning Prototype 1: Core Data Structures
==========================================

This prototype demonstrates the REAL Mini-SGLang core data structures:
- SamplingParams: Controls how tokens are sampled
- Req: Represents a single inference request
- Batch: Groups multiple requests for efficient processing

Uses actual minisgl.core code to show how the system really works.

Run: python prototypes/learning/01_core_data_structures.py
"""

import torch
from dataclasses import dataclass

# Import the REAL Mini-SGLang components
from minisgl.core import SamplingParams, Req, Batch


# We need a minimal cache handle for Req (required field)
@dataclass(frozen=True)
class DummyCacheHandle:
    """Minimal cache handle for demo purposes."""
    cached_len: int = 0


def demo_sampling_params():
    """Demonstrate SamplingParams from minisgl.core"""
    print("=" * 60)
    print("1. SamplingParams (from minisgl.core)")
    print("=" * 60)

    # Greedy sampling (deterministic)
    greedy = SamplingParams(temperature=0.0, max_tokens=100)
    print(f"\nGreedy sampling:")
    print(f"  temperature={greedy.temperature}, top_k={greedy.top_k}, top_p={greedy.top_p}")
    print(f"  is_greedy={greedy.is_greedy}  # temperature <= 0 means greedy")

    # Random sampling
    random = SamplingParams(temperature=0.8, top_k=50, top_p=0.9, max_tokens=200)
    print(f"\nRandom sampling:")
    print(f"  temperature={random.temperature}, top_k={random.top_k}, top_p={random.top_p}")
    print(f"  is_greedy={random.is_greedy}")

    # With ignore_eos (for benchmarking)
    benchmark = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=128)
    print(f"\nBenchmark mode (ignore_eos=True):")
    print(f"  ignore_eos={benchmark.ignore_eos}  # Won't stop at EOS token")


def demo_req_lifecycle():
    """Demonstrate Req lifecycle from minisgl.core"""
    print("\n" + "=" * 60)
    print("2. Req Lifecycle (from minisgl.core)")
    print("=" * 60)

    # Create input tokens as CPU tensor (required by Req)
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)  # "Hello" tokenized

    # Create a request
    params = SamplingParams(temperature=0.0, max_tokens=5)
    req = Req(
        input_ids=input_ids,
        table_idx=0,           # Index in page table
        cached_len=0,          # Nothing cached initially
        output_len=params.max_tokens,
        uid=42,                # Unique request ID
        sampling_params=params,
        cache_handle=DummyCacheHandle(0),
    )

    print(f"\nInitial state:")
    print(f"  input_ids: {req.input_ids.tolist()}")
    print(f"  cached_len: {req.cached_len} (tokens already in KV cache)")
    print(f"  device_len: {req.device_len} (total tokens on device)")
    print(f"  max_device_len: {req.max_device_len} (input + max_tokens)")
    print(f"  extend_len: {req.extend_len} (tokens to compute: device_len - cached_len)")
    print(f"  remain_len: {req.remain_len} (tokens left to generate)")
    print(f"  can_decode: {req.can_decode()}")

    # Simulate prefill completion
    print(f"\nAfter PREFILL (complete_one + append_host):")
    req.complete_one()  # Mark prefill as done
    next_token = torch.tensor([100], dtype=torch.int32)
    req.append_host(next_token)  # Add generated token

    print(f"  cached_len: {req.cached_len} (now all input is cached)")
    print(f"  device_len: {req.device_len} (grew by 1)")
    print(f"  extend_len: {req.extend_len} (only 1 new token to compute)")
    print(f"  remain_len: {req.remain_len}")
    print(f"  input_ids: {req.input_ids.tolist()}")

    # Simulate decode steps
    print(f"\nDECODE steps:")
    step = 1
    while req.can_decode():
        req.complete_one()
        next_token = torch.tensor([100 + step], dtype=torch.int32)
        req.append_host(next_token)
        print(f"  Step {step}: generated {100+step}, remain_len={req.remain_len}")
        step += 1

    print(f"\nFinal state:")
    print(f"  input_ids: {req.input_ids.tolist()}")
    print(f"  Original: [1, 2, 3, 4, 5]")
    print(f"  Generated: {req.input_ids[5:].tolist()}")


def demo_batch():
    """Demonstrate Batch from minisgl.core"""
    print("\n" + "=" * 60)
    print("3. Batch (from minisgl.core)")
    print("=" * 60)

    # Create multiple requests
    requests = []
    for i in range(3):
        input_ids = torch.tensor([i*10 + j for j in range(3 + i)], dtype=torch.int32)
        req = Req(
            input_ids=input_ids,
            table_idx=i,
            cached_len=0,
            output_len=2,
            uid=i,
            sampling_params=SamplingParams(max_tokens=2),
            cache_handle=DummyCacheHandle(0),
        )
        requests.append(req)

    # Create prefill batch
    prefill_batch = Batch(reqs=requests, phase="prefill")

    print(f"\nPrefill batch:")
    print(f"  phase: {prefill_batch.phase}")
    print(f"  is_prefill: {prefill_batch.is_prefill}")
    print(f"  is_decode: {prefill_batch.is_decode}")
    print(f"  size: {prefill_batch.size} requests")

    total_tokens = sum(r.extend_len for r in prefill_batch.reqs)
    print(f"  total tokens to compute: {total_tokens}")
    print(f"  per request: {[r.extend_len for r in prefill_batch.reqs]}")

    # Simulate after prefill - create decode batch
    for r in requests:
        r.complete_one()
        r.append_host(torch.tensor([999], dtype=torch.int32))

    decode_batch = Batch(reqs=requests, phase="decode")

    print(f"\nDecode batch:")
    print(f"  phase: {decode_batch.phase}")
    print(f"  is_decode: {decode_batch.is_decode}")
    print(f"  size: {decode_batch.size} requests")
    print(f"  extend_len per request: {[r.extend_len for r in decode_batch.reqs]}")
    print(f"  (Each request computes only 1 new token in decode phase)")


def demo_prefix_caching_concept():
    """Show how cached_len enables prefix caching"""
    print("\n" + "=" * 60)
    print("4. Prefix Caching Concept (via cached_len)")
    print("=" * 60)

    # Shared prefix: "What is the capital of"
    shared_prefix = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)

    # First request: "What is the capital of France?" - no cache
    req1_ids = torch.cat([shared_prefix, torch.tensor([100], dtype=torch.int32)])
    req1 = Req(
        input_ids=req1_ids,
        table_idx=0,
        cached_len=0,  # Nothing cached
        output_len=3,
        uid=1,
        sampling_params=SamplingParams(max_tokens=3),
        cache_handle=DummyCacheHandle(0),
    )

    print(f"\nFirst request (no cache):")
    print(f"  input_ids: {req1.input_ids.tolist()}")
    print(f"  cached_len: {req1.cached_len}")
    print(f"  extend_len: {req1.extend_len}  # Must compute ALL tokens")

    # Second request: "What is the capital of Germany?" - prefix cached!
    req2_ids = torch.cat([shared_prefix, torch.tensor([200], dtype=torch.int32)])
    req2 = Req(
        input_ids=req2_ids,
        table_idx=1,
        cached_len=5,  # Shared prefix is cached!
        output_len=3,
        uid=2,
        sampling_params=SamplingParams(max_tokens=3),
        cache_handle=DummyCacheHandle(5),
    )

    print(f"\nSecond request (with cached prefix):")
    print(f"  input_ids: {req2.input_ids.tolist()}")
    print(f"  cached_len: {req2.cached_len}  # 5 tokens from cache!")
    print(f"  extend_len: {req2.extend_len}  # Only compute 1 new token!")

    savings = (req1.extend_len - req2.extend_len) / req1.extend_len * 100
    print(f"\n  Computation saved: {savings:.1f}%")
    print(f"  This is what RadixCache optimizes automatically!")


if __name__ == "__main__":
    print("Mini-SGLang Core Data Structures Demo")
    print("Using REAL code from minisgl.core")
    print()

    demo_sampling_params()
    demo_req_lifecycle()
    demo_batch()
    demo_prefix_caching_concept()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. SamplingParams controls token generation (temperature, top_k, top_p)
   - is_greedy property determines deterministic vs random sampling

2. Req tracks request state through its lifecycle:
   - extend_len = device_len - cached_len (tokens to compute)
   - remain_len = max_device_len - device_len (tokens left)
   - complete_one() marks a step done, append_host() adds token

3. Batch groups requests for GPU processing:
   - phase="prefill": process all input tokens (compute-bound)
   - phase="decode": generate one token per request (memory-bound)

4. Prefix caching via cached_len:
   - cached_len > 0 means KV cache is reused
   - Reduces extend_len, saving computation
""")
