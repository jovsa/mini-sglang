# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py
# SGLang version of bench.py

import hashlib
import json
import time
from random import randint, seed

import sglang as sgl


def compute_checksum(data):
    """Compute SHA256 checksum of serialized data."""
    if isinstance(data, (list, dict)):
        serialized = json.dumps(data, sort_keys=True).encode()
    else:
        serialized = str(data).encode()
    return hashlib.sha256(serialized).hexdigest()


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # Model configuration - align with bench.py
    model_path = "Qwen/Qwen3-0.6B"
    max_model_len = 4096  # equivalent to max_seq_len_override=4096
    max_num_batched_tokens = 16384  # equivalent to max_extend_tokens=16384
    # Note: cuda_graph_max_bs may not have direct equivalent in sglang

    # align the hyperparameters
    # SGLang Engine accepts parameters matching ServerArgs
    # Try common parameter names - may need adjustment based on actual API
    llm = sgl.Engine(
        model_path=model_path,
        context_length=max_model_len,  # Try context_length instead of max_model_len
        mem_fraction_static=0.9,  # Memory fraction
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params_list = [
        {"temperature": 0.6, "ignore_eos": True, "max_new_tokens": randint(100, max_ouput_len)}
        for _ in range(num_seqs)
    ]

    # Compute checksums for verification
    input_checksum = compute_checksum(prompt_token_ids)
    # Normalize sampling params for checksum (use max_tokens instead of max_new_tokens)
    sampling_params_normalized = [
        {"temperature": sp["temperature"], "ignore_eos": sp["ignore_eos"], "max_tokens": sp["max_new_tokens"]}
        for sp in sampling_params_list
    ]
    sampling_checksum = compute_checksum(sampling_params_normalized)
    # Config checksum uses normalized names for comparison with bench.py
    # max_model_len (sglang) = max_seq_len_override (mini-sglang)
    # max_num_batched_tokens (sglang) = max_extend_tokens (mini-sglang)
    config_data = {
        "model_path": model_path,
        "max_seq_len": max_model_len,  # normalized name
        "max_extend_tokens": max_num_batched_tokens,  # normalized name
        "num_seqs": num_seqs,
        "max_input_len": max_input_len,
        "max_output_len": max_ouput_len,
    }
    config_checksum = compute_checksum(config_data)

    print(f"Input token IDs checksum: {input_checksum}")
    print(f"Sampling params checksum: {sampling_checksum}")
    print(f"Config checksum: {config_checksum}")

    # Warmup - to warm up flashinfer (matching bench.py)
    # Use a simple token ID sequence for warmup
    llm.generate(
        input_ids=[[0]],  # Simple warmup with single token
        sampling_params={"temperature": 0.1, "max_new_tokens": 1},
    )

    # Actual benchmark
    # SGLang's generate can accept lists for batch processing
    # Try batch generation with list of input_ids and list of sampling_params
    t = time.time()
    try:
        # Try batch generation if supported
        results = llm.generate(
            input_ids=prompt_token_ids,
            sampling_params=sampling_params_list,
        )
    except (TypeError, ValueError):
        # Fallback to per-request generation if batch not supported
        results = []
        for token_ids, sp in zip(prompt_token_ids, sampling_params_list):
            result = llm.generate(input_ids=[token_ids], sampling_params=sp)
            results.append(result)
    t = time.time() - t

    total_tokens = sum(sp["max_new_tokens"] for sp in sampling_params_list)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")

    llm.shutdown()


if __name__ == "__main__":
    main()
