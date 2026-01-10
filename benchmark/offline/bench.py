# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import hashlib
import json
import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


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

    # Model configuration
    model_path = "Qwen/Qwen3-0.6B"
    max_seq_len_override = 4096
    max_extend_tokens = 16384
    cuda_graph_max_bs = 256

    # align the hyperparameters
    llm = LLM(
        model_path, max_seq_len_override=max_seq_len_override, max_extend_tokens=max_extend_tokens, cuda_graph_max_bs=cuda_graph_max_bs
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]

    # Compute checksums for verification
    input_checksum = compute_checksum(prompt_token_ids)
    # Convert SamplingParams to dict for checksum
    sampling_params_dict = [
        {"temperature": sp.temperature, "ignore_eos": sp.ignore_eos, "max_tokens": sp.max_tokens}
        for sp in sampling_params
    ]
    sampling_checksum = compute_checksum(sampling_params_dict)
    # Config checksum - normalized for comparison with bench_sglang.py
    # Excludes implementation-specific optimizations (cuda_graph_max_bs)
    config_data = {
        "model_path": model_path,
        "max_seq_len": max_seq_len_override,  # normalized name
        "max_extend_tokens": max_extend_tokens,
        "num_seqs": num_seqs,
        "max_input_len": max_input_len,
        "max_output_len": max_ouput_len,
    }
    config_checksum = compute_checksum(config_data)

    print(f"Input token IDs checksum: {input_checksum}")
    print(f"Sampling params checksum: {sampling_checksum}")
    print(f"Config checksum: {config_checksum}")

    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
