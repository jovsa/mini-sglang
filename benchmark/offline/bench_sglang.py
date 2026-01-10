# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py
# SGLang version of bench.py

import argparse
import time
from random import randint, seed

import sglang as sgl


def main():
    parser = argparse.ArgumentParser(description="SGLang offline benchmark")
    parser.add_argument("--num-seqs", type=int, default=256, help="Number of sequences")
    parser.add_argument("--max-input-len", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--max-output-len", type=int, default=1024, help="Maximum output length")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B", help="Model path")
    parser.add_argument("--max-seq-len-override", type=int, default=4096, help="Max sequence length override")
    parser.add_argument("--max-extend-tokens", type=int, default=16384, help="Max extend tokens")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=256, help="CUDA graph max batch size (not used in SGLang)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    seed(args.seed)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_ouput_len = args.max_output_len

    # Model configuration - align with bench.py
    model_path = args.model_path
    max_model_len = args.max_seq_len_override  # equivalent to max_seq_len_override
    max_num_batched_tokens = args.max_extend_tokens  # equivalent to max_extend_tokens
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
    print(f"SGLANG:  Total: {total_tokens:6d}tok, Time: {t:6.2f}s, Throughput: {throughput:8.2f}tok/s")

    llm.shutdown()


if __name__ == "__main__":
    main()
