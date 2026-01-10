# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import argparse
import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    parser = argparse.ArgumentParser(description="Mini-SGLang offline benchmark")
    parser.add_argument("--num-seqs", type=int, default=256, help="Number of sequences")
    parser.add_argument("--max-input-len", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--max-output-len", type=int, default=1024, help="Maximum output length")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B", help="Model path")
    parser.add_argument("--max-seq-len-override", type=int, default=4096, help="Max sequence length override")
    parser.add_argument("--max-extend-tokens", type=int, default=16384, help="Max extend tokens")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=256, help="CUDA graph max batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    seed(args.seed)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_ouput_len = args.max_output_len

    # Model configuration
    model_path = args.model_path
    max_seq_len_override = args.max_seq_len_override
    max_extend_tokens = args.max_extend_tokens
    cuda_graph_max_bs = args.cuda_graph_max_bs

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
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"MINISGL: Total: {total_tokens:6d}tok, Time: {t:6.2f}s, Throughput: {throughput:8.2f}tok/s")


if __name__ == "__main__":
    main()
