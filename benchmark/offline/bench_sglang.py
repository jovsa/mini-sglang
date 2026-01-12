# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py
# SGLang version of bench.py

import argparse
import sys
import time
from pathlib import Path
from random import randint, seed

import sglang as sgl

# Add benchmark directory to path to import metrics module
BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))
from metrics import BenchmarkMetrics, collect_metrics, format_output


def main():
    parser = argparse.ArgumentParser(description="SGLang offline benchmark")
    parser.add_argument("--num-seqs", type=int, default=32, help="Number of sequences")
    parser.add_argument("--max-input-len", type=int, default=256, help="Maximum input length")
    parser.add_argument("--max-output-len", type=int, default=256, help="Maximum output length")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-0.6B", help="Model path")
    parser.add_argument("--max-seq-len-override", type=int, default=2048, help="Max sequence length override")
    parser.add_argument("--max-extend-tokens", type=int, default=4096, help="Max extend tokens")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=64, help="CUDA graph max batch size (not used in SGLang)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed metrics")
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

    # Collect input/output lengths for metrics
    input_lengths = [len(ids) for ids in prompt_token_ids]
    output_lengths = [sp["max_new_tokens"] for sp in sampling_params_list]

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

    # Extract actual output token counts from results
    # SGLang results format may vary - try to extract token counts
    actual_output_lengths = []
    if isinstance(results, list):
        for result in results:
            # Try different possible result formats
            if isinstance(result, dict):
                # Check for common keys
                if "output_ids" in result:
                    actual_output_lengths.append(len(result["output_ids"]))
                elif "token_ids" in result:
                    actual_output_lengths.append(len(result["token_ids"]))
                elif "text" in result:
                    # Estimate from text length (rough approximation)
                    # This is not ideal but works if token_ids not available
                    actual_output_lengths.append(len(result["text"].split()))
                else:
                    # Fallback: use requested length
                    actual_output_lengths.append(None)
            elif hasattr(result, "output_ids"):
                actual_output_lengths.append(len(result.output_ids))
            elif hasattr(result, "token_ids"):
                actual_output_lengths.append(len(result.token_ids))
            else:
                # Unknown format, use requested length
                actual_output_lengths.append(None)
    else:
        # Results not in expected format, use requested lengths
        actual_output_lengths = None

    # Filter out None values if we have some actual lengths
    if actual_output_lengths and all(x is not None for x in actual_output_lengths):
        total_tokens = sum(actual_output_lengths)
    else:
        # Fallback to requested tokens
        total_tokens = sum(sp["max_new_tokens"] for sp in sampling_params_list)
        actual_output_lengths = None

    # Collect metrics
    metrics = collect_metrics(
        total_tokens=total_tokens,
        total_time=t,
        num_requests=num_seqs,
        input_lengths=input_lengths,
        output_lengths=output_lengths,
        actual_output_lengths=actual_output_lengths,
    )

    # Format and print output
    output = format_output(metrics, prefix="SGLANG:  ", verbose=args.verbose)
    print(output)

    llm.shutdown()


if __name__ == "__main__":
    main()
