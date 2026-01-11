"""Shared metrics module for offline benchmarks.

This module provides framework-agnostic metrics collection and reporting
to ensure consistent measurement between mini-sglang and sglang benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BenchmarkMetrics:
    """Container for all benchmark metrics"""

    total_tokens: int
    total_time: float
    throughput: float
    num_requests: int
    requests_per_sec: float
    # Latency metrics (if per-request data available)
    ttft_avg: float | None = None
    ttft_p50: float | None = None
    ttft_p90: float | None = None
    ttft_p99: float | None = None
    tpot_avg: float | None = None
    tpot_p50: float | None = None
    tpot_p90: float | None = None
    tpot_p99: float | None = None
    e2e_avg: float | None = None
    e2e_p50: float | None = None
    e2e_p90: float | None = None
    e2e_p99: float | None = None
    # Input/output stats
    avg_input_len: float | None = None
    avg_output_len: float | None = None
    total_input_tokens: int | None = None
    input_output_ratio: float | None = None  # output / input ratio
    # Actual vs requested tokens
    actual_output_tokens: int | None = None  # Actual tokens generated
    requested_output_tokens: int | None = None  # Requested tokens
    token_efficiency: float | None = None  # actual / requested ratio


def calculate_percentiles(times: List[float]) -> Tuple[float, float, float, float, float]:
    """Calculate average, p50, p90, p99, and max from a list of times."""
    if not times:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    sorted_times = sorted(times)
    n = len(sorted_times)

    avg = sum(sorted_times) / n
    p50 = sorted_times[int(n * 0.5)]
    p90 = sorted_times[int(n * 0.9)] if n > 1 else sorted_times[0]
    p99 = sorted_times[int(n * 0.99)] if n > 10 else sorted_times[-1]
    max_val = sorted_times[-1]

    return avg, p50, p90, p99, max_val


def collect_metrics(
    total_tokens: int,
    total_time: float,
    num_requests: int,
    input_lengths: List[int] | None = None,
    output_lengths: List[int] | None = None,
    actual_output_lengths: List[int] | None = None,
    per_request_times: List[float] | None = None,
    per_token_times: List[List[float]] | None = None,
) -> BenchmarkMetrics:
    """Calculate all metrics from raw benchmark data.

    Args:
        total_tokens: Total number of output tokens generated (actual)
        total_time: Total wall-clock time in seconds
        num_requests: Number of requests processed
        input_lengths: List of input sequence lengths (optional)
        output_lengths: List of requested output sequence lengths (optional)
        actual_output_lengths: List of actual output sequence lengths (optional)
        per_request_times: List of per-request end-to-end times in seconds (optional)
        per_token_times: List of lists, where each inner list contains token generation times (optional)

    Returns:
        BenchmarkMetrics object with all calculated metrics
    """
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    requests_per_sec = num_requests / total_time if total_time > 0 else 0.0

    # Calculate input/output statistics
    avg_input_len = None
    avg_output_len = None
    total_input_tokens = None

    if input_lengths:
        avg_input_len = sum(input_lengths) / len(input_lengths)
        total_input_tokens = sum(input_lengths)

    # Use actual_output_lengths if provided, otherwise use output_lengths
    actual_output_tokens = None
    requested_output_tokens = None
    token_efficiency = None
    input_output_ratio = None

    if actual_output_lengths:
        avg_output_len = sum(actual_output_lengths) / len(actual_output_lengths)
        actual_output_tokens = sum(actual_output_lengths)
    elif output_lengths:
        avg_output_len = sum(output_lengths) / len(output_lengths)

    # Calculate input/output ratio
    if total_input_tokens is not None and total_input_tokens > 0:
        if actual_output_tokens is not None:
            input_output_ratio = actual_output_tokens / total_input_tokens
        elif total_tokens > 0:
            input_output_ratio = total_tokens / total_input_tokens

    # Calculate token efficiency (actual vs requested)
    if output_lengths:
        requested_output_tokens = sum(output_lengths)
        if actual_output_tokens is not None and requested_output_tokens > 0:
            token_efficiency = actual_output_tokens / requested_output_tokens
        elif total_tokens > 0 and requested_output_tokens > 0:
            # Use total_tokens as actual if actual_output_lengths not provided
            token_efficiency = total_tokens / requested_output_tokens

    # Calculate latency metrics if per-request data is available
    ttft_avg = ttft_p50 = ttft_p90 = ttft_p99 = None
    tpot_avg = tpot_p50 = tpot_p90 = tpot_p99 = None
    e2e_avg = e2e_p50 = e2e_p90 = e2e_p99 = None

    if per_token_times:
        # Calculate TTFT (Time To First Token) - time to first token for each request
        first_times = []
        for token_times in per_token_times:
            if token_times:
                first_times.append(token_times[0])

        if first_times:
            ttft_avg, ttft_p50, ttft_p90, ttft_p99, _ = calculate_percentiles(first_times)
            # Convert to milliseconds
            ttft_avg *= 1000
            ttft_p50 *= 1000
            ttft_p90 *= 1000
            ttft_p99 *= 1000

        # Calculate TPOT (Time Per Output Token) - average time per token after first
        all_tpot_times = []
        for token_times in per_token_times:
            if len(token_times) > 1:
                # Skip first token, use rest for TPOT
                all_tpot_times.extend(token_times[1:])

        if all_tpot_times:
            tpot_avg, tpot_p50, tpot_p90, tpot_p99, _ = calculate_percentiles(all_tpot_times)
            # Convert to milliseconds
            tpot_avg *= 1000
            tpot_p50 *= 1000
            tpot_p90 *= 1000
            tpot_p99 *= 1000

    if per_request_times:
        e2e_avg, e2e_p50, e2e_p90, e2e_p99, _ = calculate_percentiles(per_request_times)

    return BenchmarkMetrics(
        total_tokens=total_tokens,
        total_time=total_time,
        throughput=throughput,
        num_requests=num_requests,
        requests_per_sec=requests_per_sec,
        ttft_avg=ttft_avg,
        ttft_p50=ttft_p50,
        ttft_p90=ttft_p90,
        ttft_p99=ttft_p99,
        tpot_avg=tpot_avg,
        tpot_p50=tpot_p50,
        tpot_p90=tpot_p90,
        tpot_p99=tpot_p99,
        e2e_avg=e2e_avg,
        e2e_p50=e2e_p50,
        e2e_p90=e2e_p90,
        e2e_p99=e2e_p99,
        avg_input_len=avg_input_len,
        avg_output_len=avg_output_len,
        total_input_tokens=total_input_tokens,
        input_output_ratio=input_output_ratio,
        actual_output_tokens=actual_output_tokens,
        requested_output_tokens=requested_output_tokens,
        token_efficiency=token_efficiency,
    )


def format_number(value: float, unit: str = "") -> str:
    """Format a number for display."""
    if value >= 1000:
        return f"{int(value):>6}{unit}"
    elif value >= 10:
        return f"{value:>6.2f}{unit}"
    else:
        return f"{value:>6.4f}{unit}"


def format_output(metrics: BenchmarkMetrics, prefix: str = "", verbose: bool = False) -> str:
    """Format metrics for consistent output.

    Args:
        metrics: BenchmarkMetrics object to format
        prefix: Prefix string (e.g., "MINISGL:" or "SGLANG:")
        verbose: If True, include detailed metrics

    Returns:
        Formatted string with metrics
    """
    lines = []

    # Main line with core metrics
    main_line = f"{prefix} Total: {metrics.total_tokens:6d}tok, Time: {metrics.total_time:6.2f}s, Throughput: {metrics.throughput:8.2f}tok/s"

    # Add TTFT if available
    if metrics.ttft_p50 is not None:
        main_line += f", TTFT: {metrics.ttft_p50:.2f}ms (p50)"

    lines.append(main_line)

    # Detailed metrics in verbose mode
    if verbose:
        if metrics.ttft_avg is not None:
            lines.append(
                f"  TTFT: avg={format_number(metrics.ttft_avg, 'ms')}, "
                f"p50={format_number(metrics.ttft_p50 or 0, 'ms')}, "
                f"p90={format_number(metrics.ttft_p90 or 0, 'ms')}, "
                f"p99={format_number(metrics.ttft_p99 or 0, 'ms')}"
            )

        if metrics.tpot_avg is not None:
            lines.append(
                f"  TPOT: avg={format_number(metrics.tpot_avg, 'ms')}, "
                f"p50={format_number(metrics.tpot_p50 or 0, 'ms')}, "
                f"p90={format_number(metrics.tpot_p90 or 0, 'ms')}, "
                f"p99={format_number(metrics.tpot_p99 or 0, 'ms')}"
            )

        if metrics.e2e_avg is not None:
            lines.append(
                f"  E2E:  avg={format_number(metrics.e2e_avg, 's')}, "
                f"p50={format_number(metrics.e2e_p50 or 0, 's')}, "
                f"p90={format_number(metrics.e2e_p90 or 0, 's')}, "
                f"p99={format_number(metrics.e2e_p99 or 0, 's')}"
            )

        if metrics.avg_input_len is not None:
            lines.append(f"  Input: avg={metrics.avg_input_len:.1f}tok")
        if metrics.avg_output_len is not None:
            lines.append(f"  Output: avg={metrics.avg_output_len:.1f}tok")
        if metrics.input_output_ratio is not None:
            lines.append(f"  Input/Output Ratio: {metrics.input_output_ratio:.2f}")
        if metrics.token_efficiency is not None:
            lines.append(
                f"  Token Efficiency: {metrics.token_efficiency:.2%} "
                f"(actual: {metrics.actual_output_tokens or metrics.total_tokens}, "
                f"requested: {metrics.requested_output_tokens})"
            )

        lines.append(f"  Requests: {metrics.num_requests}, Req/s: {metrics.requests_per_sec:.2f}")

    return "\n".join(lines)
