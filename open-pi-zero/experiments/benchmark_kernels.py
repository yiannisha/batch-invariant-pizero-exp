#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import uuid
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from experiments.common import append_jsonl, seed_everything, utc_timestamp, write_json


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def time_call(function, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def launch_count(function) -> int:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        function()
        torch.cuda.synchronize()
    # Device events are kernels and device-side memory operations.  Restrict
    # to events that expose a CUDA device type and a nonzero duration.
    return sum(
        1
        for event in profiler.events()
        if str(event.device_type).endswith("CUDA") and event.device_time_total > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/performance/kernel_timings.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/performance/kernel_summary.json"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seed_everything(args.seed)
    dtype = getattr(torch, args.dtype)
    from batch_invariant_ops import bmm_persistent, matmul_persistent

    summaries = []
    for batch_size in args.batch_sizes:
        heads = 8
        generator = torch.Generator(device="cuda").manual_seed(args.seed + batch_size)
        left = torch.randn(
            batch_size * heads, 4, 281,
            generator=generator, device="cuda", dtype=dtype,
        )
        right = torch.randn(
            batch_size * heads, 281, 256,
            generator=generator, device="cuda", dtype=dtype,
        )
        functions = {
            "native_bmm": lambda: torch.bmm(left, right),
            "explicit_invariant_mm": lambda: torch.stack(
                [matmul_persistent(left[index], right[index]) for index in range(left.shape[0])]
            ),
            "persistent_invariant_bmm": lambda: bmm_persistent(left, right),
        }
        for implementation, function in functions.items():
            samples = time_call(function, args.warmup, args.repetitions)
            launches = launch_count(function)
            for sample_index, latency_ms in enumerate(samples):
                append_jsonl(
                    args.output,
                    [
                        {
                            "schema_version": 1,
                            "experiment_id": str(uuid.uuid4()),
                            "timestamp": utc_timestamp(),
                            "operator": "PV",
                            "implementation": implementation,
                            "batch_size": batch_size,
                            "shape_left": list(left.shape),
                            "shape_right": list(right.shape),
                            "dtype": args.dtype,
                            "sample_index": sample_index,
                            "latency_ms": latency_ms,
                        }
                    ],
                )
            summaries.append(
                {
                    "operator": "PV",
                    "implementation": implementation,
                    "batch_size": batch_size,
                    "dtype": args.dtype,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "median_latency_ms": statistics.median(samples),
                    "p95_latency_ms": percentile(samples, 0.95),
                    "requests_per_second": batch_size * 1000 / statistics.median(samples),
                    "kernel_launch_count": launches,
                }
            )
    write_json(args.summary, summaries)
    print(args.output)


if __name__ == "__main__":
    main()
