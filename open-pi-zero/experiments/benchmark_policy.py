#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
import uuid
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from experiments.benchmark_kernels import percentile
from experiments.common import (
    append_jsonl,
    implementation_context,
    index_inputs,
    load_pretrained_policy,
    make_raw_inputs,
    prepare_inputs,
    seed_everything,
    tensor_metrics,
    utc_timestamp,
    write_json,
)


def move(inputs):
    return {key: value.cuda() for key, value in inputs.items()}


def launch_count(function) -> int:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        function()
        torch.cuda.synchronize()
    return sum(
        1
        for event in profiler.events()
        if str(event.device_type).endswith("CUDA") and event.device_time_total > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/performance/policy_timings.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/performance/policy_summary.json"))
    parser.add_argument("--logical-batch-size", type=int, default=4)
    parser.add_argument("--fixed-batch-size", type=int, default=8)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.logical_batch_size > args.fixed_batch_size:
        raise ValueError("logical batch cannot exceed fixed batch")
    seed_everything(args.seed)
    dtype = getattr(torch, args.dtype)
    model, config = load_pretrained_policy(
        args.checkpoint, config_path=args.config, dtype=dtype
    )
    raw = make_raw_inputs(config, args.fixed_batch_size, dtype, args.seed)
    prepared = prepare_inputs(model, raw, dtype)
    logical = move(index_inputs(prepared, list(range(args.logical_batch_size))))
    singleton = [move(index_inputs(prepared, [index])) for index in range(args.logical_batch_size)]
    pad_indices = list(range(args.logical_batch_size)) + [0] * (
        args.fixed_batch_size - args.logical_batch_size
    )
    padded = move(index_inputs(prepared, pad_indices))

    configurations = {
        "native_dynamic": ("native", lambda: model(**logical)[: args.logical_batch_size]),
        "invariant_dynamic": (
            "full_invariant",
            lambda: model(**logical)[: args.logical_batch_size],
        ),
        "serial_singleton": (
            "native",
            lambda: torch.cat([model(**request) for request in singleton]),
        ),
        "fixed_size_padded": (
            "native",
            lambda: model(**padded)[: args.logical_batch_size],
        ),
    }

    summaries = []
    with torch.inference_mode():
        for name, (implementation, function) in configurations.items():
            session_summaries = []
            output = None
            launches = None
            for session in range(args.sessions):
                with implementation_context(implementation) as overrides:
                    for _ in range(args.warmup):
                        output = function()
                    torch.cuda.synchronize()
                    if launches is None:
                        launches = launch_count(function)
                    torch.cuda.reset_peak_memory_stats()
                    samples = []
                    for repetition in range(args.repetitions):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = function()
                        torch.cuda.synchronize()
                        latency_ms = (time.perf_counter() - start) * 1000
                        samples.append(latency_ms)
                        append_jsonl(
                            args.output,
                            [
                                {
                                    "schema_version": 1,
                                    "experiment_id": str(uuid.uuid4()),
                                    "timestamp": utc_timestamp(),
                                    "configuration": name,
                                    "implementation": implementation,
                                    "dtype": args.dtype,
                                    "logical_batch_size": args.logical_batch_size,
                                    "physical_batch_size": (
                                        args.fixed_batch_size
                                        if name == "fixed_size_padded"
                                        else args.logical_batch_size
                                    ),
                                    "session": session,
                                    "repetition": repetition,
                                    "latency_ms": latency_ms,
                                }
                            ],
                        )
                    if implementation == "full_invariant":
                        overrides.assert_coverage()
                    session_summaries.append(
                        {
                            "session": session,
                            "median_latency_ms": statistics.median(samples),
                            "p95_latency_ms": percentile(samples, 0.95),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        }
                    )
            all_medians = [entry["median_latency_ms"] for entry in session_summaries]
            summaries.append(
                {
                    "configuration": name,
                    "implementation": implementation,
                    "dtype": args.dtype,
                    "logical_batch_size": args.logical_batch_size,
                    "physical_batch_size": args.fixed_batch_size
                    if name == "fixed_size_padded"
                    else args.logical_batch_size,
                    "sessions": session_summaries,
                    "median_latency_ms": statistics.median(all_medians),
                    "p95_latency_ms": percentile(
                        [entry["p95_latency_ms"] for entry in session_summaries], 0.95
                    ),
                    "requests_per_second": args.logical_batch_size
                    * 1000
                    / statistics.median(all_medians),
                    "peak_allocated_bytes": max(
                        entry["peak_allocated_bytes"] for entry in session_summaries
                    ),
                    "kernel_launch_count": launches,
                }
            )

        # Numerical checks are deliberately outside all timed regions.
        singleton_native = []
        singleton_invariant = []
        with implementation_context("native"):
            singleton_native = [model(**request).detach() for request in singleton]
        with implementation_context("full_invariant") as overrides:
            singleton_invariant = [model(**request).detach() for request in singleton]
            overrides.assert_coverage()
        for summary in summaries:
            implementation, function = configurations[summary["configuration"]]
            with implementation_context(implementation):
                candidate = function().detach()
            if summary["configuration"] == "fixed_size_padded":
                fixed_contract = []
                canonical = candidate
                for request_index in range(args.logical_batch_size):
                    for position in sorted({0, args.fixed_batch_size // 2, args.fixed_batch_size - 1}):
                        permutation = list(pad_indices)
                        source_position = permutation.index(request_index)
                        permutation[source_position], permutation[position] = (
                            permutation[position],
                            permutation[source_position],
                        )
                        permuted_inputs = move(index_inputs(prepared, permutation))
                        with implementation_context(implementation):
                            permuted = model(**permuted_inputs).detach()
                        metric = tensor_metrics(
                            canonical[request_index : request_index + 1],
                            permuted[position : position + 1],
                        )
                        fixed_contract.append(
                            {
                                "request_index": request_index,
                                "target_position": position,
                                **metric,
                            }
                        )
                summary["numerical_contract"] = fixed_contract
                summary["numerical_contract_satisfied"] = all(
                    item["exact"] for item in fixed_contract
                )
                continue
            references = (
                singleton_invariant
                if implementation == "full_invariant"
                else singleton_native
            )
            summary["numerical_contract"] = [
                tensor_metrics(reference, candidate[index : index + 1])
                for index, reference in enumerate(references)
            ]
            summary["numerical_contract_satisfied"] = all(
                item["exact"] for item in summary["numerical_contract"]
            )

    write_json(args.summary, summaries)
    print(args.output)


if __name__ == "__main__":
    main()
