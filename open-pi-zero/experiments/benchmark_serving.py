#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path

import numpy as np
import torch

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


CONFIGURATIONS = {
    "native_dynamic": ("native", "dynamic"),
    "invariant_dynamic": ("full_invariant", "dynamic"),
    "serial_singleton": ("native", "serial"),
    "fixed_size_padded": ("native", "fixed"),
}


def moved_subset(inputs: dict, indices: list[int]) -> dict:
    return {key: value[indices].cuda() for key, value in inputs.items()}


def call(model, inputs, implementation):
    with implementation_context(implementation) as overrides, torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(**inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if implementation == "full_invariant":
            overrides.assert_coverage()
    return output.detach().cpu(), elapsed


def singleton_references(model, inputs, implementation, count, fixed_size=None):
    references = []
    for index in range(count):
        if fixed_size is None:
            indices = [index]
        else:
            indices = [index] * fixed_size
        output, _ = call(model, moved_subset(inputs, indices), implementation)
        references.append(output[:1])
    return references


def make_batches(arrivals, mode, maximum_batch_size, max_delay):
    batches = []
    cursor = 0.0
    next_request = 0
    batch_id = 0
    while next_request < len(arrivals):
        cursor = max(cursor, float(arrivals[next_request]))
        if mode == "serial":
            selected = [next_request]
            service_start = cursor
        else:
            first = next_request
            deadline = max(cursor, float(arrivals[first])) + max_delay
            end = first + 1
            while (
                end < len(arrivals)
                and end - first < maximum_batch_size
                and arrivals[end] <= deadline
            ):
                end += 1
            selected = list(range(first, end))
            service_start = (
                max(cursor, float(arrivals[end - 1]))
                if len(selected) == maximum_batch_size
                else deadline
            )
        batches.append(
            {
                "batch_id": batch_id,
                "request_indices": selected,
                "service_start": service_start,
            }
        )
        next_request += len(selected)
        # Filled with real service time by the caller before scheduling the next batch.
        yield batches[-1]
        cursor = batches[-1]["completion"]
        batch_id += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/serving/requests.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/serving/summary.json"))
    parser.add_argument("--offered-loads", type=float, nargs="+", default=(1, 4, 8, 12, 20))
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--maximum-batch-size", type=int, default=8)
    parser.add_argument("--max-queue-delay-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20250203)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seed_everything(args.seed)
    dtype = getattr(torch, args.dtype)
    model, config = load_pretrained_policy(args.checkpoint, config_path=args.config, dtype=dtype)
    raw = make_raw_inputs(config, args.requests, dtype, args.seed)
    inputs = prepare_inputs(model, raw, dtype)
    references = {
        "native": singleton_references(model, inputs, "native", args.requests),
        "full_invariant": singleton_references(
            model, inputs, "full_invariant", args.requests
        ),
        "fixed": singleton_references(
            model,
            inputs,
            "native",
            args.requests,
            fixed_size=args.maximum_batch_size,
        ),
    }
    summaries = []
    for configuration, (implementation, mode) in CONFIGURATIONS.items():
        for offered_load in args.offered_loads:
            rng = np.random.default_rng(args.seed + int(offered_load * 1000))
            arrivals = np.cumsum(rng.exponential(1 / offered_load, args.requests))
            records = []
            for batch in make_batches(
                arrivals,
                mode,
                args.maximum_batch_size,
                args.max_queue_delay_ms / 1000,
            ):
                request_indices = batch["request_indices"]
                physical_indices = list(request_indices)
                if mode == "fixed":
                    physical_indices.extend(
                        [request_indices[-1]]
                        * (args.maximum_batch_size - len(request_indices))
                    )
                batch_inputs = moved_subset(inputs, physical_indices)
                output, service_seconds = call(model, batch_inputs, implementation)
                batch["completion"] = batch["service_start"] + service_seconds
                for position, request_index in enumerate(request_indices):
                    reference_key = "fixed" if mode == "fixed" else implementation
                    numerical = tensor_metrics(
                        references[reference_key][request_index],
                        output[position : position + 1],
                    )
                    record = {
                        "schema_version": 1,
                        "experiment_id": str(uuid.uuid4()),
                        "timestamp": utc_timestamp(),
                        "configuration": configuration,
                        "implementation": implementation,
                        "dtype": args.dtype,
                        "offered_load": offered_load,
                        "request_id": request_index,
                        "arrival_time": float(arrivals[request_index]),
                        "batch_assignment": batch["batch_id"],
                        "batch_occupancy": len(request_indices),
                        "physical_batch_size": len(physical_indices),
                        "target_batch_position": position,
                        "service_start": batch["service_start"],
                        "completion_time": batch["completion"],
                        "queueing_latency_seconds": batch["service_start"] - arrivals[request_index],
                        "service_latency_seconds": service_seconds,
                        "arrival_to_completion_seconds": batch["completion"] - arrivals[request_index],
                        "numerical_contract": numerical,
                    }
                    records.append(record)
                    append_jsonl(args.output, [record])
            end_to_end_ms = [item["arrival_to_completion_seconds"] * 1000 for item in records]
            completion_span = max(item["completion_time"] for item in records) - min(
                item["arrival_time"] for item in records
            )
            summaries.append(
                {
                    "configuration": configuration,
                    "implementation": implementation,
                    "dtype": args.dtype,
                    "offered_load": offered_load,
                    "request_count": len(records),
                    "throughput_requests_per_second": len(records) / completion_span,
                    "median_arrival_to_completion_ms": statistics.median(end_to_end_ms),
                    "p95_arrival_to_completion_ms": percentile(end_to_end_ms, 0.95),
                    "batch_occupancy_distribution": {
                        str(size): sum(item["batch_occupancy"] == size for item in records)
                        for size in range(1, args.maximum_batch_size + 1)
                    },
                    "numerical_contract_satisfied": all(
                        item["numerical_contract"]["exact"] for item in records
                    ),
                    "numerical_violation_count": sum(
                        not item["numerical_contract"]["exact"] for item in records
                    ),
                }
            )
    write_json(args.summary, summaries)
    print(args.output)


if __name__ == "__main__":
    main()
