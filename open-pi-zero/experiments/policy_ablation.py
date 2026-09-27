#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import uuid
from pathlib import Path

import torch

from experiments.common import (
    IMPLEMENTATIONS,
    append_jsonl,
    compare_traces,
    index_inputs,
    load_pretrained_policy,
    make_raw_inputs,
    prepare_inputs,
    run_policy,
    seed_everything,
    tensor_metrics,
    utc_timestamp,
    write_json,
)


def arrangements(batch_sizes: list[int], available: int):
    for batch_size in batch_sizes:
        if batch_size == 1:
            yield {
                "batch_size": 1,
                "target_position": 0,
                "companions": "none",
                "indices": [0],
            }
            continue
        for companions in ("duplicate", "diverse"):
            for position_name, position in (
                ("first", 0),
                ("middle", batch_size // 2),
                ("last", batch_size - 1),
            ):
                if companions == "duplicate":
                    indices = [0] * batch_size
                else:
                    if available < batch_size:
                        raise ValueError("not enough companion inputs")
                    indices = list(range(1, batch_size))
                    indices.insert(position, 0)
                yield {
                    "batch_size": batch_size,
                    "target_position": position_name,
                    "target_index": position,
                    "companions": companions,
                    "indices": indices,
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostic/policy_ablation.jsonl"))
    parser.add_argument("--trace-output", type=Path, default=Path("results/diagnostic/first_divergence.json"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=IMPLEMENTATIONS)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --resume")
    seed_everything(args.seed)
    dtype = getattr(torch, args.dtype)
    model, config = load_pretrained_policy(
        args.checkpoint, config_path=args.config, dtype=dtype
    )
    maximum_batch = max(args.batch_sizes)
    raw = make_raw_inputs(config, maximum_batch, dtype, args.seed)
    all_inputs = prepare_inputs(model, raw, dtype)
    reference_inputs = index_inputs(all_inputs, [0])
    trace_reports = {}

    for implementation in args.implementations:
        reference, reference_trace, reference_counts, reference_elapsed = run_policy(
            model,
            reference_inputs,
            implementation,
            trace=not args.no_trace,
        )
        for arrangement in arrangements(args.batch_sizes, maximum_batch):
            candidate_inputs = index_inputs(all_inputs, arrangement["indices"])
            collect_trace = (
                not args.no_trace
                and arrangement["batch_size"] == 2
                and arrangement["companions"] == "diverse"
                and arrangement["target_position"] == "first"
            )
            candidate, candidate_trace, counts, elapsed = run_policy(
                model, candidate_inputs, implementation, trace=collect_trace
            )
            target_index = int(arrangement.get("target_index", 0))
            metrics = tensor_metrics(
                reference, candidate[target_index : target_index + 1]
            )
            record = {
                "schema_version": 1,
                "experiment_id": str(uuid.uuid4()),
                "timestamp": utc_timestamp(),
                "request_id": "synthetic-diagnostic/episode-000/observation-000/noise-000",
                "task": "synthetic-diagnostic",
                "episode_id": 0,
                "observation_index": 0,
                "noise_id": 0,
                "implementation": implementation,
                "dtype": args.dtype,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "execution_mode": "eager",
                "batch_size": arrangement["batch_size"],
                "target_batch_position": arrangement["target_position"],
                "companion_type": arrangement["companions"],
                "companion_ids": arrangement["indices"],
                "request_ordering": arrangement["indices"],
                "partition_id": "single_batch",
                "latency_seconds": elapsed,
                "reference_latency_seconds": reference_elapsed,
                "override_invocations": counts,
                "reference_override_invocations": reference_counts,
                **metrics,
            }
            append_jsonl(args.output, [record])
            if collect_trace:
                report = compare_traces(reference_trace, candidate_trace, target_index)
                trace_reports[implementation] = {
                    "arrangement": arrangement,
                    "first_divergence": report["first_divergence"],
                    "comparisons": report["comparisons"],
                }
        gc.collect()
        torch.cuda.empty_cache()

    if trace_reports:
        write_json(args.trace_output, trace_reports)
    print(args.output)


if __name__ == "__main__":
    main()
