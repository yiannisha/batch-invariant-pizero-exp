#!/usr/bin/env python3
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import torch

from experiments.common import (
    append_jsonl,
    index_inputs,
    load_pretrained_policy,
    make_raw_inputs,
    prepare_inputs,
    run_policy,
    seed_everything,
    tensor_metrics,
    utc_timestamp,
)


def per_request_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> list[dict]:
    return [
        {"request_index": index, **tensor_metrics(reference[index], candidate[index])}
        for index in range(reference.shape[0])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostic/batch_transformations.jsonl"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(2, 4, 8, 16, 32))
    parser.add_argument("--implementations", nargs="+", default=("native", "full_invariant"))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    seed_everything(args.seed)
    dtype = getattr(torch, args.dtype)
    model, config = load_pretrained_policy(args.checkpoint, config_path=args.config, dtype=dtype)
    maximum_batch = max(args.batch_sizes)
    inputs = prepare_inputs(
        model,
        make_raw_inputs(config, maximum_batch, dtype, args.seed),
        dtype,
    )

    for implementation in args.implementations:
        for batch_size in args.batch_sizes:
            canonical_indices = list(range(batch_size))
            canonical, _, canonical_counts, canonical_elapsed = run_policy(
                model, index_inputs(inputs, canonical_indices), implementation
            )

            permutation = list(reversed(canonical_indices))
            permuted, _, counts, elapsed = run_policy(
                model, index_inputs(inputs, permutation), implementation
            )
            restored = permuted[[permutation.index(index) for index in canonical_indices]]
            append_jsonl(
                args.output,
                [{
                    "schema_version": 1,
                    "experiment_id": str(uuid.uuid4()),
                    "timestamp": utc_timestamp(),
                    "request_set_id": f"synthetic-{args.seed}-B{batch_size}",
                    "implementation": implementation,
                    "dtype": args.dtype,
                    "transformation": "permutation",
                    "batch_size": batch_size,
                    "canonical_order": canonical_indices,
                    "candidate_order": permutation,
                    "restored_to_canonical_order": True,
                    "output": tensor_metrics(canonical, restored),
                    "per_request": per_request_metrics(canonical, restored),
                    "canonical_override_invocations": canonical_counts,
                    "candidate_override_invocations": counts,
                    "canonical_latency_seconds": canonical_elapsed,
                    "candidate_latency_seconds": elapsed,
                }],
            )

            split = batch_size // 2
            first, _, first_counts, first_elapsed = run_policy(
                model, index_inputs(inputs, canonical_indices[:split]), implementation
            )
            second, _, second_counts, second_elapsed = run_policy(
                model, index_inputs(inputs, canonical_indices[split:]), implementation
            )
            partitioned = torch.cat((first, second), dim=0)
            append_jsonl(
                args.output,
                [{
                    "schema_version": 1,
                    "experiment_id": str(uuid.uuid4()),
                    "timestamp": utc_timestamp(),
                    "request_set_id": f"synthetic-{args.seed}-B{batch_size}",
                    "implementation": implementation,
                    "dtype": args.dtype,
                    "transformation": "partition",
                    "batch_size": batch_size,
                    "partition_sizes": [split, batch_size - split],
                    "restored_to_canonical_order": True,
                    "output": tensor_metrics(canonical, partitioned),
                    "per_request": per_request_metrics(canonical, partitioned),
                    "canonical_override_invocations": canonical_counts,
                    "partition_override_invocations": [first_counts, second_counts],
                    "canonical_latency_seconds": canonical_elapsed,
                    "candidate_latency_seconds": first_elapsed + second_elapsed,
                }],
            )
    print(args.output)


if __name__ == "__main__":
    main()
