#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from experiments.common import (
    append_jsonl,
    compare_traces,
    flatten_tensors,
    load_numerical_freeze,
    load_pretrained_policy,
    prepare_inputs,
    run_policy,
    tensor_metrics,
    utc_timestamp,
)


@lru_cache(maxsize=128)
def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def raw_request(observation: dict, noise_id: int, dtype: torch.dtype) -> dict:
    data = load_npz(observation["tensor_path"])
    return {
        "input_ids": torch.from_numpy(data["input_ids"]).long(),
        "attention_mask": torch.from_numpy(data["attention_mask"]).long(),
        "pixel_values": torch.from_numpy(data["pixel_values"]).to(dtype),
        "proprios": torch.from_numpy(data["proprios"]).to(dtype),
        "initial_action": torch.from_numpy(data["initial_actions"][noise_id : noise_id + 1]).to(dtype),
    }


def concatenate(requests: list[dict]) -> dict:
    return {key: torch.cat([request[key] for request in requests]) for key in requests[0]}


def position_specs(batch_size: int):
    positions = [("first", 0), ("last", batch_size - 1)]
    middle = batch_size // 2
    if middle not in {0, batch_size - 1}:
        positions.insert(1, ("middle", middle))
    return positions


def arrangements(target: dict, companions: list[dict], batch_sizes: list[int]):
    yield "singleton", "none", "first", [target], 0, ["target"]
    for batch_size in batch_sizes:
        if batch_size == 1:
            continue
        for companion_type in ("duplicate", "diverse"):
            selected = (
                [target] * (batch_size - 1)
                if companion_type == "duplicate"
                else companions[: batch_size - 1]
            )
            for position_name, position in position_specs(batch_size):
                requests = list(selected)
                requests.insert(position, target)
                ordering = ["target" if request is target else f"companion-{index}" for index, request in enumerate(requests)]
                yield "batch_position", companion_type, position_name, requests, position, ordering

        diverse = [target, *companions[: batch_size - 1]]
        permuted = [target, *reversed(diverse[1:])]
        yield "permutation", "diverse", "first", permuted, 0, ["target", *[f"companion-{i}" for i in reversed(range(batch_size - 1))]]

        # Execute the same logical set in two partitions.  This record is for
        # the target-containing partition; request IDs retain the full logical set.
        split = max(1, batch_size // 2)
        partition = diverse[:split]
        yield "partition", "diverse", "first", partition, 0, ["target", *[f"companion-{i}" for i in range(split - 1)]]


def flow_comparisons(reference_trace: dict, candidate_trace: dict, target_index: int):
    report = compare_traces(reference_trace, candidate_trace, target_index)
    return [
        item
        for item in report["comparisons"]
        if item["trace_key"].startswith("flow.")
        or item["trace_key"].startswith("final_action.")
    ]


def denormalize(actions: torch.Tensor, statistics: dict) -> torch.Tensor:
    low = torch.tensor(statistics["action"]["p01"], dtype=torch.float64)
    high = torch.tensor(statistics["action"]["p99"], dtype=torch.float64)
    result = actions.detach().cpu().double().clone()
    result[..., :-1] = (result[..., :-1] + 1) / 2 * (high[:-1] - low[:-1]) + low[:-1]
    return result


def categorized_metrics(reference: torch.Tensor, candidate: torch.Tensor, statistics: dict):
    reference = denormalize(reference, statistics)
    candidate = denormalize(candidate, statistics)
    return {
        "full": tensor_metrics(reference, candidate),
        "translation": tensor_metrics(reference[..., :3], candidate[..., :3]),
        "rotation_euler": tensor_metrics(reference[..., 3:6], candidate[..., 3:6]),
        "gripper": tensor_metrics(reference[..., 6:], candidate[..., 6:]),
    }


def completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            keys.add(
                (
                    record["request_id"], record["implementation"],
                    record["transformation"], record["batch_size"],
                    record["target_batch_position"], record["companion_type"],
                )
            )
    return keys


def completed_fidelity_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as stream:
        return {
            record["request_id"]
            for line in stream
            if line.strip() and (record := json.loads(line))
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--freeze", type=Path, default=Path("results/numerical_freeze.json"))
    parser.add_argument("--statistics", type=Path, default=Path("config/fractal_statistics.json"))
    parser.add_argument("--output", type=Path, default=Path("results/heldout/invariance.jsonl"))
    parser.add_argument("--fidelity-output", type=Path, default=Path("results/heldout/singleton_fidelity.jsonl"))
    parser.add_argument("--split", choices=("diagnostic", "heldout"), default="heldout")
    parser.add_argument("--implementations", nargs="+", default=("native", "full_invariant"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; use --resume")
    freeze = load_numerical_freeze(args.freeze, args.checkpoint)
    if args.output.exists():
        with args.output.open() as stream:
            revisions = {
                record.get("numerical_policy_sha")
                for line in stream
                if line.strip() and (record := json.loads(line))
            }
        if revisions != {freeze["batch_invariant_pizero_sha"]}:
            raise RuntimeError(
                f"refusing to combine held-out revisions: existing={revisions}, "
                f"frozen={freeze['batch_invariant_pizero_sha']}"
            )
    manifest = json.loads(args.manifest.read_text())
    statistics = json.loads(args.statistics.read_text())
    observations = [item for item in manifest["observations"] if item["split"] == args.split]
    if args.limit is not None:
        observations = observations[: args.limit]
    dtype = getattr(torch, args.dtype)
    model, _ = load_pretrained_policy(args.checkpoint, config_path=args.config, dtype=dtype)
    done = completed_keys(args.output)
    fidelity_done = completed_fidelity_keys(args.fidelity_output)
    all_observations = manifest["observations"]

    for observation_index, observation in enumerate(observations):
        companion_observations = [
            all_observations[(observation_index + offset) % len(all_observations)]
            for offset in range(1, max(args.batch_sizes))
        ]
        for noise_id in range(3):
            target_raw = raw_request(observation, noise_id, dtype)
            companions_raw = [raw_request(item, noise_id, dtype) for item in companion_observations]
            request_id = observation["request_ids"][noise_id]
            request_ids_by_object = {id(target_raw): request_id}
            request_ids_by_object.update(
                {
                    id(raw): item["request_ids"][noise_id]
                    for raw, item in zip(
                        companions_raw, companion_observations, strict=True
                    )
                }
            )
            singleton_outputs = {}
            singleton_traces = {}
            for implementation in args.implementations:
                singleton_inputs = prepare_inputs(model, target_raw, dtype)
                singleton, trace, _, _ = run_policy(
                    model,
                    singleton_inputs,
                    implementation,
                    trace=True,
                    trace_prefixes=("flow.", "final_action."),
                )
                singleton_outputs[implementation] = singleton
                singleton_traces[implementation] = trace
                for transformation, companion_type, position_name, requests, target_position, ordering in arrangements(
                    target_raw, companions_raw, args.batch_sizes
                ):
                    key = (
                        request_id, implementation, transformation, len(requests),
                        position_name, companion_type,
                    )
                    if key in done:
                        continue
                    inputs = prepare_inputs(model, concatenate(requests), dtype)
                    candidate, candidate_trace, counts, elapsed = run_policy(
                        model,
                        inputs,
                        implementation,
                        trace=True,
                        trace_prefixes=("flow.", "final_action."),
                    )
                    target_output = candidate[target_position : target_position + 1]
                    metrics = tensor_metrics(singleton, target_output)
                    record = {
                        "schema_version": 1,
                        "experiment_id": str(uuid.uuid4()),
                        "timestamp": utc_timestamp(),
                        "request_id": request_id,
                        "numerical_policy_sha": freeze["batch_invariant_pizero_sha"],
                        "batch_invariant_ops_sha": freeze["batch_invariant_ops_sha"],
                        "checkpoint_sha256": freeze["checkpoint"]["sha256"],
                        "task": observation["task"],
                        "episode_id": observation["episode_id"],
                        "observation_index": observation["observation_index"],
                        "noise_id": noise_id,
                        "implementation": implementation,
                        "dtype": args.dtype,
                        "tf32": torch.backends.cuda.matmul.allow_tf32,
                        "execution_mode": "eager",
                        "transformation": transformation,
                        "batch_size": len(requests),
                        "target_batch_position": position_name,
                        "companion_type": companion_type,
                        "companion_ids": [
                            request_ids_by_object[id(request)]
                            for position, request in enumerate(requests)
                            if position != target_position
                        ],
                        "request_ordering": ordering,
                        "partition_id": transformation if transformation == "partition" else "single_batch",
                        "latency_seconds": elapsed,
                        "override_invocations": counts,
                        "output": metrics,
                        "preclip_and_flow": flow_comparisons(
                            singleton_traces[implementation], candidate_trace, target_position
                        ),
                        "denormalized": categorized_metrics(singleton, target_output, statistics),
                        "executed_first_two_actions": tensor_metrics(
                            singleton[:, :2], target_output[:, :2]
                        ),
                    }
                    append_jsonl(args.output, [record])
                    done.add(key)

            if (
                {"native", "full_invariant"}.issubset(singleton_outputs)
                and request_id not in fidelity_done
            ):
                append_jsonl(
                    args.fidelity_output,
                    [
                        {
                            "schema_version": 1,
                            "timestamp": utc_timestamp(),
                            "request_id": request_id,
                            "numerical_policy_sha": freeze["batch_invariant_pizero_sha"],
                            "batch_invariant_ops_sha": freeze["batch_invariant_ops_sha"],
                            "checkpoint_sha256": freeze["checkpoint"]["sha256"],
                            "task": observation["task"],
                            "episode_id": observation["episode_id"],
                            "observation_index": observation["observation_index"],
                            "noise_id": noise_id,
                            "comparison": "full_invariant_singleton_vs_native_singleton",
                            "normalized": tensor_metrics(
                                singleton_outputs["native"], singleton_outputs["full_invariant"]
                            ),
                            "denormalized": categorized_metrics(
                                singleton_outputs["native"],
                                singleton_outputs["full_invariant"],
                                statistics,
                            ),
                        }
                    ],
                )
                fidelity_done.add(request_id)
    print(args.output)


if __name__ == "__main__":
    main()
