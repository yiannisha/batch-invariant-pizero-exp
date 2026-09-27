#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.common import write_json


def read_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def paired_bootstrap(values: np.ndarray, seed: int, resamples: int) -> dict:
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for index in range(resamples):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return {
        "estimate": float(values.mean()),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "unit": "episode",
    }


def quaternion_angle(left, right):
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return 2 * np.arccos(np.clip(abs(np.dot(left, right)), 0, 1))


def trajectory_metrics(left: dict, right: dict) -> dict:
    count = min(len(left["trajectory"]), len(right["trajectory"]))
    position = []
    rotation = []
    gripper = []
    for index in range(count):
        left_eef = np.asarray(left["trajectory"][index]["agent"]["eef_pos"])
        right_eef = np.asarray(right["trajectory"][index]["agent"]["eef_pos"])
        position.append(float(np.linalg.norm(left_eef[:3] - right_eef[:3])))
        rotation.append(float(quaternion_angle(left_eef[3:7], right_eef[3:7])))
        gripper.append(float(abs(left_eef[7] - right_eef[7])))
    return {
        "aligned_state_count": count,
        "end_effector_position_max_m": max(position),
        "end_effector_position_terminal_m": position[-1],
        "end_effector_rotation_max_rad": max(rotation),
        "end_effector_rotation_terminal_rad": rotation[-1],
        "gripper_max_native_units": max(gripper),
        "gripper_terminal_native_units": gripper[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("results/simpler/episodes.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/simpler/summary.json"))
    parser.add_argument("--bootstrap-seed", type=int, default=20250401)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    episodes = read_jsonl(args.episodes)
    by_key = {
        (item["task"], item["initialization_id"], item["condition"]): item
        for item in episodes
    }
    tasks = sorted({item["task"] for item in episodes})
    result = {
        "bootstrap": {
            "seed": args.bootstrap_seed,
            "resamples": args.bootstrap_resamples,
            "unit": "matched episode initialization",
        },
        "tasks": {},
    }
    for task_index, task in enumerate(tasks):
        initialization_ids = sorted(
            {
                item["initialization_id"]
                for item in episodes
                if item["task"] == task
            }
        )
        conditions = {
            condition: [
                by_key[(task, initialization, condition)]
                for initialization in initialization_ids
            ]
            for condition in (
                "native_singleton",
                "native_dynamic",
                "patched_singleton",
                "patched_dynamic",
            )
        }
        task_result = {
            "episodes": len(initialization_ids),
            "success_rate": {
                name: float(np.mean([item["success"] for item in records]))
                for name, records in conditions.items()
            },
            "paired_differences": {},
            "trajectory_pairs": defaultdict(list),
        }
        comparisons = {
            "native_dynamic_minus_singleton": ("native_dynamic", "native_singleton"),
            "patched_dynamic_minus_singleton": ("patched_dynamic", "patched_singleton"),
            "patched_minus_native_singleton": ("patched_singleton", "native_singleton"),
        }
        for name, (left_name, right_name) in comparisons.items():
            differences = np.asarray(
                [
                    int(left["success"]) - int(right["success"])
                    for left, right in zip(
                        conditions[left_name], conditions[right_name], strict=True
                    )
                ],
                dtype=float,
            )
            task_result["paired_differences"][name] = {
                **paired_bootstrap(
                    differences,
                    args.bootstrap_seed + task_index,
                    args.bootstrap_resamples,
                ),
                "disagreement_count": int(np.count_nonzero(differences)),
            }
            for left, right in zip(
                conditions[left_name], conditions[right_name], strict=True
            ):
                task_result["trajectory_pairs"][name].append(
                    trajectory_metrics(left, right)
                )
        task_result["trajectory_pairs"] = dict(task_result["trajectory_pairs"])
        result["tasks"][task] = task_result
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
