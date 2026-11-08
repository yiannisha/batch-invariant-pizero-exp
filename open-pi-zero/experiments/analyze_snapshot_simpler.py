#!/usr/bin/env python3
"""Audit exact-start SIMPLER pairs before attributing outcome differences."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.analyze_simpler import paired_bootstrap
from experiments.common import sha256_file, write_json
from experiments.prepare_replay import TASKS
from experiments.simpler_snapshot_eval import CONDITIONS


def first_action_difference(singleton: dict, dynamic: dict) -> dict | None:
    left = singleton["action_sequence"]
    right = dynamic["action_sequence"]
    for index, (a, b) in enumerate(zip(left, right)):
        if a["normalized_action"] == b["normalized_action"]:
            continue
        return {
            "action_index": index,
            "policy_call": b["policy_call"],
            "batch_size": b["batch_size"],
            "same_target_input": a["target_input_sha256"] == b["target_input_sha256"],
            "same_prior_recorded_states": singleton["trajectory"][: index + 1] == dynamic["trajectory"][: index + 1],
        }
    if len(left) != len(right):
        return {"action_index": min(len(left), len(right)), "reason": "trajectory_length_only"}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--initializations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20250401)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    groups = defaultdict(dict)
    file_hashes = {}
    for path in args.episodes:
        file_hashes[str(path)] = sha256_file(path)
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["task"], record["initialization_id"])
                condition = record["condition"]
                if condition in groups[key]:
                    raise RuntimeError(f"duplicate record: {key}/{condition}")
                groups[key][condition] = record
    expected = {(task, index) for task in TASKS for index in range(args.initializations)}
    if set(groups) != expected:
        raise RuntimeError(f"block coverage mismatch: found {len(groups)}, expected {len(expected)}")
    result = {"schema_version": 1, "episode_sha256": file_hashes, "tasks": {}}
    for task_index, task in enumerate(TASKS):
        blocks = [groups[(task, index)] for index in range(args.initializations)]
        first_differences = []
        native_outcome_disagreements = 0
        qualified_native_outcome_disagreements = 0
        patched_outcome_disagreements = 0
        for index, block in enumerate(blocks):
            if set(block) != set(CONDITIONS):
                raise RuntimeError(f"condition coverage mismatch: {task}/{index}")
            records = list(block.values())
            if not all(record["aa_qualified"] and all(record["restore_checks"].values()) for record in records):
                raise RuntimeError(f"A/A or restore check failed: {task}/{index}")
            if len({record["snapshot"]["sim_state_sha256"] for record in records}) != 1:
                raise RuntimeError(f"initial physics state mismatch: {task}/{index}")
            if len({record["snapshot"]["observation_sha256"] for record in records}) != 1:
                raise RuntimeError(f"initial observation mismatch: {task}/{index}")
            if len({record["initial_input_sha256"] for record in records}) != 1:
                raise RuntimeError(f"initial policy input mismatch: {task}/{index}")
            native_singleton = block["native_singleton"]
            native_dynamic = block["native_dynamic"]
            patched_singleton = block["patched_singleton"]
            patched_dynamic = block["patched_dynamic"]
            difference = first_action_difference(native_singleton, native_dynamic)
            first_differences.append(difference)
            if native_singleton["success"] != native_dynamic["success"]:
                native_outcome_disagreements += 1
                if difference and difference.get("batch_size", 1) > 1 and difference.get("same_target_input") and difference.get("same_prior_recorded_states"):
                    qualified_native_outcome_disagreements += 1
            if patched_singleton["success"] != patched_dynamic["success"]:
                patched_outcome_disagreements += 1
            patched_actions_equal = (
                [item["normalized_action"] for item in patched_singleton["action_sequence"]]
                == [item["normalized_action"] for item in patched_dynamic["action_sequence"]]
            )
            if not patched_actions_equal or patched_singleton["trajectory"] != patched_dynamic["trajectory"]:
                raise RuntimeError(f"patched closed-loop pair diverged: {task}/{index}")
        if qualified_native_outcome_disagreements != native_outcome_disagreements:
            raise RuntimeError(
                f"unattributable native outcome disagreement in {task}: "
                f"{qualified_native_outcome_disagreements}/{native_outcome_disagreements} qualified"
            )
        conditions = {
            name: [block[name] for block in blocks]
            for name in ("native_singleton", "native_dynamic", "patched_singleton", "patched_dynamic")
        }
        comparisons = {}
        for name, left_name, right_name in (
            ("native_dynamic_minus_singleton", "native_dynamic", "native_singleton"),
            ("patched_dynamic_minus_singleton", "patched_dynamic", "patched_singleton"),
            ("patched_singleton_minus_native_singleton", "patched_singleton", "native_singleton"),
            ("patched_dynamic_minus_native_singleton", "patched_dynamic", "native_singleton"),
            ("patched_dynamic_minus_native_dynamic", "patched_dynamic", "native_dynamic"),
        ):
            values = np.asarray(
                [int(left["success"]) - int(right["success"]) for left, right in zip(conditions[left_name], conditions[right_name])],
                dtype=float,
            )
            comparisons[name] = {
                **paired_bootstrap(values, args.bootstrap_seed + task_index, args.bootstrap_resamples),
                "disagreements": int(np.count_nonzero(values)),
            }
        result["tasks"][task] = {
            "blocks": len(blocks),
            "success_rates": {name: float(np.mean([record["success"] for record in records])) for name, records in conditions.items()},
            "paired_differences": comparisons,
            "native_action_divergence_count": sum(item is not None for item in first_differences),
            "native_outcome_disagreements": native_outcome_disagreements,
            "qualified_native_outcome_disagreements": qualified_native_outcome_disagreements,
            "patched_outcome_disagreements": patched_outcome_disagreements,
            "first_native_action_differences": first_differences,
        }
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
