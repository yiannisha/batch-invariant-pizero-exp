#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from uuid import UUID

from experiments.common import sha256_file, utc_timestamp, write_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def arrangement_key(item: dict) -> tuple:
    return (
        item["transformation"],
        item["batch_size"],
        item["target_batch_position"],
        item["companion_type"],
    )


def assert_complete_arrangement_groups(groups: dict) -> None:
    expected_transformations = {
        "singleton": 1,
        "batch_position": 16,
        "permutation": 3,
        "partition": 3,
    }
    expected_batch_sizes = {1: 2, 2: 6, 4: 8, 8: 7}
    for items in groups.values():
        assert len(items) == 23
        assert len({arrangement_key(item) for item in items}) == 23
        assert {
            name: sum(item["transformation"] == name for item in items)
            for name in expected_transformations
        } == expected_transformations
        assert {
            batch_size: sum(item["batch_size"] == batch_size for item in items)
            for batch_size in expected_batch_sizes
        } == expected_batch_sizes


def policy_audit(path: Path, expected: int) -> dict:
    records = jsonl(path)
    assert len(records) == expected, (path, len(records), expected)
    nonsingleton = [item for item in records if item["batch_size"] > 1]
    by_implementation = defaultdict(list)
    for item in nonsingleton:
        by_implementation[item["implementation"]].append(item)
    assert all(item["exact"] for item in by_implementation["full_invariant"])
    return {
        "records": len(records),
        "implementations": {
            name: {
                "arrangements": len(items),
                "violations": sum(not item["exact"] for item in items),
                "maximum_error": max(item["max_absolute_error"] for item in items),
            }
            for name, items in sorted(by_implementation.items())
        },
    }


def heldout_audit(path: Path, fidelity_path: Path, freeze: dict) -> dict:
    records = jsonl(path)
    fidelity = jsonl(fidelity_path)
    assert len(records) == 110_400, (path, len(records), 110_400)
    assert len(fidelity) == 2_400, (fidelity_path, len(fidelity), 2_400)
    assert len({item["request_id"] for item in records}) == 2_400
    assert len({item["request_id"] for item in fidelity}) == 2_400
    groups = defaultdict(list)
    by_implementation = defaultdict(list)
    for item in records:
        assert item["numerical_policy_sha"] == freeze["batch_invariant_pizero_sha"]
        assert item["batch_invariant_ops_sha"] == freeze["batch_invariant_ops_sha"]
        assert item["checkpoint_sha256"] == freeze["checkpoint"]["sha256"]
        groups[(item["request_id"], item["implementation"])].append(item)
        if item["batch_size"] > 1:
            by_implementation[item["implementation"]].append(item)
    assert len(groups) == 4_800
    assert_complete_arrangement_groups(groups)
    full = by_implementation["full_invariant"]
    # There are 23 arrangements per implementation/request.  The canonical
    # singleton and the B=2 partition (a one-request physical partition) are
    # both size one, leaving 21 physical nonsingleton arrangements.
    assert len(full) == 50_400
    assert all(item["output"]["exact"] for item in full)
    return {
        "arrangement_records": len(records),
        "fixed_request_noise_pairs": len(fidelity),
        "implementations": {
            name: {
                "nonsingleton_arrangements": len(items),
                "violations": sum(not item["output"]["exact"] for item in items),
                "maximum_error": max(
                    item["output"]["max_absolute_error"] for item in items
                ),
            }
            for name, items in sorted(by_implementation.items())
        },
        "singleton_fidelity_exact": sum(
            item["normalized"]["exact"] for item in fidelity
        ),
        "singleton_fidelity_maximum_error": max(
            item["normalized"]["max_absolute_error"] for item in fidelity
        ),
    }


def diagnostic_replay_audit(path: Path, fidelity_path: Path, freeze: dict) -> dict:
    records = jsonl(path)
    fidelity = jsonl(fidelity_path)
    assert len(records) == 1_104, (path, len(records), 1_104)
    assert len(fidelity) == 24, (fidelity_path, len(fidelity), 24)
    assert {item["task"] for item in records} == {
        "pick_can", "move_near", "open_drawer", "close_drawer",
    }
    groups = defaultdict(list)
    by_implementation = defaultdict(list)
    for item in records:
        assert item["numerical_policy_sha"] == freeze["batch_invariant_pizero_sha"]
        assert item["batch_invariant_ops_sha"] == freeze["batch_invariant_ops_sha"]
        assert item["checkpoint_sha256"] == freeze["checkpoint"]["sha256"]
        groups[(item["request_id"], item["implementation"])].append(item)
        by_implementation[item["implementation"]].append(item)
    assert len(groups) == 48
    assert_complete_arrangement_groups(groups)
    assert all(item["output"]["exact"] for item in by_implementation["full_invariant"])
    return {
        "arrangement_records": len(records),
        "fixed_request_noise_pairs": len(fidelity),
        "tasks": sorted({item["task"] for item in records}),
        "implementations": {
            name: {
                "arrangements": len(items),
                "violations": sum(not item["output"]["exact"] for item in items),
                "maximum_error": max(
                    item["output"]["max_absolute_error"] for item in items
                ),
            }
            for name, items in sorted(by_implementation.items())
        },
    }


def flow_summary_audit(path: Path) -> dict:
    summary = json.loads(path.read_text())
    assert set(summary) == {"native", "full_invariant"}
    for steps in summary.values():
        assert set(steps) == {str(step) for step in range(1, 11)}
        assert all(item["comparisons"] == 50_400 for item in steps.values())
    assert all(
        item["exact"] == item["comparisons"]
        and item["maximum_absolute_error"] == 0.0
        for item in summary["full_invariant"].values()
    )
    return summary


def diagnostic_flow_summary_audit(path: Path, trace_path: Path) -> dict:
    summary = json.loads(path.read_text())
    traces = json.loads(trace_path.read_text())
    expected_implementations = {
        "native",
        "existing_invariant_ops",
        "invariant_plus_patch_projection",
        "full_invariant",
    }
    assert set(summary["implementations"]) == expected_implementations
    assert summary["scope"] == "single frozen B=2 diverse diagnostic arrangement"
    assert summary["central_statistic"] == "mean absolute error across action-state elements"
    assert summary["tail_statistic"] == "maximum absolute error across action-state elements"
    for implementation, steps in summary["implementations"].items():
        assert set(steps) == {str(step) for step in range(11)}
        expected = {
            "0": {
                "exact": True,
                "mean_absolute_error": 0.0,
                "maximum_absolute_error": 0.0,
            }
        }
        for item in traces[implementation]["comparisons"]:
            if not item["trace_key"].startswith("flow.step_") \
                    or ".updated_action_state" not in item["trace_key"]:
                continue
            step = str(int(item["trace_key"].split(".")[1].split("_")[1]) + 1)
            expected[step] = {
                "exact": item["exact"],
                "mean_absolute_error": item["mean_absolute_error"],
                "maximum_absolute_error": item["max_absolute_error"],
            }
        assert steps == expected
    assert all(
        item["exact"]
        and item["mean_absolute_error"] == 0.0
        and item["maximum_absolute_error"] == 0.0
        for item in summary["implementations"]["full_invariant"].values()
    )
    return summary


def simpler_audit(
    path: Path,
    summary_path: Path,
    runtime_path: Path,
    freeze: dict,
    rtx_environment: dict,
) -> dict:
    episodes = jsonl(path)
    assert len(episodes) == 800, (path, len(episodes), 800)
    keys = {
        (item["task"], item["initialization_id"], item["condition"])
        for item in episodes
    }
    assert len(keys) == 800
    assert {item["task"] for item in episodes} == {
        "pick_can", "move_near", "open_drawer", "close_drawer",
    }
    assert {item["condition"] for item in episodes} == {
        "native_singleton", "native_dynamic",
        "patched_singleton", "patched_dynamic",
    }
    expected_conditions = {
        "native_singleton": ("native", False, [1]),
        "native_dynamic": ("native", True, [1, 2, 4, 8]),
        "patched_singleton": ("full_invariant", False, [1]),
        "patched_dynamic": ("full_invariant", True, [1, 2, 4, 8]),
    }
    assert all(
        item["numerical_policy_sha"] == freeze["batch_invariant_pizero_sha"]
        and item["batch_invariant_ops_sha"] == freeze["batch_invariant_ops_sha"]
        and item["checkpoint_sha256"] == freeze["checkpoint"]["sha256"]
        for item in episodes
    )
    assert all(
        item["schema_version"] == 1
        and str(UUID(item["experiment_id"])) == item["experiment_id"]
        and item["request_id"]
        == (
            f"{item['task']}/initialization-{item['initialization_id']:03d}/"
            f"condition-{item['condition']}"
        )
        and item["episode_id"] == item["initialization_id"]
        and item["dtype"] == "float32"
        and item["tf32"] is False
        and item["cudnn_tf32"] is True
        and item["execution_mode"] == "eager"
        and item["action_sequence"]
        and len(item["trajectory"]) == len(item["action_sequence"]) + 1
        for item in episodes
    )
    for episode in episodes:
        calls = {item["policy_call"] for item in episode["action_sequence"]}
        assert calls == set(range(episode["policy_call_count"]))
        for action in episode["action_sequence"]:
            expected_batch_size = (
                (1, 2, 4, 8)[action["policy_call"] % 4]
                if episode["dynamic_batching"] else 1
            )
            assert action["batch_size"] == expected_batch_size
            assert action["target_position"] == action["policy_call"] % expected_batch_size
            expected_noise_id = (
                f"{episode['task']}/initialization-{episode['initialization_id']}/"
                f"policy-call-{action['policy_call']}"
            )
            assert action["noise_id"] == expected_noise_id
            assert len(action["companion_request_ids"]) == expected_batch_size - 1
            assert len(action["request_ordering"]) == expected_batch_size
            assert action["request_ordering"][action["target_position"]] == expected_noise_id
            assert len(action["normalized_action"]) == 7
            assert len(action["environment_action"]) == 7
            assert math.isfinite(action["reward"])
            assert all(math.isfinite(value) for value in action["normalized_action"])
            assert all(math.isfinite(value) for value in action["environment_action"])
    assert all(
        (
            item["implementation"],
            item["dynamic_batching"],
            item["dynamic_batch_cycle"],
        )
        == expected_conditions[item["condition"]]
        and item["initialization_seed"] == 20250311 + item["initialization_id"]
        and item["policy_noise_scheme"]
        == "sha256-indexed-by-task-initialization-policy-call"
        and item["trajectory"]
        for item in episodes
    )
    for task in {item["task"] for item in episodes}:
        for condition in expected_conditions:
            assert {
                item["initialization_id"]
                for item in episodes
                if item["task"] == task and item["condition"] == condition
            } == set(range(50))
    runtime = json.loads(runtime_path.read_text())
    assert runtime["campaign"] == "paired_closed_loop_simpler"
    assert runtime["expected_episodes"] == 800
    assert runtime["initializations_per_task"] == 50
    assert runtime["seed"] == 20250311
    assert runtime["tasks"] == sorted({item["task"] for item in episodes})
    assert runtime["numerical_policy_sha"] == freeze["batch_invariant_pizero_sha"]
    assert runtime["frozen_operator_baseline_sha"] == freeze["batch_invariant_ops_sha"]
    assert runtime["checkpoint_sha256"] == freeze["checkpoint"]["sha256"]
    assert runtime["rtx_launch_adaptation_sha"] == rtx_environment["repositories"]["batch_invariant_ops"]["git_sha"]
    assert runtime["replay_manifest_sha256"] == sha256_file(
        RESULTS / "replay_manifest.json"
    )
    assert runtime["rtx_environment_sha256"] == sha256_file(
        RESULTS / "rtx_environment.json"
    )
    assert runtime["repositories"] == {
        "simpler_env_sha": "59ad9e1539042ed333fd8ebba1b0395f5662f0bd",
        "maniskill2_real2sim_sha": "91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c",
    }
    source_directory = ROOT / "experiments"
    assert runtime["source_sha256"] == {
        name: sha256_file(source_directory / name)
        for name in (
            "simpler_eval.py",
            "simpler_worker.py",
            "simpler_support.py",
            "analyze_simpler.py",
        )
    }
    summary = json.loads(summary_path.read_text())
    assert summary["bootstrap"] == {
        "seed": 20250401,
        "resamples": 10000,
        "unit": "matched episode initialization",
    }
    assert set(summary["tasks"]) == {item["task"] for item in episodes}
    assert all(item["episodes"] == 50 for item in summary["tasks"].values())
    expected_comparisons = {
        "native_dynamic_minus_singleton",
        "patched_dynamic_minus_singleton",
        "patched_minus_native_singleton",
    }
    for task_index, task in enumerate(sorted(summary["tasks"])):
        task_summary = summary["tasks"][task]
        assert set(task_summary["success_rate"]) == set(expected_conditions)
        assert all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in task_summary["success_rate"].values()
        )
        assert set(task_summary["paired_differences"]) == expected_comparisons
        assert set(task_summary["trajectory_pairs"]) == expected_comparisons
        for comparison in expected_comparisons:
            paired = task_summary["paired_differences"][comparison]
            assert paired["bootstrap_seed"] == 20250401 + task_index
            assert paired["bootstrap_resamples"] == 10000
            assert paired["unit"] == "episode"
            assert 0 <= paired["disagreement_count"] <= 50
            assert math.isfinite(paired["estimate"])
            assert len(paired["ci95"]) == 2
            assert all(math.isfinite(value) for value in paired["ci95"])
            trajectories = task_summary["trajectory_pairs"][comparison]
            assert len(trajectories) == 50
            assert all(
                item["aligned_state_count"] > 0
                and all(
                    math.isfinite(item[key]) and item[key] >= 0.0
                    for key in (
                        "end_effector_position_max_m",
                        "end_effector_position_terminal_m",
                        "end_effector_rotation_max_rad",
                        "end_effector_rotation_terminal_rad",
                        "gripper_max_native_units",
                        "gripper_terminal_native_units",
                    )
                )
                for item in trajectories
            )
    return {
        "episodes": len(episodes),
        "matched_initializations_per_task": 50,
        "conditions": 4,
        "bootstrap": summary["bootstrap"],
        "runtime_provenance": {
            "evaluation_harness_sha": runtime["evaluation_harness_sha"],
            "rtx_launch_adaptation_sha": runtime["rtx_launch_adaptation_sha"],
            "simpler_env_sha": runtime["repositories"]["simpler_env_sha"],
            "maniskill2_real2sim_sha": runtime["repositories"]["maniskill2_real2sim_sha"],
        },
    }


def main() -> None:
    operator = jsonl(RESULTS / "operator" / "raw.jsonl")
    assert len(operator) == 72_000
    operator_groups = defaultdict(list)
    for item in operator:
        key = (
            item["operator"], item["implementation"], item["batch_size"],
            item["dtype"], item["layout"],
        )
        operator_groups[key].append(item)
    assert all(len({item["seed"] for item in items}) == 100 for items in operator_groups.values())
    invariant_operator = [
        item for item in operator if item["implementation"] == "full_invariant"
    ]
    assert len(invariant_operator) == 36_000
    assert all(item["invariance"]["exact"] for item in invariant_operator)
    dispatch = json.loads((RESULTS / "operator" / "dispatch_report.json").read_text())
    assert {item["source_operation"] for item in dispatch} == {
        "rank2_mm", "rank3_attention", "siglip_projection", "rmsnorm_mean",
        "qualified_log_softmax", "attention_softmax",
    }
    assert all(
        item["invariant_path_selected"]
        for item in dispatch if item["source_operation"] != "attention_softmax"
    )
    softmax = next(
        item for item in dispatch if item["source_operation"] == "attention_softmax"
    )
    assert softmax["path_classification"] == "audited_native_batch_local"
    assert softmax["native_batch_invariance"]["exact"]

    transformations = jsonl(RESULTS / "diagnostic" / "batch_transformations.jsonl")
    assert len(transformations) == 20
    full_transformations = [
        item for item in transformations if item["implementation"] == "full_invariant"
    ]
    assert len(full_transformations) == 10
    assert all(item["output"]["exact"] for item in full_transformations)

    kernel = jsonl(RESULTS / "performance" / "kernel_timings.jsonl")
    policy_timing = jsonl(RESULTS / "performance" / "policy_timings.jsonl")
    serving = jsonl(RESULTS / "serving" / "requests.jsonl")
    assert len(kernel) == 3_600
    assert len(policy_timing) == 4_000
    assert len(serving) == 4_000
    serving_summary = json.loads((RESULTS / "serving" / "summary.json").read_text())
    assert len(serving_summary) == 20

    environment = json.loads((RESULTS / "environment.json").read_text())
    rtx_environment = json.loads((RESULTS / "rtx_environment.json").read_text())
    heldout_runtime = json.loads((RESULTS / "heldout" / "runtime.json").read_text())
    freeze = json.loads((RESULTS / "numerical_freeze.json").read_text())
    assert freeze["numerical_implementation_frozen"] and not freeze["dirty_status"]
    replay = json.loads((RESULTS / "replay_manifest.json").read_text())
    assert replay["counts"] == {
        "episodes": 100,
        "observations": 1000,
        "diagnostic_observations": 200,
        "heldout_observations": 800,
        "request_noise_pairs": 3000,
    }
    assert replay["provenance"]["checkpoint_sha256"] == freeze["checkpoint"]["sha256"]
    assert heldout_runtime["numerical_policy_sha"] == freeze["batch_invariant_pizero_sha"]
    assert heldout_runtime["frozen_operator_baseline_sha"] == freeze["batch_invariant_ops_sha"]
    assert heldout_runtime["checkpoint_sha256"] == freeze["checkpoint"]["sha256"]
    assert heldout_runtime["rtx_launch_adaptation_sha"] == rtx_environment["repositories"]["batch_invariant_ops"]["git_sha"]
    assert heldout_runtime["replay_manifest_sha256"] == sha256_file(
        RESULTS / "replay_manifest.json"
    )
    assert heldout_runtime["rtx_environment_sha256"] == sha256_file(
        RESULTS / "rtx_environment.json"
    )
    assert all(Path(item["tensor_path"]).is_file() for item in replay["observations"])
    heldout = heldout_audit(
        RESULTS / "heldout" / "invariance.jsonl",
        RESULTS / "heldout" / "singleton_fidelity.jsonl",
        freeze,
    )
    diagnostic_replay = diagnostic_replay_audit(
        RESULTS / "diagnostic" / "simpler_invariance.jsonl",
        RESULTS / "diagnostic" / "simpler_singleton_fidelity.jsonl",
        freeze,
    )
    simpler = simpler_audit(
        RESULTS / "simpler" / "episodes.jsonl",
        RESULTS / "simpler" / "summary.json",
        RESULTS / "simpler" / "runtime.json",
        freeze,
        rtx_environment,
    )
    flow_summary = flow_summary_audit(
        RESULTS / "heldout" / "flow_step_summary.json"
    )
    diagnostic_flow_summary = diagnostic_flow_summary_audit(
        RESULTS / "diagnostic" / "flow_step_summary.json",
        RESULTS / "diagnostic" / "first_divergence.json",
    )
    historical_blocker = json.loads((RESULTS / "simpler" / "blocker.json").read_text())
    for relative in (
        "artifacts/figures/flow_step_propagation.png",
        "artifacts/figures/flow_step_propagation_heldout.png",
        "artifacts/figures/serving_tradeoff.png",
        "artifacts/tables/operator_projection.csv",
        "artifacts/tables/policy_ablation.csv",
        "artifacts/tables/policy_heldout.csv",
        "artifacts/tables/simpler.csv",
        "results/heldout/flow_step_summary.json",
        "results/diagnostic/flow_step_summary.json",
        "results/simpler/summary.json",
        "results/EXPERIMENT_REPORT.md",
    ):
        assert (ROOT / relative).stat().st_size > 0
    assert len((ROOT / "artifacts/tables/operator_projection.csv").read_text().splitlines()) == 721
    assert len((ROOT / "artifacts/tables/policy_ablation.csv").read_text().splitlines()) == 13
    assert len((ROOT / "artifacts/tables/policy_heldout.csv").read_text().splitlines()) == 3
    assert len((ROOT / "artifacts/tables/simpler.csv").read_text().splitlines()) == 5
    report = (RESULTS / "EXPERIMENT_REPORT.md").read_text()
    assert all(f"## {letter}." in report for letter in "ABCDEFGH")
    assert "0/800 planned episodes were executed" not in report
    assert "replay dataset could not be collected" not in report
    assert "First-divergence category frequencies" in report
    assert "Trajectory formulas:" in report
    assert "Persistent-vs-explicit invariant kernel differences:" in report
    assert "Complete-policy invariant-vs-native dynamic differences" in report

    audit = {
        "schema_version": 1,
        "audited_at": utc_timestamp(),
        "status": "complete_campaign_verified",
        "operator": {
            "records": len(operator),
            "configurations": len(operator_groups),
            "full_invariant_records": len(invariant_operator),
            "full_invariant_failures": sum(
                not item["invariance"]["exact"] for item in invariant_operator
            ),
            "dispatch_cases": len(dispatch),
        },
        "policy": {
            "float32_b2_b8": policy_audit(
                RESULTS / "diagnostic" / "policy_ablation.jsonl", 114
            ),
            "float32_b16_b32": policy_audit(
                RESULTS / "diagnostic" / "policy_ablation_large_batches.jsonl", 72
            ),
            "bfloat16_b2_b8": policy_audit(
                RESULTS / "diagnostic" / "policy_ablation_bfloat16.jsonl", 114
            ),
            "transformation_records": len(transformations),
            "full_invariant_transformation_failures": sum(
                not item["output"]["exact"] for item in full_transformations
            ),
            "heldout": heldout,
            "rtx_diagnostic": diagnostic_replay,
            "heldout_flow_steps": {
                implementation: {
                    "steps": len(steps),
                    "comparisons_per_step": next(iter(steps.values()))["comparisons"],
                    "exact_per_step": [item["exact"] for item in steps.values()],
                }
                for implementation, steps in flow_summary.items()
            },
            "diagnostic_flow_steps": {
                "scope": diagnostic_flow_summary["scope"],
                "implementations": sorted(
                    diagnostic_flow_summary["implementations"]
                ),
                "steps_per_implementation": 11,
            },
        },
        "replay": replay["counts"],
        "simpler": simpler,
        "performance": {
            "kernel_timing_samples": len(kernel),
            "policy_timing_samples": len(policy_timing),
            "serving_requests": len(serving),
            "serving_load_points": len(serving_summary),
        },
        "provenance": {
            "environment_policy_sha": environment["repositories"]["batch_invariant_pizero"]["git_sha"],
            "numerical_policy_sha": freeze["batch_invariant_pizero_sha"],
            "operator_sha": freeze["batch_invariant_ops_sha"],
            "checkpoint_sha256": freeze["checkpoint"]["sha256"],
            "rtx_environment_policy_sha": rtx_environment["repositories"]["batch_invariant_pizero"]["git_sha"],
            "rtx_runtime_operator_sha": rtx_environment["repositories"]["batch_invariant_ops"]["git_sha"],
            "heldout_evaluation_harness_sha": heldout_runtime["evaluation_harness_sha"],
            "simpler_evaluation_harness_sha": simpler["runtime_provenance"]["evaluation_harness_sha"],
        },
        "historical_h100_simulator_blocker": historical_blocker,
    }
    write_json(RESULTS / "audit.json", audit)
    print(RESULTS / "audit.json")


if __name__ == "__main__":
    main()
