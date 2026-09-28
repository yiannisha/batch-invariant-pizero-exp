#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.common import sha256_file, utc_timestamp, write_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


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
    assert all(len(items) == 23 for items in groups.values())
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


def simpler_audit(path: Path, summary_path: Path, freeze: dict) -> dict:
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
    assert all(
        item["numerical_policy_sha"] == freeze["batch_invariant_pizero_sha"]
        and item["batch_invariant_ops_sha"] == freeze["batch_invariant_ops_sha"]
        and item["checkpoint_sha256"] == freeze["checkpoint"]["sha256"]
        for item in episodes
    )
    summary = json.loads(summary_path.read_text())
    assert summary["bootstrap"] == {
        "seed": 20250401,
        "resamples": 10000,
        "unit": "matched episode initialization",
    }
    assert set(summary["tasks"]) == {item["task"] for item in episodes}
    assert all(item["episodes"] == 50 for item in summary["tasks"].values())
    return {
        "episodes": len(episodes),
        "matched_initializations_per_task": 50,
        "conditions": 4,
        "bootstrap": summary["bootstrap"],
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
    simpler = simpler_audit(
        RESULTS / "simpler" / "episodes.jsonl",
        RESULTS / "simpler" / "summary.json",
        freeze,
    )
    historical_blocker = json.loads((RESULTS / "simpler" / "blocker.json").read_text())
    for relative in (
        "artifacts/figures/flow_step_propagation.png",
        "artifacts/figures/serving_tradeoff.png",
        "artifacts/tables/operator_projection.csv",
        "artifacts/tables/policy_ablation.csv",
        "artifacts/tables/simpler.csv",
        "results/heldout/flow_step_summary.json",
        "results/simpler/summary.json",
        "results/EXPERIMENT_REPORT.md",
    ):
        assert (ROOT / relative).stat().st_size > 0

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
        },
        "historical_h100_simulator_blocker": historical_blocker,
    }
    write_json(RESULTS / "audit.json", audit)
    print(RESULTS / "audit.json")


if __name__ == "__main__":
    main()
