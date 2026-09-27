#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.common import utc_timestamp, write_json


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
    freeze = json.loads((RESULTS / "numerical_freeze.json").read_text())
    assert freeze["numerical_implementation_frozen"] and not freeze["dirty_status"]
    assert (RESULTS / "simpler" / "blocker.json").exists()
    for relative in (
        "artifacts/figures/flow_step_propagation.png",
        "artifacts/figures/serving_tradeoff.png",
        "results/EXPERIMENT_REPORT.md",
    ):
        assert (ROOT / relative).stat().st_size > 0

    audit = {
        "schema_version": 1,
        "audited_at": utc_timestamp(),
        "status": "all_runnable_campaigns_verified",
        "operator": {
            "records": len(operator),
            "configurations": len(operator_groups),
            "full_invariant_records": len(invariant_operator),
            "full_invariant_failures": sum(
                not item["invariance"]["exact"] for item in invariant_operator
            ),
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
        },
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
        },
        "blocked": json.loads((RESULTS / "simpler" / "blocker.json").read_text()),
    }
    write_json(RESULTS / "audit.json", audit)
    print(RESULTS / "audit.json")


if __name__ == "__main__":
    main()
