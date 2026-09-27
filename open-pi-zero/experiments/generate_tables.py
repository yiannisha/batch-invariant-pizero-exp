#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


LABELS = {
    "native": "Native",
    "native_deterministic": "Native + deterministic settings",
    "existing_invariant_ops": "Existing invariant operators",
    "invariant_plus_patch_projection": "Operators + patch projection",
    "explicit_per_matrix_attention": "Explicit per-matrix reference",
    "full_invariant": "Persistent BMM + audited path",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metric(record: dict) -> dict:
    return record.get("output", record)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "No measurements available."
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def operator_rows(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        key = (
            record["operator"], record["implementation"], record["batch_size"],
            record["dtype"], record["layout"],
        )
        groups[key].append(record)
    rows = []
    for key, values in sorted(groups.items()):
        operation, implementation, batch_size, dtype, layout = key
        rows.append(
            {
                "Operator": operation,
                "Implementation": implementation,
                "Batch size": batch_size,
                "Precision": dtype,
                "Layout": layout,
                "Max singleton-vs-batched difference": max(
                    item["invariance"]["max_absolute_error"] for item in values
                ),
                "Failure count": sum(
                    not item["invariance"]["exact"] for item in values
                ),
                "Inputs": len(values),
            }
        )
    return rows


def policy_rows(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        if record.get("transformation") == "singleton" or record.get("batch_size") == 1:
            continue
        groups[(record["implementation"], record.get("dtype", "unknown"))].append(record)
    rows = []
    for dtype in sorted({key[1] for key in groups}):
        for implementation in LABELS:
            values = groups.get((implementation, dtype), [])
            if not values:
                continue
            violations = [item for item in values if not metric(item)["exact"]]
            request_violations = {
                item["request_id"] for item in violations
            }
            requests = {item["request_id"] for item in values}
            rows.append(
                {
                    "Implementation": LABELS[implementation],
                    "Precision": dtype,
                    "Violations (%)": f"{100 * len(request_violations) / len(requests):.3f}",
                    "Exact violation count": len(request_violations),
                    "Arrangement violations": len(violations),
                    "Arrangements": len(values),
                    "Max action error": max(
                        metric(item)["max_absolute_error"] for item in values
                    ),
                }
            )
    return rows


def simpler_rows(records: list[dict]) -> list[dict]:
    rows = []
    for task in ("pick_can", "move_near", "open_drawer", "close_drawer"):
        task_records = [item for item in records if item["task"] == task]
        by_condition = defaultdict(list)
        for item in task_records:
            by_condition[item["condition"]].append(bool(item["success"]))
        native_single = {
            item["initialization_id"]: bool(item["success"])
            for item in task_records if item["condition"] == "native_singleton"
        }
        native_dynamic = {
            item["initialization_id"]: bool(item["success"])
            for item in task_records if item["condition"] == "native_dynamic"
        }
        patched_single = {
            item["initialization_id"]: bool(item["success"])
            for item in task_records if item["condition"] == "patched_singleton"
        }
        patched_dynamic = {
            item["initialization_id"]: bool(item["success"])
            for item in task_records if item["condition"] == "patched_dynamic"
        }
        rows.append(
            {
                "Task": task.replace("_", " ").title(),
                "Native dynamic success": statistics.mean(by_condition["native_dynamic"]) if by_condition["native_dynamic"] else "NA",
                "Patched dynamic success": statistics.mean(by_condition["patched_dynamic"]) if by_condition["patched_dynamic"] else "NA",
                "Native paired disagreements": sum(native_single.get(key) != value for key, value in native_dynamic.items()),
                "Patched paired disagreements": sum(patched_single.get(key) != value for key, value in patched_dynamic.items()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", type=Path, default=Path("results/operator/raw.jsonl"))
    parser.add_argument("--policy", type=Path, default=Path("results/heldout/invariance.jsonl"))
    parser.add_argument("--diagnostic-policy", type=Path, default=Path("results/diagnostic/policy_ablation.jsonl"))
    parser.add_argument("--simpler", type=Path, default=Path("results/simpler/episodes.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tables"))
    args = parser.parse_args()
    operators = operator_rows(read_jsonl(args.operator))
    policy_source = read_jsonl(args.policy)
    if not policy_source:
        policy_source = read_jsonl(args.diagnostic_policy)
        policy_source += read_jsonl(Path("results/diagnostic/policy_ablation_large_batches.jsonl"))
        policy_source += read_jsonl(Path("results/diagnostic/policy_ablation_bfloat16.jsonl"))
    policies = policy_rows(policy_source)
    simpler = simpler_rows(read_jsonl(args.simpler))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("operator_projection", operators),
        ("policy_ablation", policies),
        ("simpler", simpler),
    ):
        write_csv(args.output_dir / f"{name}.csv", rows)
        (args.output_dir / f"{name}.md").write_text(markdown_table(rows) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
