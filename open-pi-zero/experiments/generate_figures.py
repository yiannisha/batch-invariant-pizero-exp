#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FLOW_PATTERN = re.compile(r"flow\.step_(\d+)\.updated_action_state")
FLOW_IMPLEMENTATIONS = (
    "native",
    "existing_invariant_ops",
    "invariant_plus_patch_projection",
    "full_invariant",
)
FLOW_LABELS = {
    "native": "Native",
    "existing_invariant_ops": "Existing invariant operators",
    "invariant_plus_patch_projection": "Operators + patch projection",
    "full_invariant": "Complete invariant path",
}
FLOW_MARKERS = {
    "native": "o",
    "existing_invariant_ops": "s",
    "invariant_plus_patch_projection": "^",
    "full_invariant": "D",
}


def flow_figure(trace_path: Path, output: Path, summary_path: Path) -> None:
    traces = json.loads(trace_path.read_text())
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    summary = {
        "scope": "single frozen B=2 diverse diagnostic arrangement",
        "central_statistic": "mean absolute error across action-state elements",
        "tail_statistic": "maximum absolute error across action-state elements",
        "implementations": {},
    }
    for implementation in FLOW_IMPLEMENTATIONS:
        payload = traces[implementation]
        points = {
            0: {
                "exact": True,
                "mean_absolute_error": 0.0,
                "maximum_absolute_error": 0.0,
            }
        }
        for item in payload["comparisons"]:
            match = FLOW_PATTERN.match(item["trace_key"])
            if match:
                points[int(match.group(1)) + 1] = {
                    "exact": item["exact"],
                    "mean_absolute_error": item["mean_absolute_error"],
                    "maximum_absolute_error": item["max_absolute_error"],
                }
        if points:
            x = sorted(points)
            mean_line = axis.plot(
                x,
                [points[index]["mean_absolute_error"] for index in x],
                marker=FLOW_MARKERS[implementation],
                linewidth=3.0 if implementation == "existing_invariant_ops" else 1.5,
                markersize=7 if implementation == "existing_invariant_ops" else 5,
                label=f"{FLOW_LABELS[implementation]} mean",
            )[0]
            axis.plot(
                x,
                [points[index]["maximum_absolute_error"] for index in x],
                linestyle="--",
                color=mean_line.get_color(),
                linewidth=3.0 if implementation == "existing_invariant_ops" else 1.5,
                label=f"{FLOW_LABELS[implementation]} maximum",
            )
            summary["implementations"][implementation] = {
                str(index): points[index] for index in x
            }
    axis.set_xlabel("Flow step")
    axis.set_ylabel("Absolute action-state error")
    axis.set_title("Frozen B=2 diverse diagnostic arrangement")
    axis.set_xticks(range(11))
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def heldout_flow_figure(records_path: Path, output: Path, summary_path: Path) -> None:
    grouped = defaultdict(lambda: defaultdict(list))
    exact = defaultdict(lambda: defaultdict(int))
    with records_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["batch_size"] == 1:
                continue
            implementation = record["implementation"]
            for item in record.get("preclip_and_flow", []):
                match = FLOW_PATTERN.match(item["trace_key"])
                if not match:
                    continue
                step = int(match.group(1)) + 1
                grouped[implementation][step].append(item["max_absolute_error"])
                exact[implementation][step] += int(item["exact"])

    summary = {}
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for implementation, step_values in sorted(grouped.items()):
        steps = sorted(step_values)
        medians = [float(np.median(step_values[step])) for step in steps]
        p95 = [float(np.quantile(step_values[step], 0.95)) for step in steps]
        axis.plot(steps, medians, marker="o", label=f"{implementation} median")
        axis.plot(steps, p95, linestyle="--", label=f"{implementation} p95")
        summary[implementation] = {
            str(step): {
                "comparisons": len(step_values[step]),
                "exact": exact[implementation][step],
                "median_max_absolute_error": float(np.median(step_values[step])),
                "p95_max_absolute_error": float(np.quantile(step_values[step], 0.95)),
                "maximum_absolute_error": max(step_values[step]),
            }
            for step in steps
        }
    axis.set_xlabel("Flow step")
    axis.set_ylabel("Maximum absolute action-state error")
    axis.set_title("Held-out replay: all non-singleton arrangements")
    axis.set_xticks(range(1, 11))
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def serving_figure(summary_path: Path, output: Path) -> None:
    summaries = json.loads(summary_path.read_text())
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    grouped = defaultdict(list)
    for record in summaries:
        grouped[record["configuration"]].append(record)
    for configuration, records in sorted(grouped.items()):
        records.sort(key=lambda item: item["offered_load"])
        contract_states = {
            item["numerical_contract_satisfied"] for item in records
        }
        contract_label = (
            "pass" if contract_states == {True}
            else "fail" if contract_states == {False}
            else "mixed"
        )
        style = "o" if contract_label == "pass" else "X"
        line = axis.plot(
            [item["throughput_requests_per_second"] for item in records],
            [item["p95_arrival_to_completion_ms"] for item in records],
            marker=style,
            label=f"{configuration} (contract {contract_label})",
        )
        for item in records:
            axis.annotate(
                f"load={item['offered_load']:g}",
                (
                    item["throughput_requests_per_second"],
                    item["p95_arrival_to_completion_ms"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                color=line[0].get_color(),
            )
    axis.set_xlabel("Throughput (requests/s)")
    axis.set_ylabel("p95 arrival-to-completion latency (ms)")
    axis.set_title("Serving tradeoff by offered load")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("results/diagnostic/first_divergence.json"))
    parser.add_argument("--heldout", type=Path, default=Path("results/heldout/invariance.jsonl"))
    parser.add_argument(
        "--flow-summary",
        type=Path,
        default=Path("results/heldout/flow_step_summary.json"),
    )
    parser.add_argument(
        "--diagnostic-flow-summary",
        type=Path,
        default=Path("results/diagnostic/flow_step_summary.json"),
    )
    parser.add_argument("--serving", type=Path, default=Path("results/serving/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()
    if args.trace.exists():
        flow_figure(
            args.trace,
            args.output_dir / "flow_step_propagation.png",
            args.diagnostic_flow_summary,
        )
    if args.heldout.exists():
        heldout_flow_figure(
            args.heldout,
            args.output_dir / "flow_step_propagation_heldout.png",
            args.flow_summary,
        )
    if args.serving.exists():
        serving_figure(args.serving, args.output_dir / "serving_tradeoff.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
