#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


FLOW_PATTERN = re.compile(r"flow\.step_(\d+)\.updated_action_state")


def flow_figure(trace_path: Path, output: Path) -> None:
    traces = json.loads(trace_path.read_text())
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for implementation, payload in traces.items():
        points = {}
        for item in payload["comparisons"]:
            match = FLOW_PATTERN.match(item["trace_key"])
            if match:
                points[int(match.group(1)) + 1] = item["max_absolute_error"]
        points[0] = 0.0
        if points:
            x = sorted(points)
            axis.plot(x, [points[index] for index in x], marker="o", label=implementation)
    axis.set_xlabel("Flow step")
    axis.set_ylabel("Maximum absolute action-state error")
    axis.set_xticks(range(11))
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def serving_figure(summary_path: Path, output: Path) -> None:
    summaries = json.loads(summary_path.read_text())
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for record in summaries:
        style = "o" if record["numerical_contract_satisfied"] else "x"
        axis.scatter(
            record["throughput_requests_per_second"],
            record["p95_arrival_to_completion_ms"],
            marker=style,
            label=f"{record['configuration']} @ {record['offered_load']}",
        )
    axis.set_xlabel("Throughput (requests/s)")
    axis.set_ylabel("p95 arrival-to-completion latency (ms)")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("results/diagnostic/pretrained_first_divergence.json"))
    parser.add_argument("--serving", type=Path, default=Path("results/serving/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()
    if args.trace.exists():
        flow_figure(args.trace, args.output_dir / "flow_step_propagation.png")
    if args.serving.exists():
        serving_figure(args.serving, args.output_dir / "serving_tradeoff.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
