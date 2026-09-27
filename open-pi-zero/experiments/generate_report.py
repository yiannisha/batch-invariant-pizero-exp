#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FLOW = re.compile(r"flow\.step_(\d+)\.updated_action_state")


def load_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metric(record: dict) -> dict:
    return record.get("output", record)


def fmt(value: float) -> str:
    return f"{value:.9g}"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def policy_summary(records: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for record in records:
        if record.get("transformation") == "singleton" or record.get("batch_size") == 1:
            continue
        grouped[record["implementation"]].append(record)
    result = {}
    for implementation, values in grouped.items():
        violations = [item for item in values if not metric(item).get("exact", False)]
        result[implementation] = {
            "arrangements": len(values),
            "arrangement_violations": len(violations),
            "requests": len({item["request_id"] for item in values}),
            "request_violations": len({item["request_id"] for item in violations}),
            "max_error": max(metric(item)["max_absolute_error"] for item in values),
        }
    return result


def main() -> None:
    environment = load_json(RESULTS / "environment.json", {})
    freeze = load_json(RESULTS / "numerical_freeze.json", {})
    heldout = load_jsonl(RESULTS / "heldout" / "invariance.jsonl")
    diagnostic = load_jsonl(RESULTS / "diagnostic" / "policy_ablation.jsonl")
    policies = policy_summary(heldout or diagnostic)
    scope = "held-out replay" if heldout else "synthetic diagnostic"
    trace = load_json(RESULTS / "diagnostic" / "first_divergence.json", {})
    fidelity = load_json(RESULTS / "diagnostic" / "singleton_fidelity.json")
    operator = load_jsonl(RESULTS / "operator" / "raw.jsonl")
    kernel = load_json(RESULTS / "performance" / "kernel_summary.json", [])
    policy_perf = load_json(RESULTS / "performance" / "policy_summary.json", [])
    serving = load_json(RESULTS / "serving" / "summary.json", [])
    simpler_blocker = load_json(RESULTS / "simpler" / "blocker.json")

    lines = [
        "# Experimental report",
        "",
        "This report is generated from the retained machine-readable measurements. "
        "Missing campaigns are marked unavailable rather than estimated.",
        "",
        "## Provenance",
        "",
        f"- Policy revision: `{freeze.get('batch_invariant_pizero_sha') or environment.get('repositories', {}).get('batch_invariant_pizero', {}).get('git_sha', 'unavailable')}`",
        f"- Operator revision: `{freeze.get('batch_invariant_ops_sha') or environment.get('repositories', {}).get('batch_invariant_ops', {}).get('git_sha', 'unavailable')}`",
        f"- Checkpoint SHA-256: `{environment.get('checkpoint', {}).get('sha256', 'unavailable')}`",
        f"- Evaluation scope used below: **{scope}**",
        "",
        "## A. Does native inference violate request-level batch invariance?",
        "",
    ]
    native = policies.get("native")
    if native:
        rate = 100 * native["request_violations"] / native["requests"]
        arrangement_rate = 100 * native["arrangement_violations"] / native["arrangements"]
        lines += [
            f"Yes within the {scope} campaign: {native['request_violations']}/{native['requests']} "
            f"requests violated ({rate:.3f}%), and {native['arrangement_violations']}/"
            f"{native['arrangements']} non-singleton arrangements violated ({arrangement_rate:.3f}%). "
            f"The maximum normalized action error was {fmt(native['max_error'])}.",
        ]
        native_records = [
            item for item in (heldout or diagnostic)
            if item["implementation"] == "native" and item.get("batch_size") != 1
        ]
        errors = [metric(item)["max_absolute_error"] for item in native_records]
        transformations = sorted({
            item.get("transformation", "batch_position") for item in native_records
        })
        batch_sizes = sorted({item["batch_size"] for item in native_records})
        positions = sorted({str(item["target_batch_position"]) for item in native_records})
        companions = sorted({item["companion_type"] for item in native_records})
        lines += [
            f"Tested transformations={transformations}, batch sizes={batch_sizes}, target "
            f"positions={positions}, companions={companions}. Across arrangement-level maximum "
            f"errors: median={fmt(statistics.median(errors))}, p95={fmt(percentile(errors, 0.95))}, "
            f"p99={fmt(percentile(errors, 0.99))}."
        ]
    else:
        lines += ["Unavailable: no policy-level records were produced."]
    if not heldout:
        lines += [
            "The required 2,400-request held-out result is unavailable because the replay dataset "
            "could not be collected in this container; this diagnostic result must not be read as a held-out rate."
        ]

    lines += ["", "## B. Where does native execution first diverge?", ""]
    if trace:
        for implementation, payload in trace.items():
            first = payload.get("first_divergence")
            boundary = first.get("trace_key") if first else "none (all traced tensors exact)"
            lines.append(f"- `{implementation}`: {boundary}")
    else:
        lines.append("Unavailable: no invocation-aware trace report was produced.")

    lines += ["", "## C. How do differences propagate through the flow solver?", ""]
    if trace:
        for implementation in (
            "native", "existing_invariant_ops", "invariant_plus_patch_projection", "full_invariant"
        ):
            if implementation not in trace:
                continue
            points = {}
            for item in trace[implementation].get("comparisons", []):
                match = FLOW.match(item["trace_key"])
                if match:
                    points[int(match.group(1)) + 1] = item["max_absolute_error"]
            values = ", ".join(f"{step}:{fmt(error)}" for step, error in sorted(points.items()))
            lines.append(f"- `{implementation}` step:error — {values or 'unavailable'}")
    else:
        lines.append("Unavailable.")

    lines += ["", "## D. Does the full invariant path achieve exact action equality?", ""]
    full = policies.get("full_invariant")
    if full:
        lines.append(
            f"In the {scope} records, {full['arrangement_violations']}/{full['arrangements']} "
            f"non-singleton arrangements failed exact equality; maximum error was {fmt(full['max_error'])}."
        )
    else:
        lines.append("Unavailable.")

    lines += ["", "## E. How faithful is invariant singleton inference to native singleton inference?", ""]
    if fidelity:
        item = fidelity["metrics"]
        lines.append(
            f"For the frozen synthetic diagnostic singleton: exact={item['exact']}, "
            f"max absolute error={fmt(item['max_absolute_error'])}, RMSE={fmt(item['rmse'])}, "
            f"relative L2={fmt(item['relative_l2_error'])}. This fidelity comparison is distinct "
            "from batch invariance."
        )
    else:
        lines.append("Unavailable.")

    lines += ["", "## F. Do numerical differences affect behavior?", ""]
    if native:
        lines.append(
            f"- Action divergence: measured in the {scope} numerical campaign; "
            f"{native['arrangement_violations']}/{native['arrangements']} native arrangements differed."
        )
    if simpler_blocker:
        lines += [
            "- Trajectory divergence: unavailable; no SIMPLER episode could reach `env.reset`.",
            "- Paired success disagreement: unavailable; 0/800 planned episodes were executed.",
            "- Success-rate differences: unavailable; no behavioral outcome is inferred from the numerical results.",
            "The container exposed CUDA compute but not the NVIDIA graphics/Vulkan ICD required by "
            "SAPIEN 2.2.2; native rendering segfaulted and the Lavapipe fallback lacked a required "
            "Vulkan extension. Full evidence is retained in `results/simpler/blocker.json`.",
        ]
    else:
        lines.append("Unavailable: no SIMPLER episode records were produced.")

    lines += ["", "## G. What is the computational cost?", ""]
    if kernel:
        for item in kernel:
            if item["implementation"] in {"explicit_invariant_mm", "persistent_invariant_bmm"}:
                lines.append(
                    f"- B={item['batch_size']} `{item['implementation']}`: median "
                    f"{fmt(item['median_latency_ms'])} ms, p95 {fmt(item['p95_latency_ms'])} ms, "
                    f"{item['kernel_launch_count']} profiled device launches."
                )
    else:
        lines.append("Kernel timings unavailable.")
    if policy_perf:
        for item in policy_perf:
            lines.append(
                f"- `{item['configuration']}`: median {fmt(item['median_latency_ms'])} ms, "
                f"p95 {fmt(item['p95_latency_ms'])} ms, {fmt(item['requests_per_second'])} requests/s, "
                f"peak {item['peak_allocated_bytes']} bytes, launches {item['kernel_launch_count']}."
            )
    else:
        lines.append("Complete-policy timings unavailable.")

    lines += ["", "## H. How does invariant batching compare with practical alternatives?", ""]
    if serving:
        grouped = Counter(item["configuration"] for item in serving)
        for configuration, count in sorted(grouped.items()):
            subset = [item for item in serving if item["configuration"] == configuration]
            lines.append(
                f"- `{configuration}`: {count} offered-load points; numerical contract satisfied "
                f"at {sum(item['numerical_contract_satisfied'] for item in subset)}/{count} points. "
                + "; ".join(
                    f"load {item['offered_load']:g}: {fmt(item['throughput_requests_per_second'])} req/s, "
                    f"p95 {fmt(item['p95_arrival_to_completion_ms'])} ms"
                    for item in subset
                )
            )
    else:
        lines.append("Serving/load measurements unavailable.")

    lines += ["", "## Operator qualification", ""]
    if operator:
        invariant = [item for item in operator if item["implementation"] == "full_invariant"]
        lines.append(
            f"Retained {len(operator)} operator records. The full-invariant path had "
            f"{sum(not item['invariance']['exact'] for item in invariant)} batch-invariance failures "
            f"across {len(invariant)} records."
        )
    else:
        lines.append("Operator campaign unavailable.")

    output = RESULTS / "EXPERIMENT_REPORT.md"
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
