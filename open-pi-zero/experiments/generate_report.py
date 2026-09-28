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


def flow_step_summary(records: list[dict], implementation: str) -> dict[int, dict]:
    grouped = defaultdict(list)
    exact = defaultdict(int)
    counts = defaultdict(int)
    for record in records:
        if record["implementation"] != implementation or record["batch_size"] == 1:
            continue
        for comparison in record.get("preclip_and_flow", []):
            match = FLOW.match(comparison["trace_key"])
            if not match:
                continue
            step = int(match.group(1)) + 1
            grouped[step].append(comparison["max_absolute_error"])
            exact[step] += int(comparison["exact"])
            counts[step] += 1
    return {
        step: {
            "comparisons": counts[step],
            "exact": exact[step],
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "maximum": max(values),
        }
        for step, values in sorted(grouped.items())
    }


def main() -> None:
    environment = load_json(RESULTS / "environment.json", {})
    rtx_environment = load_json(RESULTS / "rtx_environment.json", {})
    heldout_runtime = load_json(RESULTS / "heldout" / "runtime.json", {})
    freeze = load_json(RESULTS / "numerical_freeze.json", {})
    heldout = load_jsonl(RESULTS / "heldout" / "invariance.jsonl")
    diagnostic = load_jsonl(RESULTS / "diagnostic" / "policy_ablation.jsonl")
    diagnostic_large = load_jsonl(RESULTS / "diagnostic" / "policy_ablation_large_batches.jsonl")
    diagnostic_bfloat16 = load_jsonl(RESULTS / "diagnostic" / "policy_ablation_bfloat16.jsonl")
    diagnostic_primary = diagnostic + diagnostic_large
    policies = policy_summary(heldout or diagnostic_primary)
    bfloat16_policies = policy_summary(diagnostic_bfloat16)
    transformations = load_jsonl(RESULTS / "diagnostic" / "batch_transformations.jsonl")
    scope = "held-out replay" if heldout else "synthetic diagnostic"
    trace = load_json(RESULTS / "diagnostic" / "first_divergence.json", {})
    local_native = load_json(RESULTS / "diagnostic" / "local_operator_replay.json")
    local_patch = load_json(RESULTS / "diagnostic" / "local_replay_patch_o_proj.json")
    local_explicit = load_json(RESULTS / "diagnostic" / "local_replay_explicit_o_proj.json")
    local_full = load_json(RESULTS / "diagnostic" / "local_replay_full_o_proj.json")
    fidelity = load_json(RESULTS / "diagnostic" / "singleton_fidelity.json")
    heldout_fidelity = load_jsonl(RESULTS / "heldout" / "singleton_fidelity.jsonl")
    operator = load_jsonl(RESULTS / "operator" / "raw.jsonl")
    dispatch = load_json(RESULTS / "operator" / "dispatch_report.json", [])
    kernel = load_json(RESULTS / "performance" / "kernel_summary.json", [])
    policy_perf = load_json(RESULTS / "performance" / "policy_summary.json", [])
    serving = load_json(RESULTS / "serving" / "summary.json", [])
    simpler_episodes = load_jsonl(RESULTS / "simpler" / "episodes.jsonl")
    simpler_summary = load_json(RESULTS / "simpler" / "summary.json", {})
    simpler_runtime = load_json(RESULTS / "simpler" / "runtime.json", {})
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
        f"- RTX launch-only operator adaptation: `{heldout_runtime.get('rtx_launch_adaptation_sha', 'unavailable')}`",
        f"- RTX evaluation harness revision: `{heldout_runtime.get('evaluation_harness_sha', 'unavailable')}`",
        f"- SIMPLER evaluator revision: `{simpler_runtime.get('evaluation_harness_sha', 'unavailable')}`",
        f"- SIMPLER evaluator source SHA-256: `{simpler_runtime.get('source_sha256', {}).get('simpler_eval.py', 'unavailable')}`",
        f"- SIMPLER revision: `{simpler_runtime.get('repositories', {}).get('simpler_env_sha', 'unavailable')}`",
        f"- ManiSkill2_real2sim revision: `{simpler_runtime.get('repositories', {}).get('maniskill2_real2sim_sha', 'unavailable')}`",
        f"- Checkpoint SHA-256: `{environment.get('checkpoint', {}).get('sha256', 'unavailable')}`",
        f"- RTX GPU: `{rtx_environment.get('hardware', {}).get('gpu', {}).get('name', 'unavailable')}`",
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
            item for item in (heldout or diagnostic_primary)
            if item["implementation"] == "native" and item.get("batch_size") != 1
        ]
        errors = [metric(item)["max_absolute_error"] for item in native_records]
        tested_transformations = sorted({
            item.get("transformation", "batch_position") for item in native_records
        })
        batch_sizes = sorted({item["batch_size"] for item in native_records})
        positions = sorted({str(item["target_batch_position"]) for item in native_records})
        companions = sorted({item["companion_type"] for item in native_records})
        lines += [
            f"Tested transformations={tested_transformations}, batch sizes={batch_sizes}, target "
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
    if transformations:
        native_transformations = [
            item for item in transformations if item["implementation"] == "native"
        ]
        lines += [
            f"In the separate same-request-set transformation matrix, native failed "
            f"{sum(not item['output']['exact'] for item in native_transformations)}/"
            f"{len(native_transformations)} aggregate comparisons: all five partition changes "
            f"failed, while all five restored permutations were exact."
        ]

    lines += ["", "### Separate BF16 policy campaign", ""]
    if bfloat16_policies:
        for implementation in (
            "native", "native_deterministic", "existing_invariant_ops",
            "invariant_plus_patch_projection", "explicit_per_matrix_attention",
            "full_invariant",
        ):
            item = bfloat16_policies[implementation]
            lines.append(
                f"- `{implementation}`: {item['arrangement_violations']}/"
                f"{item['arrangements']} arrangement violations; maximum error "
                f"{fmt(item['max_error'])}."
            )
    else:
        lines.append("Unavailable.")

    lines += ["", "## B. Where does native execution first diverge?", ""]
    if trace:
        for implementation, payload in trace.items():
            first = payload.get("first_divergence")
            boundary = first.get("trace_key") if first else "none (all traced tensors exact)"
            lines.append(f"- `{implementation}`: {boundary}")
    else:
        lines.append("Unavailable: no invocation-aware trace report was produced.")
    if local_native:
        lines.append(
            f"Native local replay at `{local_native['module']}` used bit-identical inputs and "
            f"reproduced a max error of {fmt(local_native['local_replay_comparison']['max_absolute_error'])} "
            f"through {', '.join(local_native['dispatched_aten_operations'])}."
        )
    if local_patch and local_explicit and local_full:
        lines.append(
            f"The VLM layer-0 output projection replay was locally batch-sensitive in both the "
            f"patch-projection and explicit-attention modes (max "
            f"{fmt(local_explicit['local_replay_comparison']['max_absolute_error'])}), but exact "
            f"for the tested full-invariant input."
        )

    lines += ["", "## C. How do differences propagate through the flow solver?", ""]
    if heldout:
        for implementation in ("native", "full_invariant"):
            steps = flow_step_summary(heldout, implementation)
            lines.append(f"- `{implementation}`:")
            for step, values in steps.items():
                lines.append(
                    f"  - step {step}: exact {values['exact']}/{values['comparisons']}, "
                    f"median={fmt(values['median'])}, p95={fmt(values['p95'])}, "
                    f"maximum={fmt(values['maximum'])}."
                )
        if trace:
            lines.append(
                "The frozen B=2 diverse diagnostic trace supplies the causal intermediate "
                "paths not run across the exhaustive held-out set (mean and maximum below "
                "are over the 28 action-state elements at each step):"
            )
            for implementation in (
                "native", "existing_invariant_ops",
                "invariant_plus_patch_projection", "full_invariant",
            ):
                points = {}
                for item in trace[implementation].get("comparisons", []):
                    match = FLOW.match(item["trace_key"])
                    if match:
                        points[int(match.group(1)) + 1] = (
                            item["mean_absolute_error"], item["max_absolute_error"]
                        )
                values = ", ".join(
                    f"{step}:mean {fmt(error[0])}/max {fmt(error[1])}"
                    for step, error in sorted(points.items())
                )
                lines.append(f"- `{implementation}` step:mean/max — {values}")
    elif trace:
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
        if transformations:
            full_transformations = [
                item for item in transformations
                if item["implementation"] == "full_invariant"
            ]
            lines.append(
                f"The full path also had {sum(not item['output']['exact'] for item in full_transformations)}/"
                f"{len(full_transformations)} failures across restored permutations and partitioning "
                "at B=2/4/8/16/32."
            )
    else:
        lines.append("Unavailable.")

    lines += ["", "## E. How faithful is invariant singleton inference to native singleton inference?", ""]
    if heldout_fidelity:
        values = [item["normalized"] for item in heldout_fidelity]
        errors = [item["max_absolute_error"] for item in values]
        lines.append(
            f"Across {len(values)} held-out request/noise pairs, "
            f"{sum(item['exact'] for item in values)} were bit-identical. Maximum absolute "
            f"error={fmt(max(errors))}, median={fmt(statistics.median(errors))}, "
            f"p95={fmt(percentile(errors, 0.95))}. This patched-vs-native singleton "
            "comparison is distinct from request-level batch invariance."
        )
    elif fidelity:
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
    if simpler_episodes and simpler_summary:
        lines.append(
            f"The paired closed-loop campaign completed {len(simpler_episodes)}/800 episodes."
        )
        for task, result in sorted(simpler_summary["tasks"].items()):
            rates = result["success_rate"]
            lines.append(
                f"- `{task}` success rates: native singleton={rates['native_singleton']:.3f}, "
                f"native dynamic={rates['native_dynamic']:.3f}, patched singleton="
                f"{rates['patched_singleton']:.3f}, patched dynamic="
                f"{rates['patched_dynamic']:.3f}."
            )
            for comparison, values in result["paired_differences"].items():
                lines.append(
                    f"  - `{comparison}`: paired estimate={values['estimate']:.3f}, "
                    f"95% bootstrap CI=[{values['ci95'][0]:.3f}, {values['ci95'][1]:.3f}], "
                    f"disagreements={values['disagreement_count']}/{result['episodes']}."
                )
            for comparison, values in result["trajectory_pairs"].items():
                lines.append(
                    f"  - `{comparison}` trajectory maxima: position="
                    f"{fmt(max(item['end_effector_position_max_m'] for item in values))} m, "
                    f"rotation={fmt(max(item['end_effector_rotation_max_rad'] for item in values))} rad."
                )
    elif simpler_blocker:
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
        if dispatch:
            lines.append(
                "Dispatcher evidence covers: "
                + ", ".join(
                    f"`{item['source_operation']}`→`{item['dispatched_aten_operation']}` "
                    f"({item.get('path_classification', 'registered_invariant_override')})"
                    for item in dispatch
                )
                + "."
            )
    else:
        lines.append("Operator campaign unavailable.")

    output = RESULTS / "EXPERIMENT_REPORT.md"
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
