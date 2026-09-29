#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from experiments.common import sha256_file, utc_timestamp, write_json


CONDITIONS = (
    "native_singleton",
    "native_dynamic",
    "patched_singleton",
    "patched_dynamic",
)
POSE = re.compile(
    r"'(?P<name>[^']+)': Pose\(\[(?P<position>[^\]]+)\], "
    r"\[(?P<quaternion>[^\]]+)\]\)"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_numbers(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(","))


def parse_poses(reset_info: str) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    return {
        match.group("name"): (
            parse_numbers(match.group("position")),
            parse_numbers(match.group("quaternion")),
        )
        for match in POSE.finditer(reset_info)
    }


def static_reset_info(reset_info: str) -> str:
    return POSE.sub(lambda match: f"'{match.group('name')}': Pose(<pose>)", reset_info)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def translation_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def rotation_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if left == right:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    cosine = abs(
        sum(a * b for a, b in zip(left, right, strict=True))
        / (left_norm * right_norm)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, cosine)))


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
    }


def analyze_task(task: str, episodes: list[dict]) -> dict:
    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    for episode in episodes:
        grouped[episode["initialization_id"]][episode["condition"]] = episode
    assert set(grouped) == set(range(50))
    assert all(set(group) == set(CONDITIONS) for group in grouped.values())

    pose_fields = set(parse_poses(episodes[0]["reset_info"]))
    assert all(
        set(parse_poses(episode["reset_info"])) == pose_fields
        for episode in episodes
    )
    raw_exact = 0
    static_exact = 0
    initial_state_exact = 0
    discordant = []
    per_pose = {name: [] for name in sorted(pose_fields)}

    for initialization_id in range(50):
        group = grouped[initialization_id]
        ordered = [group[condition] for condition in CONDITIONS]
        assert {
            episode["initialization_seed"] for episode in ordered
        } == {20250311 + initialization_id}
        raw_exact += len({episode["reset_info"] for episode in ordered}) == 1
        static_exact += len({static_reset_info(episode["reset_info"]) for episode in ordered}) == 1
        initial_state_exact += all(
            episode["trajectory"][0] == ordered[0]["trajectory"][0]
            for episode in ordered
        )
        if len({episode["success"] for episode in ordered}) > 1:
            discordant.append(initialization_id)

        parsed = [parse_poses(episode["reset_info"]) for episode in ordered]
        for name in sorted(pose_fields):
            poses = [item[name] for item in parsed]
            translations = [
                translation_distance(left[0], right[0])
                for left, right in combinations(poses, 2)
            ]
            rotations = [
                rotation_distance(left[1], right[1])
                for left, right in combinations(poses, 2)
            ]
            per_pose[name].append(
                {
                    "initialization_id": initialization_id,
                    "byte_exact_across_conditions": len(set(poses)) == 1,
                    "maximum_pairwise_translation_m": max(translations),
                    "maximum_pairwise_rotation_rad": max(rotations),
                }
            )

    pose_summary = {}
    for name, values in per_pose.items():
        translations = [item["maximum_pairwise_translation_m"] for item in values]
        rotations = [item["maximum_pairwise_rotation_rad"] for item in values]
        pose_summary[name] = {
            "byte_exact_blocks": sum(
                item["byte_exact_across_conditions"] for item in values
            ),
            "maximum_pairwise_translation_m": distribution(translations),
            "maximum_pairwise_rotation_rad": distribution(rotations),
            "per_initialization": values,
        }

    return {
        "episodes": len(episodes),
        "initialization_blocks": len(grouped),
        "same_seed_across_conditions": True,
        "static_reset_fields_exact_blocks": static_exact,
        "raw_reset_info_exact_blocks": raw_exact,
        "initial_robot_eef_state_exact_blocks": initial_state_exact,
        "outcome_discordant_blocks": len(discordant),
        "outcome_discordant_initialization_ids": discordant,
        "pose_fields": pose_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit same-seed SIMPLER reset pairing and post-reset pose variation."
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("results/simpler/episodes.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/simpler/reset_audit.json"),
    )
    args = parser.parse_args()
    episodes = read_jsonl(args.episodes)
    assert len(episodes) == 800
    tasks = sorted({episode["task"] for episode in episodes})
    assert tasks == ["close_drawer", "move_near", "open_drawer", "pick_can"]

    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "source": {
            "path": str(args.episodes),
            "records": len(episodes),
            "size_bytes": args.episodes.stat().st_size,
            "sha256": sha256_file(args.episodes),
        },
        "method": {
            "pairing_unit": "task and initialization_id",
            "conditions": list(CONDITIONS),
            "pose_translation": "maximum Euclidean distance across the six condition pairs",
            "pose_rotation": "maximum 2*acos(abs(unit-quaternion dot product)) across the six condition pairs",
            "qualification": (
                "Object-task effects are observed end-to-end effects under repeated same-seed "
                "scene construction because returned post-reset object poses are not always "
                "byte-identical. Drawer-task reset records are exactly paired."
            ),
        },
        "tasks": {
            task: analyze_task(
                task, [episode for episode in episodes if episode["task"] == task]
            )
            for task in tasks
        },
    }
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
