#!/usr/bin/env python3
"""Collect deterministic companion requests from the same SIMPLER task scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common import load_numerical_freeze, sha256_file, write_json
from experiments.prepare_replay import TASKS, noise_for, save_observation
from experiments.simpler_support import FractalSimplerAdapter, make_simpler_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=25)
    parser.add_argument("--simpler-python", type=Path, required=True)
    parser.add_argument("--simpler-root", type=Path, required=True)
    parser.add_argument("--vulkan-icd", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    freeze = load_numerical_freeze(args.freeze, args.checkpoint)
    adapter = FractalSimplerAdapter(args.tokenizer, args.statistics)
    observations = []
    for task_index, (task, environment_name) in enumerate(TASKS.items()):
        env = make_simpler_env(
            environment_name, args.simpler_python, args.simpler_root, args.vulkan_icd
        )
        try:
            for episode_id in range(args.episodes_per_task):
                observation, _ = env.reset(
                    seed=20250117 + task_index * 10_000 + episode_id,
                    options={"obj_init_options": {"episode_id": episode_id}},
                )
                adapter.reset()
                inputs = adapter.preprocess(
                    env, observation, env.get_language_instruction()
                )
                stem = f"{task}/episode-{episode_id:03d}/observation-00"
                path = args.dataset_root / f"{stem}.npz"
                save_observation(
                    path,
                    {key: value.cpu() for key, value in inputs.items()},
                    np.stack([noise_for(stem, index) for index in range(3)]),
                )
                observations.append(
                    {
                        "task": task,
                        "episode_id": episode_id,
                        "observation_index": 0,
                        "tensor_path": str(path.resolve()),
                        "request_ids": [f"{stem}/noise-{index}" for index in range(3)],
                    }
                )
                print(f"companion {len(observations)}: {stem}", flush=True)
        finally:
            env.close()
    write_json(
        args.manifest,
        {
            "schema_version": 1,
            "collection": "same_scene_reset_observations",
            "provenance": {
                "numerical_policy_sha": freeze["batch_invariant_pizero_sha"],
                "checkpoint_sha256": freeze["checkpoint"]["sha256"],
                "statistics_sha256": sha256_file(args.statistics),
            },
            "observations": observations,
        },
    )


if __name__ == "__main__":
    main()
