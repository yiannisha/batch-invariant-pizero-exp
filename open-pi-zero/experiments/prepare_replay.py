#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.common import (
    PROJECT_ROOT,
    git_sha,
    index_inputs,
    load_pretrained_policy,
    load_numerical_freeze,
    prepare_inputs,
    run_policy,
    seed_everything,
    sha256_file,
    utc_timestamp,
    write_json,
)
from experiments.simpler_support import (
    FractalSimplerAdapter,
    make_simpler_env,
    observation_state,
)


TASKS = {
    "pick_can": "google_robot_pick_horizontal_coke_can",
    "move_near": "google_robot_move_near_v0",
    "open_drawer": "google_robot_open_drawer",
    "close_drawer": "google_robot_close_drawer",
}


def noise_for(request_id: str, noise_index: int) -> np.ndarray:
    digest = hashlib.sha256(f"{request_id}/noise-{noise_index}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(4, 7, generator=generator).numpy()


def save_observation(path: Path, inputs: dict, noises: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input_ids=inputs["input_ids"].numpy(),
        attention_mask=inputs["attention_mask"].numpy(),
        pixel_values=inputs["pixel_values"].numpy(),
        proprios=inputs["proprios"].numpy(),
        initial_actions=noises,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--freeze", type=Path, default=Path("results/numerical_freeze.json"))
    parser.add_argument("--statistics", type=Path, default=Path("config/fractal_statistics.json"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("results/replay_manifest.json"))
    parser.add_argument("--episodes-per-task", type=int, default=25)
    parser.add_argument("--observations-per-episode", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20250117)
    parser.add_argument("--simpler-python", type=Path)
    parser.add_argument("--simpler-root", type=Path)
    parser.add_argument("--vulkan-icd", type=Path)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    seed_everything(args.seed)
    freeze = load_numerical_freeze(args.freeze, args.checkpoint)
    model, config = load_pretrained_policy(
        args.checkpoint, config_path=args.config, dtype=torch.float32
    )
    adapter = FractalSimplerAdapter(args.tokenizer, args.statistics)
    records = []
    episodes = []
    for task_index, (task_name, environment_name) in enumerate(TASKS.items()):
        env = make_simpler_env(
            environment_name,
            args.simpler_python,
            args.simpler_root,
            args.vulkan_icd,
        )
        for episode_index in range(args.episodes_per_task):
            reset_seed = args.seed + task_index * 10_000 + episode_index
            observation, reset_info = env.reset(
                seed=reset_seed,
                options={"obj_init_options": {"episode_id": episode_index}}
            )
            adapter.reset()
            instruction = env.get_language_instruction()
            candidates = []
            policy_call = 0
            truncated = False
            success = False
            terminal_info = {}
            while not truncated:
                processed = adapter.preprocess(env, observation, instruction)
                candidates.append(
                    {
                        "policy_call": policy_call,
                        "inputs": {key: value.cpu() for key, value in processed.items()},
                        "instruction": instruction,
                        "simulator_state": observation_state(observation),
                    }
                )
                rollout_id = f"{task_name}/episode-{episode_index:03d}/call-{policy_call:03d}"
                initial = torch.from_numpy(noise_for(rollout_id, 0))[None]
                raw = {**processed, "initial_action": initial}
                model_inputs = prepare_inputs(model, raw, torch.float32)
                actions, _, _, _ = run_policy(model, model_inputs, "native")
                environment_actions = adapter.postprocess(actions[0].numpy())
                for environment_action in environment_actions[: int(config.act_steps)]:
                    observation, _, success, truncated, terminal_info = env.step(
                        environment_action
                    )
                    if truncated:
                        break
                new_instruction = env.get_language_instruction()
                if new_instruction != instruction:
                    instruction = new_instruction
                policy_call += 1
            if len(candidates) < args.observations_per_episode:
                raise RuntimeError(
                    f"{task_name} episode {episode_index} produced only {len(candidates)} observations"
                )
            selected = np.linspace(
                0, len(candidates) - 1, args.observations_per_episode, dtype=int
            )
            episode_record_ids = []
            for observation_index, candidate_index in enumerate(selected):
                candidate = candidates[int(candidate_index)]
                request_stem = (
                    f"{task_name}/episode-{episode_index:03d}/observation-{observation_index:02d}"
                )
                noises = np.stack(
                    [noise_for(request_stem, noise_index) for noise_index in range(3)]
                )
                tensor_path = args.dataset_root / f"{request_stem}.npz"
                save_observation(tensor_path, candidate["inputs"], noises)
                request_ids = [
                    f"{request_stem}/noise-{noise_index}" for noise_index in range(3)
                ]
                episode_record_ids.extend(request_ids)
                records.append(
                    {
                        "task": task_name,
                        "environment": environment_name,
                        "episode_id": episode_index,
                        "observation_index": observation_index,
                        "source_policy_call": candidate["policy_call"],
                        "split": "diagnostic" if episode_index < 5 else "heldout",
                        "instruction": candidate["instruction"],
                        "tensor_path": str(tensor_path.resolve()),
                        "noise_ids": [0, 1, 2],
                        "request_ids": request_ids,
                        "simulator_state": candidate["simulator_state"],
                    }
                )
            episodes.append(
                {
                    "task": task_name,
                    "episode_id": episode_index,
                    "reset_seed": reset_seed,
                    "success": bool(success),
                    "terminal_info": str(terminal_info),
                    "reset_info": str(reset_info),
                    "policy_calls": policy_call,
                    "request_ids": episode_record_ids,
                }
            )
        env.close()
    manifest = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "seed": args.seed,
        "provenance": {
            "batch_invariant_pizero_sha": git_sha(PROJECT_ROOT.parent),
            "numerical_policy_sha": freeze["batch_invariant_pizero_sha"],
            "batch_invariant_ops_sha": freeze["batch_invariant_ops_sha"],
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": freeze["checkpoint"]["sha256"],
            "tokenizer_path": str(args.tokenizer.resolve()),
            "statistics_path": str(args.statistics.resolve()),
        },
        "episodes_per_task": args.episodes_per_task,
        "observations_per_episode": args.observations_per_episode,
        "noise_tensors_per_observation": 3,
        "tasks": TASKS,
        "counts": {
            "episodes": len(episodes),
            "observations": len(records),
            "diagnostic_observations": sum(r["split"] == "diagnostic" for r in records),
            "heldout_observations": sum(r["split"] == "heldout" for r in records),
            "request_noise_pairs": len(records) * 3,
        },
        "episodes": episodes,
        "observations": records,
    }
    expected = {
        "episodes": 100,
        "observations": 1000,
        "diagnostic_observations": 200,
        "heldout_observations": 800,
        "request_noise_pairs": 3000,
    }
    if args.episodes_per_task == 25 and args.observations_per_episode == 10:
        if manifest["counts"] != expected:
            raise RuntimeError(f"replay cardinality mismatch: {manifest['counts']}")
    write_json(args.manifest, manifest)
    print(args.manifest)


if __name__ == "__main__":
    main()
