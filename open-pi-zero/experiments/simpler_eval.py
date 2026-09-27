#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common import (
    append_jsonl,
    load_numerical_freeze,
    load_pretrained_policy,
    prepare_inputs,
    run_policy,
    utc_timestamp,
)
from experiments.heldout_invariance import concatenate, raw_request
from experiments.prepare_replay import TASKS
from experiments.simpler_support import (
    FractalSimplerAdapter,
    make_simpler_env,
    observation_state,
)


CONDITIONS = {
    "native_singleton": ("native", False),
    "native_dynamic": ("native", True),
    "patched_singleton": ("full_invariant", False),
    "patched_dynamic": ("full_invariant", True),
}


def indexed_noise(task: str, initialization: int, policy_call: int) -> torch.Tensor:
    key = noise_id(task, initialization, policy_call)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 4, 7, generator=generator)


def noise_id(task: str, initialization: int, policy_call: int) -> str:
    return f"{task}/initialization-{initialization}/policy-call-{policy_call}"


def completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    with path.open() as stream:
        return {
            (item["task"], item["initialization_id"], item["condition"])
            for line in stream
            if (item := json.loads(line))
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--freeze", type=Path, default=Path("results/numerical_freeze.json"))
    parser.add_argument("--statistics", type=Path, default=Path("config/fractal_statistics.json"))
    parser.add_argument("--output", type=Path, default=Path("results/simpler/episodes.jsonl"))
    parser.add_argument("--initializations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20250311)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--simpler-python", type=Path)
    parser.add_argument("--simpler-root", type=Path)
    parser.add_argument("--vulkan-icd", type=Path)
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; use --resume")
    freeze = load_numerical_freeze(args.freeze, args.checkpoint)
    model, config = load_pretrained_policy(
        args.checkpoint, config_path=args.config, dtype=torch.float32
    )
    adapter = FractalSimplerAdapter(args.tokenizer, args.statistics)
    replay = json.loads(args.replay_manifest.read_text())
    replay_provenance = replay.get("provenance", {})
    if replay_provenance.get("numerical_policy_sha") != freeze["batch_invariant_pizero_sha"]:
        raise RuntimeError("replay manifest and numerical freeze revisions differ")
    if replay_provenance.get("checkpoint_sha256") != freeze["checkpoint"]["sha256"]:
        raise RuntimeError("replay manifest and frozen checkpoint differ")
    if args.output.exists():
        with args.output.open() as stream:
            revisions = {
                record.get("numerical_policy_sha")
                for line in stream
                if line.strip() and (record := json.loads(line))
            }
        if revisions != {freeze["batch_invariant_pizero_sha"]}:
            raise RuntimeError(
                f"refusing to combine SIMPLER revisions: existing={revisions}, "
                f"frozen={freeze['batch_invariant_pizero_sha']}"
            )
    companion_observations = replay["observations"]
    already_done = completed(args.output)

    for task, environment_name in TASKS.items():
        env = make_simpler_env(
            environment_name,
            args.simpler_python,
            args.simpler_root,
            args.vulkan_icd,
        )
        for initialization_id in range(args.initializations):
            for condition, (implementation, dynamic) in CONDITIONS.items():
                if (task, initialization_id, condition) in already_done:
                    continue
                observation, reset_info = env.reset(
                    seed=args.seed + initialization_id,
                    options={
                        "obj_init_options": {
                            "episode_id": initialization_id % 25
                        }
                    },
                )
                adapter.reset()
                instruction = env.get_language_instruction()
                actions_record = []
                trajectory = [observation_state(observation)]
                policy_call = 0
                truncated = False
                success = False
                terminal_info = {}
                while not truncated:
                    processed = adapter.preprocess(env, observation, instruction)
                    target = {
                        **processed,
                        "initial_action": indexed_noise(
                            task, initialization_id, policy_call
                        ),
                    }
                    batch_size = (1, 2, 4, 8)[policy_call % 4] if dynamic else 1
                    target_position = policy_call % batch_size
                    requests = []
                    companion_request_ids = []
                    for companion_index in range(batch_size - 1):
                        replay_observation = companion_observations[
                            (initialization_id * 97 + policy_call * 11 + companion_index)
                            % len(companion_observations)
                        ]
                        requests.append(
                            raw_request(
                                replay_observation,
                                policy_call % 3,
                                torch.float32,
                            )
                        )
                        companion_request_ids.append(
                            replay_observation["request_ids"][policy_call % 3]
                        )
                    requests.insert(target_position, target)
                    inputs = prepare_inputs(model, concatenate(requests), torch.float32)
                    output, _, _, _ = run_policy(model, inputs, implementation)
                    normalized_actions = output[target_position].numpy()
                    environment_actions = adapter.postprocess(normalized_actions)
                    for action_index, environment_action in enumerate(
                        environment_actions[: int(config.act_steps)]
                    ):
                        observation, reward, success, truncated, terminal_info = env.step(
                            environment_action
                        )
                        actions_record.append(
                            {
                                "policy_call": policy_call,
                                "action_index": action_index,
                                "batch_size": batch_size,
                                "target_position": target_position,
                                "noise_id": noise_id(
                                    task, initialization_id, policy_call
                                ),
                                "companion_request_ids": companion_request_ids,
                                "normalized_action": normalized_actions[action_index].tolist(),
                                "environment_action": environment_action.tolist(),
                                "reward": float(reward),
                            }
                        )
                        trajectory.append(observation_state(observation))
                        if truncated:
                            break
                    new_instruction = env.get_language_instruction()
                    if new_instruction != instruction:
                        instruction = new_instruction
                    policy_call += 1
                append_jsonl(
                    args.output,
                    [
                        {
                            "schema_version": 1,
                            "timestamp": utc_timestamp(),
                            "numerical_policy_sha": freeze["batch_invariant_pizero_sha"],
                            "batch_invariant_ops_sha": freeze["batch_invariant_ops_sha"],
                            "checkpoint_sha256": freeze["checkpoint"]["sha256"],
                            "task": task,
                            "environment": environment_name,
                            "initialization_id": initialization_id,
                            "initialization_seed": args.seed + initialization_id,
                            "condition": condition,
                            "implementation": implementation,
                            "dynamic_batching": dynamic,
                            "success": bool(success),
                            "termination_reason": str(terminal_info),
                            "reset_info": str(reset_info),
                            "policy_call_count": policy_call,
                            "action_sequence": actions_record,
                            "trajectory": trajectory,
                            "dynamic_batch_cycle": [1, 2, 4, 8] if dynamic else [1],
                            "policy_noise_scheme": "sha256-indexed-by-task-initialization-policy-call",
                        }
                    ],
                )
        env.close()
    print(args.output)


if __name__ == "__main__":
    main()
