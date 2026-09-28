#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from experiments.common import (
    append_jsonl,
    load_numerical_freeze,
    load_pretrained_policy,
    prepare_inputs,
    run_text,
    run_policy,
    sha256_file,
    sha256_tree,
    utc_timestamp,
    write_json,
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


def source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        name: sha256_file(directory / name)
        for name in (
            "simpler_eval.py",
            "simpler_worker.py",
            "simpler_support.py",
            "analyze_simpler.py",
        )
    }


def git_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    return run_text(["git", "rev-parse", "HEAD"], cwd=path)


def file_identity(path: Path) -> dict:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def egl_loader_identity() -> dict | None:
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not directory:
            continue
        candidate = Path(directory) / "libEGL.so.1"
        if candidate.is_file():
            return file_identity(candidate)
    return None


def write_runtime_provenance(args, freeze: dict) -> Path:
    runtime_path = args.output.parent / "runtime.json"
    project_root = Path(__file__).resolve().parents[1]
    ops_root_value = os.environ.get("BATCH_INVARIANT_OPS_REPO")
    ops_root = Path(ops_root_value).resolve() if ops_root_value else None
    simpler_root = args.simpler_root.resolve() if args.simpler_root else None
    rtx_environment_path = project_root / "results" / "rtx_environment.json"
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    identity = {
        "campaign": "paired_closed_loop_simpler",
        "evaluation_harness_sha": git_sha(project_root),
        "numerical_policy_sha": freeze["batch_invariant_pizero_sha"],
        "frozen_operator_baseline_sha": freeze["batch_invariant_ops_sha"],
        "rtx_launch_adaptation_sha": git_sha(ops_root),
        "checkpoint_sha256": freeze["checkpoint"]["sha256"],
        "replay_manifest_sha256": sha256_file(args.replay_manifest),
        "rtx_environment_sha256": sha256_file(rtx_environment_path),
        "tokenizer": sha256_tree(args.tokenizer),
        "graphics_runtime": {
            "vulkan_icd": (
                file_identity(args.vulkan_icd) if args.vulkan_icd else None
            ),
            "egl_loader": egl_loader_identity(),
            "VK_DRIVER_FILES": os.environ.get("VK_DRIVER_FILES"),
            "VK_LOADER_LAYERS_DISABLE": os.environ.get(
                "VK_LOADER_LAYERS_DISABLE"
            ),
            "NVIDIA_DRIVER_CAPABILITIES": os.environ.get(
                "NVIDIA_DRIVER_CAPABILITIES"
            ),
            "vulkaninfo_summary": run_text(["vulkaninfo", "--summary"]),
        },
        "source_sha256": source_hashes(),
        "repositories": {
            "simpler_env_sha": git_sha(simpler_root),
            "maniskill2_real2sim_sha": git_sha(
                simpler_root / "ManiSkill2_real2sim" if simpler_root else None
            ),
        },
        "conditions": {
            name: {
                "implementation": implementation,
                "dynamic_batching": dynamic,
            }
            for name, (implementation, dynamic) in CONDITIONS.items()
        },
        "initializations_per_task": args.initializations,
        "tasks": sorted(TASKS),
        "seed": args.seed,
        "expected_episodes": args.initializations * len(TASKS) * len(CONDITIONS),
        "policy_noise_scheme": "sha256-indexed-by-task-initialization-policy-call",
        "dynamic_batch_cycle": [1, 2, 4, 8],
        "execution": {
            "policy_python": os.path.realpath(os.sys.executable),
            "simpler_python": str(args.simpler_python.resolve()) if args.simpler_python else None,
            "tokenizer_path": str(args.tokenizer.resolve()),
            "vulkan_icd": str(args.vulkan_icd.resolve()) if args.vulkan_icd else None,
            "resume_enabled": args.resume,
        },
        "hardware": {
            "gpu_name": device.name,
            "compute_capability": [device.major, device.minor],
            "total_memory_bytes": device.total_memory,
        },
        "software": {
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }
    if runtime_path.exists():
        existing = json.loads(runtime_path.read_text())
        for key, value in identity.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"refusing to resume SIMPLER with changed runtime provenance: {key}"
                )
        existing.setdefault("resume_events", []).append(utc_timestamp())
        write_json(runtime_path, existing)
        return runtime_path
    write_json(
        runtime_path,
        {
            "schema_version": 1,
            "launched_at": utc_timestamp(),
            **identity,
            "resume_events": [],
        },
    )
    return runtime_path


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
    runtime_path = write_runtime_provenance(args, freeze)
    print(runtime_path)
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
                    target_noise_id = noise_id(
                        task, initialization_id, policy_call
                    )
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
                    request_ordering = list(companion_request_ids)
                    request_ordering.insert(target_position, target_noise_id)
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
                                "noise_id": target_noise_id,
                                "companion_request_ids": companion_request_ids,
                                "request_ordering": request_ordering,
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
                            "experiment_id": str(uuid.uuid4()),
                            "timestamp": utc_timestamp(),
                            "numerical_policy_sha": freeze["batch_invariant_pizero_sha"],
                            "batch_invariant_ops_sha": freeze["batch_invariant_ops_sha"],
                            "checkpoint_sha256": freeze["checkpoint"]["sha256"],
                            "request_id": (
                                f"{task}/initialization-{initialization_id:03d}/"
                                f"condition-{condition}"
                            ),
                            "episode_id": initialization_id,
                            "task": task,
                            "environment": environment_name,
                            "initialization_id": initialization_id,
                            "initialization_seed": args.seed + initialization_id,
                            "condition": condition,
                            "implementation": implementation,
                            "dynamic_batching": dynamic,
                            "dtype": "float32",
                            "tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                            "execution_mode": "eager",
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
