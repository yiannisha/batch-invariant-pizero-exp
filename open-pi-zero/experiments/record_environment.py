#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import socket
from pathlib import Path

import torch

from experiments.common import (
    PROJECT_ROOT,
    git_sha,
    run_text,
    sha256_file,
    utc_timestamp,
    write_json,
)


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/environment.json"))
    parser.add_argument(
        "--ops-repository",
        type=Path,
        default=Path(os.environ.get("BATCH_INVARIANT_OPS_REPO", "../batch_invariant_ops")),
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    ops_repository = args.ops_repository.resolve()
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessor_count": properties.multi_processor_count,
            "nvidia_smi": run_text(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.memory,persistence_mode",
                    "--format=csv,noheader,nounits",
                ]
            ),
        }
    relevant_environment = {
        name: os.environ[name]
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE",
            "PYTORCH_CUDA_ALLOC_CONF",
            "TORCH_LOGS",
            "TOKENIZERS_PARALLELISM",
        )
        if name in os.environ
    }
    record = {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "repositories": {
            "batch_invariant_pizero": {
                "path": str(PROJECT_ROOT.parent),
                "git_sha": git_sha(PROJECT_ROOT.parent),
            },
            "batch_invariant_ops": {
                "path": str(ops_repository),
                "git_sha": git_sha(ops_repository),
                "pr_27_base_revision": run_text(
                    ["git", "rev-parse", "679da28^{commit}"], ops_repository
                ),
                "integration_patch_revision": git_sha(ops_repository),
            },
        },
        "checkpoint": {
            "identifier": "allenzren/open-pi-zero/fractal_beta_step29576_2024-12-29_13-10_42.pt",
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
        },
        "hardware": {"gpu": gpu},
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_toolkit": run_text(["nvcc", "--version"]),
            "nvidia_driver": run_text(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "cudnn": torch.backends.cudnn.version(),
            "cublas": os.environ.get("NV_LIBCUBLAS_VERSION"),
            "triton": package_version("triton"),
            "transformers": package_version("transformers"),
            "simpler": package_version("simpler-env"),
        },
        "numerics": {
            "primary_dtype": "float32",
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "execution": "eager",
        },
        "environment": relevant_environment,
    }
    write_json(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
