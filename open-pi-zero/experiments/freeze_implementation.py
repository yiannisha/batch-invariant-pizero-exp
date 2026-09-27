#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from experiments.common import PROJECT_ROOT, git_sha, sha256_file, utc_timestamp, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ops-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/numerical_freeze.json"))
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="diagnostic only; a dirty revision is not a valid held-out freeze",
    )
    args = parser.parse_args()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT.parent, text=True
    ).strip()
    if status and not args.allow_dirty:
        raise RuntimeError("commit the numerical implementation before freezing it")
    record = {
        "schema_version": 1,
        "frozen_at": utc_timestamp(),
        "numerical_implementation_frozen": not bool(status),
        "dirty_status": status.splitlines(),
        "batch_invariant_pizero_sha": git_sha(PROJECT_ROOT.parent),
        "batch_invariant_ops_sha": git_sha(args.ops_repository),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
    }
    write_json(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
