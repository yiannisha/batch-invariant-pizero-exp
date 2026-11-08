#!/usr/bin/env python3
"""Track four detached exact-start tasks and run the final audit when complete."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.common import utc_timestamp, write_json
from experiments.prepare_replay import TASKS


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as stream:
        return sum(bool(line.strip()) for line in stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--initializations", type=int, default=50)
    parser.add_argument("--pid", action="append", required=True, help="task=process-id")
    args = parser.parse_args()
    pids = {}
    for assignment in args.pid:
        task, raw_pid = assignment.split("=", 1)
        if task not in TASKS or task in pids:
            raise ValueError(assignment)
        pids[task] = int(raw_pid)
    if set(pids) != set(TASKS):
        raise ValueError("one PID is required for each task")
    expected_lines = args.initializations * 8
    status_path = args.results_root / "status.json"
    while True:
        progress = {
            task: {
                "records": count_lines(args.results_root / task / "episodes.jsonl"),
                "expected_records": expected_lines,
                "pid": pids[task],
                "process_alive": alive(pids[task]),
            }
            for task in TASKS
        }
        overfull = [task for task, value in progress.items() if value["records"] > expected_lines]
        failed = [
            task for task, value in progress.items()
            if not value["process_alive"] and value["records"] < expected_lines
        ]
        if overfull or failed:
            write_json(status_path, {
                "status": "failed", "updated_at": utc_timestamp(),
                "overfull_tasks": overfull, "incomplete_exited_tasks": failed,
                "progress": progress,
            })
            raise RuntimeError(f"snapshot campaign stopped: {overfull=}, {failed=}")
        if all(value["records"] == expected_lines for value in progress.values()):
            command = [
                sys.executable,
                str(Path(__file__).with_name("analyze_snapshot_simpler.py")),
                "--episodes",
                *[str(args.results_root / task / "episodes.jsonl") for task in TASKS],
                "--initializations", str(args.initializations),
                "--output", str(args.results_root / "summary.json"),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            write_json(status_path, {
                "status": "complete_verified" if completed.returncode == 0 else "analysis_failed",
                "updated_at": utc_timestamp(),
                "progress": progress,
                "analysis_stdout": completed.stdout,
                "analysis_stderr": completed.stderr,
            })
            if completed.returncode:
                raise RuntimeError(completed.stderr)
            return
        write_json(status_path, {
            "status": "running", "updated_at": utc_timestamp(), "progress": progress
        })
        time.sleep(60)


if __name__ == "__main__":
    main()
