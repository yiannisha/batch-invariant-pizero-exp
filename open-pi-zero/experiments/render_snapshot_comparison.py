#!/usr/bin/env python3
"""Replay recorded exact-start SIMPLER actions and render paired MP4 videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.common import sha256_file, write_json
from experiments.prepare_replay import TASKS
from experiments.simpler_support import make_simpler_env, observation_state


CONDITIONS = (
    "native_singleton",
    "native_dynamic",
    "patched_singleton",
    "patched_dynamic",
)


def rgb(observation: dict) -> np.ndarray:
    return next(iter(observation["image"].values()))["rgb"]


def load_block(path: Path, task: str, initialization: int) -> dict:
    with path.open() as stream:
        rows = [
            row for line in stream if line.strip()
            if (row := json.loads(line))["task"] == task
            and row["initialization_id"] == initialization
        ]
    block = {row["condition"]: row for row in rows}
    if len(rows) != 8 or len(block) != 8 or not set(CONDITIONS).issubset(block):
        raise RuntimeError("expected one complete eight-condition block")
    if not all(row["aa_qualified"] and all(row["restore_checks"].values()) for row in rows):
        raise RuntimeError("block did not pass A/A and restore checks")
    if len({json.dumps(row["snapshot"], sort_keys=True) for row in rows}) != 1:
        raise RuntimeError("condition snapshots differ")
    return block


def replay(env, block: dict, snapshot: dict) -> tuple[dict, dict]:
    frames = {}
    outcomes = {}
    for condition in CONDITIONS:
        record = block[condition]
        restored = env.restore_snapshot()
        if not all(restored["checks"].values()):
            raise RuntimeError(f"restore failed: {condition}")
        for key in ("sim_state_sha256", "observation_sha256"):
            if restored[key] != snapshot[key]:
                raise RuntimeError(f"snapshot digest mismatch: {condition}/{key}")
        observation = restored["observation"]
        if observation_state(observation) != record["trajectory"][0]:
            raise RuntimeError(f"initial recorded trajectory mismatch: {condition}")
        condition_frames = [rgb(observation).copy()]
        success = False
        truncated = False
        for index, action in enumerate(record["action_sequence"]):
            observation, reward, success, truncated, _ = env.step(
                np.asarray(action["environment_action"], dtype=np.float64)
            )
            if observation_state(observation) != record["trajectory"][index + 1]:
                raise RuntimeError(f"trajectory mismatch: {condition}/step-{index + 1}")
            if float(reward) != action["reward"]:
                raise RuntimeError(f"reward mismatch: {condition}/step-{index + 1}")
            condition_frames.append(rgb(observation).copy())
        if not truncated or bool(success) != record["success"]:
            raise RuntimeError(f"outcome mismatch: {condition}")
        frames[condition] = condition_frames
        outcomes[condition] = bool(success)
    if len(frames["patched_singleton"]) != len(frames["patched_dynamic"]):
        raise RuntimeError("patched frame counts differ")
    if not all(np.array_equal(a, b) for a, b in zip(
        frames["patched_singleton"], frames["patched_dynamic"]
    )):
        raise RuntimeError("patched RGB frames differ")
    return frames, outcomes


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def render_pair(
    output: Path,
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
    *,
    title: str,
    left_name: str,
    right_name: str,
    left_success: bool,
    right_success: bool,
    note: str,
) -> None:
    if len(left_frames) != len(right_frames):
        raise RuntimeError("pair frame counts differ")
    height, width = left_frames[0].shape[:2]
    scale = 2
    panel_width = width * scale
    panel_height = height * scale
    canvas_size = (2 * panel_width, panel_height + 164)
    with tempfile.TemporaryDirectory(prefix="snapshot-video-") as temporary:
        frame_dir = Path(temporary)
        for index, (left, right) in enumerate(zip(left_frames, right_frames)):
            canvas = Image.new("RGB", canvas_size, (20, 25, 33))
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 10), title, fill="white", font=font(24))
            draw.text((18, 47), f"Shared start  |  simulator step {index}/{len(left_frames) - 1}", fill=(205, 214, 224), font=font(17))
            draw.text((18, 75), left_name, fill=(110, 202, 255), font=font(20))
            draw.text((panel_width + 18, 75), right_name, fill=(255, 198, 109), font=font(20))
            canvas.paste(Image.fromarray(left).resize((panel_width, panel_height)), (0, 105))
            canvas.paste(Image.fromarray(right).resize((panel_width, panel_height)), (panel_width, 105))
            draw.line((panel_width, 105, panel_width, 105 + panel_height), fill="white", width=2)
            final = index == len(left_frames) - 1
            left_label = "SUCCESS" if left_success else "FAILURE"
            right_label = "SUCCESS" if right_success else "FAILURE"
            draw.text((18, panel_height + 113), f"Outcome: {left_label}" if final else "Outcome shown at end", fill=(105, 232, 152) if final and left_success else (255, 125, 125) if final else (205, 214, 224), font=font(18))
            draw.text((panel_width + 18, panel_height + 113), f"Outcome: {right_label}" if final else "Outcome shown at end", fill=(105, 232, 152) if final and right_success else (255, 125, 125) if final else (205, 214, 224), font=font(18))
            draw.text((18, panel_height + 140), note, fill=(205, 214, 224), font=font(15))
            canvas.save(frame_dir / f"frame_{index:04d}.png")
        # Hold the final state so the outcome can be read comfortably.
        final_frame = frame_dir / f"frame_{len(left_frames) - 1:04d}.png"
        final_bytes = final_frame.read_bytes()
        for index in range(len(left_frames), len(left_frames) + 20):
            (frame_dir / f"frame_{index:04d}.png").write_bytes(final_bytes)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10",
             "-i", str(frame_dir / "frame_%04d.png"), "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="close_drawer")
    parser.add_argument("--initialization", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simpler-python", type=Path, required=True)
    parser.add_argument("--simpler-root", type=Path, required=True)
    parser.add_argument("--vulkan-icd", type=Path, required=True)
    args = parser.parse_args()
    block = load_block(args.episodes, args.task, args.initialization)
    reference = block["native_singleton"]
    if block["native_singleton"]["success"] == block["native_dynamic"]["success"]:
        raise RuntimeError("selected native pair does not have differing outcomes")
    env = make_simpler_env(TASKS[args.task], args.simpler_python, args.simpler_root, args.vulkan_icd)
    try:
        env.reset(
            seed=reference["initialization_seed"],
            options={"obj_init_options": {"episode_id": args.initialization % 25}},
        )
        snapshot = env.capture_snapshot()
        if snapshot != reference["snapshot"]:
            raise RuntimeError(f"fresh replay did not reproduce recorded start: {snapshot}")
        frames, outcomes = replay(env, block, snapshot)
    finally:
        env.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_start_{args.initialization:02d}"
    task_label = args.task.replace("_", " ").title()
    native_video = args.output_dir / f"{stem}_native_singleton_vs_batched.mp4"
    patched_video = args.output_dir / f"{stem}_patched_singleton_vs_batched.mp4"
    render_pair(
        native_video, frames["native_singleton"], frames["native_dynamic"],
        title=f"{task_label}  |  Native inference", left_name="Singleton", right_name="Dynamic batch (1, 2, 4, 8)",
        left_success=outcomes["native_singleton"], right_success=outcomes["native_dynamic"],
        note="Recorded native actions replayed from the exact same simulator snapshot",
    )
    render_pair(
        patched_video, frames["patched_singleton"], frames["patched_dynamic"],
        title=f"{task_label}  |  Batch-invariant inference", left_name="Singleton", right_name="Dynamic batch (1, 2, 4, 8)",
        left_success=outcomes["patched_singleton"], right_success=outcomes["patched_dynamic"],
        note="Recorded patched actions and all rendered frames match exactly",
    )
    metadata = {
        "task": args.task,
        "initialization": args.initialization,
        "source_episodes_sha256": sha256_file(args.episodes),
        "snapshot": snapshot,
        "outcomes": outcomes,
        "frame_count": {name: len(value) for name, value in frames.items()},
        "patched_rgb_frames_exact": True,
        "native_video": str(native_video),
        "patched_video": str(patched_video),
    }
    write_json(args.output_dir / f"{stem}_video_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
