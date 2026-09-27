from __future__ import annotations

import json
import os
import pickle
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from transformers import AutoTokenizer

from src.model.vla.processing import VLAProcessor


class FractalSimplerAdapter:
    """Matching Fractal preprocessing and SIMPLER action conversion."""

    def __init__(self, tokenizer_path: Path | str, statistics_path: Path | str):
        self.statistics = json.loads(Path(statistics_path).read_text())
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
        self.processor = VLAProcessor(tokenizer, num_image_tokens=256, max_seq_len=276)
        self.reset()

    def reset(self) -> None:
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0

    @staticmethod
    def normalize_bound(data, low, high):
        return np.clip(2 * (data - low) / (high - low + 1e-8) - 1, -1, 1)

    @staticmethod
    def denormalize_bound(data, low, high):
        return (data + 1) / 2 * (high - low + 1e-8) + low

    def preprocess(self, env, observation: dict, instruction: str) -> dict:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError(f"unsupported robot: {env.robot_uid}")
        image = observation["image"][camera_name]["rgb"]
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        images = torch.as_tensor(image, dtype=torch.uint8).permute(2, 0, 1)[None]
        processed = self.processor(text=[instruction], images=images)
        eef = np.asarray(observation["agent"]["eef_pos"])
        quaternion_xyzw = np.roll(eef[3:7], -1)
        raw_proprio = np.concatenate((eef[:3], quaternion_xyzw, [1 - eef[7]]))
        proprio_stats = self.statistics["proprio"]
        proprio = self.normalize_bound(
            raw_proprio,
            np.asarray(proprio_stats["p01"]),
            np.asarray(proprio_stats["p99"]),
        )
        return {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "pixel_values": processed["pixel_values"],
            "proprios": torch.as_tensor(proprio, dtype=torch.float32)[None, None],
        }

    def _gripper(self, action: float) -> float:
        relative = -((action * 2) - 1)
        if abs(relative) > 0.5 and not self.sticky_action_is_on:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative = self.sticky_gripper_action
        if self.gripper_action_repeat == 15:
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
        return float(relative)

    def postprocess(self, normalized_actions: np.ndarray) -> np.ndarray:
        stats = self.statistics["action"]
        raw_six = self.denormalize_bound(
            normalized_actions[:, :-1],
            np.asarray(stats["p01"][:-1]),
            np.asarray(stats["p99"][:-1]),
        )
        raw = np.concatenate((raw_six, normalized_actions[:, -1:]), axis=1)
        result = np.zeros((len(raw), 7), dtype=np.float64)
        for index, action in enumerate(raw):
            rotation_vector = Rotation.from_euler("xyz", action[3:6]).as_rotvec()
            result[index] = np.concatenate(
                (action[:3], rotation_vector, [self._gripper(action[-1])])
            )
        return result


def observation_state(observation: dict) -> dict:
    """Retain only reliably serializable simulator state supplied in obs."""

    result = {}
    for group in ("agent", "extra"):
        if group not in observation:
            continue
        result[group] = {}
        for key, value in observation[group].items():
            if isinstance(value, np.ndarray):
                result[group][key] = value.tolist()
            elif np.isscalar(value):
                result[group][key] = np.asarray(value).item()
    return result


class RemoteSimplerEnv:
    """SIMPLER environment hosted in a separate Python 3.10 process."""

    def __init__(
        self,
        environment_name: str,
        python: Path | str,
        simpler_root: Path | str,
        vulkan_icd: Path | str,
        startup_timeout: float = 120.0,
    ):
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="pizero-simpler-"
        )
        temporary_path = Path(self._temporary_directory.name)
        runtime_path = temporary_path / "runtime"
        runtime_path.mkdir(mode=0o700)
        self._socket_path = temporary_path / "worker.sock"
        simpler_root = Path(simpler_root).resolve()
        environment = os.environ.copy()
        python_paths = [
            str(simpler_root),
            str(simpler_root / "ManiSkill2_real2sim"),
        ]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(python_paths),
                "VK_ICD_FILENAMES": str(Path(vulkan_icd).resolve()),
                "XDG_RUNTIME_DIR": str(runtime_path),
                "DISPLAY": "",
            }
        )
        worker = Path(__file__).with_name("simpler_worker.py")
        self._process = subprocess.Popen(
            [
                # Preserve a virtualenv's symlink path.  Resolving it to the
                # base interpreter would discard that environment's packages.
                str(Path(python).absolute()),
                str(worker),
                "--socket",
                str(self._socket_path),
                "--environment",
                environment_name,
            ],
            env=environment,
        )
        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + startup_timeout
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"SIMPLER worker exited with code {self._process.returncode}"
                )
            try:
                self._connection.connect(str(self._socket_path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    self._process.terminate()
                    raise TimeoutError("timed out starting SIMPLER worker")
                time.sleep(0.05)
        metadata = self._request({"command": "metadata"})
        self.robot_uid = metadata["robot_uid"]

    @staticmethod
    def _receive_exact(connection: socket.socket, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = connection.recv(remaining)
            if not chunk:
                raise EOFError("SIMPLER worker disconnected")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _request(self, request: dict) -> dict:
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        self._connection.sendall(struct.pack("!Q", len(payload)) + payload)
        size = struct.unpack("!Q", self._receive_exact(self._connection, 8))[0]
        response = pickle.loads(self._receive_exact(self._connection, size))
        if not response["ok"]:
            raise RuntimeError(
                f"SIMPLER worker error: {response['error']}\n{response['traceback']}"
            )
        return response["result"]

    def reset(self, *, seed: int, options: dict | None = None):
        result = self._request(
            {"command": "reset", "seed": seed, "options": options}
        )
        return result["observation"], result["info"]

    def step(self, action):
        # Plain lists keep the protocol compatible between NumPy 1.x in the
        # simulator and NumPy 2.x in the policy environment.
        result = self._request({"command": "step", "action": action.tolist()})
        return (
            result["observation"],
            result["reward"],
            result["success"],
            result["truncated"],
            result["info"],
        )

    def get_language_instruction(self) -> str:
        return self._request({"command": "instruction"})["instruction"]

    def close(self) -> None:
        if getattr(self, "_connection", None) is None:
            return
        try:
            self._request({"command": "close"})
        finally:
            self._connection.close()
            self._connection = None
            self._process.wait(timeout=30)
            self._temporary_directory.cleanup()


def make_simpler_env(
    environment_name: str,
    simpler_python: Path | None,
    simpler_root: Path | None,
    vulkan_icd: Path | None,
):
    if simpler_python is None:
        import simpler_env

        return simpler_env.make(environment_name)
    if simpler_root is None or vulkan_icd is None:
        raise ValueError(
            "--simpler-root and --vulkan-icd are required with --simpler-python"
        )
    return RemoteSimplerEnv(
        environment_name,
        python=simpler_python,
        simpler_root=simpler_root,
        vulkan_icd=vulkan_icd,
    )
