from __future__ import annotations

import json
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
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        image = get_image_from_maniskill2_obs_dict(env, observation)
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
