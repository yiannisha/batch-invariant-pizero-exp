#!/usr/bin/env python3
"""Run SIMPLER behind a tiny local socket for cross-Python evaluation.

SAPIEN 2.2.2 only provides a Python 3.10 wheel, while the frozen policy may
run in a different Python environment.  Keeping the simulator in this worker
preserves both environments without converting observations or actions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import pickle
import random
import socket
import struct
import traceback
from pathlib import Path

import numpy as np
import gymnasium as gym


_snapshot = None
_reset_observation = None
_reset_request = None
_TASK_STATE_FIELDS = (
    "consecutive_grasp", "lifted_obj", "obj_height_after_settle", "episode_stats"
)


def _equal(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, "p") and hasattr(left, "q"):
        return _equal(np.asarray(left.p), np.asarray(right.p)) and _equal(np.asarray(left.q), np.asarray(right.q))
    return left == right


def _digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _observe(environment):
    base = environment.unwrapped
    observation = base.get_obs()
    wrappers = []
    wrapper = environment
    while wrapper is not base:
        if isinstance(wrapper, gym.ObservationWrapper):
            wrappers.append(wrapper)
        wrapper = wrapper.env
    for wrapper in reversed(wrappers):
        observation = wrapper.observation(observation)
    return _compact_observation(environment, observation)


def _capture(environment):
    global _snapshot
    if _reset_observation is None:
        raise RuntimeError("reset must precede snapshot capture")
    base = environment.unwrapped
    _snapshot = {
        "reset_request": copy.deepcopy(_reset_request),
        "scene_pack": copy.deepcopy(base._scene.pack()),
        "agent_state": copy.deepcopy(base.agent.get_state()),
        "task_state": {
            name: copy.deepcopy(getattr(base, name))
            for name in _TASK_STATE_FIELDS if hasattr(base, name)
        },
        "elapsed_steps": base.elapsed_steps,
        "main_rng": copy.deepcopy(base._main_rng.get_state()),
        "episode_rng": copy.deepcopy(base._episode_rng.get_state()),
        "numpy_rng": copy.deepcopy(np.random.get_state()),
        "python_rng": copy.deepcopy(random.getstate()),
        "instruction": environment.get_wrapper_attr("get_language_instruction")(),
    }
    # SAPIEN normalizes some poses on unpack. Define the common starting state
    # after one unpack, then require every later unpack to reproduce it exactly.
    base._scene.unpack(copy.deepcopy(_snapshot["scene_pack"]))
    _snapshot["sim_state"] = copy.deepcopy(base.get_state())
    _snapshot["agent_state"] = copy.deepcopy(base.agent.get_state())
    _snapshot["observation"] = _observe(environment)
    return {"sim_state_sha256": _digest(_snapshot["sim_state"]), "observation_sha256": _digest(_snapshot["observation"])}


def _restore(environment):
    if _snapshot is None:
        raise RuntimeError("no captured snapshot")
    # Reset task-specific counters and success predicates, then replace the
    # newly settled physics state with the one shared by every condition.
    reset_request = copy.deepcopy(_snapshot["reset_request"])
    reset_request["options"] = dict(reset_request["options"] or {})
    reset_request["options"]["reconfigure"] = True
    environment.reset(**reset_request)
    base = environment.unwrapped
    base._scene.unpack(copy.deepcopy(_snapshot["scene_pack"]))
    base.agent.controller.set_state(copy.deepcopy(_snapshot["agent_state"]["controller"]))
    base.agent.robot.set_qacc(copy.deepcopy(_snapshot["agent_state"]["robot_qacc"]))
    for name, value in _snapshot["task_state"].items():
        setattr(base, name, copy.deepcopy(value))
    base._elapsed_steps = _snapshot["elapsed_steps"]
    base._main_rng.set_state(copy.deepcopy(_snapshot["main_rng"]))
    base._episode_rng.set_state(copy.deepcopy(_snapshot["episode_rng"]))
    np.random.set_state(copy.deepcopy(_snapshot["numpy_rng"]))
    random.setstate(copy.deepcopy(_snapshot["python_rng"]))
    observation = _observe(environment)
    checks = {
        "sim_state": _equal(base.get_state(), _snapshot["sim_state"]),
        "agent_state": _equal(base.agent.get_state(), _snapshot["agent_state"]),
        "task_state": all(_equal(getattr(base, name), value) for name, value in _snapshot["task_state"].items()),
        "elapsed_steps": base.elapsed_steps == _snapshot["elapsed_steps"],
        "main_rng": _equal(base._main_rng.get_state(), _snapshot["main_rng"]),
        "episode_rng": _equal(base._episode_rng.get_state(), _snapshot["episode_rng"]),
        "numpy_rng": _equal(np.random.get_state(), _snapshot["numpy_rng"]),
        "python_rng": _equal(random.getstate(), _snapshot["python_rng"]),
        "observation": _equal(observation, _snapshot["observation"]),
        "instruction": environment.get_wrapper_attr("get_language_instruction")() == _snapshot["instruction"],
    }
    if not all(checks.values()):
        current_state = np.asarray(base.get_state())
        saved_state = np.asarray(_snapshot["sim_state"])
        mismatch = np.flatnonzero(current_state != saved_state)
        detail = {
            "sim_mismatch_count": int(len(mismatch)),
            "sim_first_mismatches": [(int(i), float(saved_state[i]), float(current_state[i])) for i in mismatch[:10]],
            "observation_keys": list(observation["image"].values())[0].keys(),
            "rgb_mismatch_count": int(np.count_nonzero(list(observation["image"].values())[0]["rgb"] != list(_snapshot["observation"]["image"].values())[0]["rgb"])),
        }
        raise RuntimeError(f"snapshot restore failed exact checks: {checks}; detail={detail}")
    return {"observation": observation, "checks": checks, "sim_state_sha256": _digest(base.get_state()), "observation_sha256": _digest(observation)}


def _receive(connection: socket.socket):
    header = _receive_exact(connection, 8)
    size = struct.unpack("!Q", header)[0]
    return pickle.loads(_receive_exact(connection, size))


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("SIMPLER client disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send(connection: socket.socket, value) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def _compact_observation(environment, observation: dict) -> dict:
    robot_uid = environment.get_wrapper_attr("robot_uid")
    camera = "overhead_camera" if "google_robot" in robot_uid else "3rd_view_camera"
    return {
        "agent": observation["agent"],
        "extra": observation.get("extra", {}),
        "image": {camera: {"rgb": observation["image"][camera]["rgb"]}},
    }


def _handle(environment, request: dict) -> tuple[dict, bool]:
    global _reset_observation, _snapshot, _reset_request
    command = request["command"]
    if command == "metadata":
        return {
            "robot_uid": environment.get_wrapper_attr("robot_uid"),
            "pid_protocol": 1,
        }, False
    if command == "reset":
        _snapshot = None
        _reset_request = {"seed": request["seed"], "options": request.get("options")}
        observation, info = environment.reset(
            seed=request["seed"], options=request.get("options")
        )
        _reset_observation = _compact_observation(environment, observation)
        return {
            "observation": _reset_observation,
            # Reset metadata contains sapien.Pose objects, which must not leak
            # into the policy process where SAPIEN is intentionally absent.
            "info": str(info),
        }, False
    if command == "capture_snapshot":
        return _capture(environment), False
    if command == "restore_snapshot":
        return _restore(environment), False
    if command == "inspect_state":
        base = environment.unwrapped
        return {
            "sim_state": base.get_state(),
            "task_state": {
                name: copy.deepcopy(getattr(base, name))
                for name in _TASK_STATE_FIELDS if hasattr(base, name)
            },
        }, False
    if command == "step":
        observation, reward, success, truncated, info = environment.step(
            np.asarray(request["action"], dtype=np.float64)
        )
        return {
            "observation": _compact_observation(environment, observation),
            "reward": reward,
            "success": success,
            "truncated": truncated,
            "info": str(info),
        }, False
    if command == "instruction":
        instruction = environment.get_wrapper_attr("get_language_instruction")()
        return {"instruction": instruction}, False
    if command == "close":
        environment.close()
        return {"closed": True}, True
    raise ValueError(f"unknown command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    args = parser.parse_args()

    import simpler_env

    environment = simpler_env.make(args.environment)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(1)
    connection, _ = server.accept()
    try:
        while True:
            try:
                request = _receive(connection)
            except EOFError:
                break
            try:
                result, finished = _handle(environment, request)
                _send(connection, {"ok": True, "result": result})
            except Exception as error:  # relay the original worker traceback
                _send(
                    connection,
                    {
                        "ok": False,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                finished = False
            if finished:
                break
    finally:
        connection.close()
        server.close()


if __name__ == "__main__":
    main()
