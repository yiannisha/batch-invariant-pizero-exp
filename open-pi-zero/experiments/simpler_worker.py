#!/usr/bin/env python3
"""Run SIMPLER behind a tiny local socket for cross-Python evaluation.

SAPIEN 2.2.2 only provides a Python 3.10 wheel, while the frozen policy may
run in a different Python environment.  Keeping the simulator in this worker
preserves both environments without converting observations or actions.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
import traceback
from pathlib import Path

import numpy as np


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
    command = request["command"]
    if command == "metadata":
        return {
            "robot_uid": environment.get_wrapper_attr("robot_uid"),
            "pid_protocol": 1,
        }, False
    if command == "reset":
        observation, info = environment.reset(
            seed=request["seed"], options=request.get("options")
        )
        return {
            "observation": _compact_observation(environment, observation),
            # Reset metadata contains sapien.Pose objects, which must not leak
            # into the policy process where SAPIEN is intentionally absent.
            "info": str(info),
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
