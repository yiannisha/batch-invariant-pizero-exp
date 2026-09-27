#!/usr/bin/env python3
"""Disable SAPIEN 2.2.2's unused CUDA semaphore-fd requirement.

Lavapipe can render SIMPLER offscreen but does not advertise
VK_KHR_external_semaphore_fd.  SAPIEN requests that extension even when the
selected device has no CUDA interop.  Replacing it with the already-requested
base extension leaves ordinary Vulkan rendering unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


NEEDLE = b"VK_KHR_external_semaphore_fd"
REPLACEMENT = b"VK_KHR_external_semaphore\0\0\0"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    data = args.library.read_bytes()
    occurrences = data.count(NEEDLE)
    if occurrences == 0 and REPLACEMENT in data:
        print(f"already patched: {args.library} sha256={digest(data)}")
        return
    if occurrences != 1:
        raise RuntimeError(
            f"expected exactly one semaphore-fd extension in {args.library}, "
            f"found {occurrences}"
        )
    patched = data.replace(NEEDLE, REPLACEMENT)
    if len(patched) != len(data):
        raise RuntimeError("binary patch changed the library size")
    args.library.write_bytes(patched)
    print(
        f"patched: {args.library} before={digest(data)} after={digest(patched)}"
    )


if __name__ == "__main__":
    main()
