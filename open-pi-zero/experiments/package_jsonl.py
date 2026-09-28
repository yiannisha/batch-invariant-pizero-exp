#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 16 * 1024 * 1024


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def stream_identity(stream: BinaryIO, count_lines: bool = True) -> dict:
    digest = hashlib.sha256()
    size = 0
    lines = 0
    while chunk := stream.read(CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
        if count_lines:
            lines += chunk.count(b"\n")
    identity = {"size_bytes": size, "sha256": digest.hexdigest()}
    if count_lines:
        identity["line_count"] = lines
    return identity


def file_identity(path: Path) -> dict:
    with path.open("rb") as stream:
        return stream_identity(stream)


def archive_identity(path: Path) -> dict:
    with path.open("rb") as stream:
        compressed = stream_identity(stream, count_lines=False)
    with gzip.open(path, "rb") as stream:
        uncompressed = stream_identity(stream)
    return {"compressed": compressed, "uncompressed": uncompressed}


def package(path: Path, force: bool) -> dict:
    path = path.resolve()
    output = Path(f"{path}.gz")
    temporary = Path(f"{output}.tmp")
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")

    raw_digest = hashlib.sha256()
    raw_size = 0
    line_count = 0
    with path.open("rb") as source, temporary.open("wb") as destination:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=destination,
            compresslevel=6,
            mtime=0,
        ) as archive:
            while chunk := source.read(CHUNK_SIZE):
                raw_digest.update(chunk)
                raw_size += len(chunk)
                line_count += chunk.count(b"\n")
                archive.write(chunk)
    temporary.replace(output)

    raw = {
        "size_bytes": raw_size,
        "line_count": line_count,
        "sha256": raw_digest.hexdigest(),
    }
    observed = archive_identity(output)
    if observed["uncompressed"] != raw:
        raise RuntimeError(f"archive verification failed for {output}")
    return {
        "source": relative(path),
        "archive": relative(output),
        "format": "gzip",
        "compression_level": 6,
        "gzip_mtime": 0,
        "raw": raw,
        "compressed": observed["compressed"],
    }


def verify(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["archives"]:
        archive_path = ROOT / item["archive"]
        observed = archive_identity(archive_path)
        assert observed["compressed"] == item["compressed"]
        assert observed["uncompressed"] == item["raw"]
        source_path = ROOT / item["source"]
        if source_path.exists():
            assert file_identity(source_path) == item["raw"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="*")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/archive_manifest.json"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.manifest)
        print(args.manifest)
        return
    if not args.paths:
        parser.error("at least one JSONL path is required unless --verify-only is used")

    record = {
        "schema_version": 1,
        "archives": [package(path, args.force) for path in args.paths],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.manifest)
    verify(args.manifest)
    print(args.manifest)


if __name__ == "__main__":
    main()
