#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.common import (
    load_pretrained_policy,
    make_raw_inputs,
    prepare_inputs,
    run_policy,
    seed_everything,
    tensor_metrics,
    utc_timestamp,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostic/singleton_fidelity.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    model, config = load_pretrained_policy(args.checkpoint, config_path=args.config)
    inputs = prepare_inputs(
        model, make_raw_inputs(config, 1, torch.float32, args.seed), torch.float32
    )
    native, _, _, _ = run_policy(model, inputs, "native")
    invariant, _, counts, _ = run_policy(model, inputs, "full_invariant")
    write_json(
        args.output,
        {
            "schema_version": 1,
            "timestamp": utc_timestamp(),
            "request_id": "synthetic-diagnostic/episode-000/observation-000/noise-000",
            "comparison": "full_invariant_singleton_vs_native_singleton",
            "dtype": "float32",
            "metrics": tensor_metrics(native, invariant),
            "full_invariant_override_invocations": counts,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
