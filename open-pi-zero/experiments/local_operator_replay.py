#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from experiments.common import (
    implementation_context,
    index_inputs,
    load_pretrained_policy,
    make_raw_inputs,
    prepare_inputs,
    seed_everything,
    tensor_metrics,
    utc_timestamp,
    write_json,
)


def resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    value = model
    for component in path.split("."):
        if component.isdigit():
            value = value[int(component)]
        else:
            value = getattr(value, component)
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"{path!r} does not resolve to a module")
    return value


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    return value


def run_with_capture(model, inputs, module, implementation):
    captures = []

    def hook(_module, arguments, output):
        captures.append({"inputs": clone(arguments), "output": clone(output)})

    handle = module.register_forward_hook(hook)
    try:
        moved = {key: value.cuda() for key, value in inputs.items()}
        with implementation_context(implementation), torch.inference_mode():
            output = model(**moved)
            torch.cuda.synchronize()
        return output.detach(), captures
    finally:
        handle.remove()


def describe(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "rank": tensor.ndim,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/fractal_beta.yaml"))
    parser.add_argument("--module", default="proprio_encoder")
    parser.add_argument("--invocation", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--implementation", default="native")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/diagnostic/local_operator_replay.json"))
    args = parser.parse_args()

    seed_everything(args.seed)
    model, config = load_pretrained_policy(
        args.checkpoint, config_path=args.config, dtype=torch.float32
    )
    module = resolve_module(model, args.module)
    raw = make_raw_inputs(config, args.batch_size, torch.float32, args.seed)
    inputs = prepare_inputs(model, raw, torch.float32)
    singleton_inputs = index_inputs(inputs, [0])
    _, singleton_captures = run_with_capture(
        model, singleton_inputs, module, args.implementation
    )
    _, batch_captures = run_with_capture(model, inputs, module, args.implementation)
    if args.invocation >= len(singleton_captures) or args.invocation >= len(batch_captures):
        raise IndexError(
            f"invocation {args.invocation} absent; singleton={len(singleton_captures)}, "
            f"batch={len(batch_captures)}"
        )
    singleton = singleton_captures[args.invocation]
    batched = batch_captures[args.invocation]
    singleton_argument = singleton["inputs"][0]
    batched_argument = batched["inputs"][0]
    input_metrics = tensor_metrics(singleton_argument, batched_argument[:1])

    # Replay only the producing module.  If its input is already different,
    # this comparison is propagation rather than a local batch-sensitivity test.
    with implementation_context(args.implementation), torch.inference_mode():
        local_singleton = module(singleton_argument)
        local_batched = module(batched_argument)
        torch.cuda.synchronize()
    local_metrics = tensor_metrics(local_singleton, local_batched[:1])
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        with implementation_context(args.implementation), torch.inference_mode():
            module(batched_argument)
            torch.cuda.synchronize()
    dispatched = sorted(
        event.key for event in profiler.key_averages() if event.key.startswith("aten::")
    )
    record = {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "module": args.module,
        "invocation": args.invocation,
        "implementation": args.implementation,
        "batch_size": args.batch_size,
        "singleton_input": describe(singleton_argument),
        "batched_input": describe(batched_argument),
        "input_comparison": input_metrics,
        "captured_output_comparison": tensor_metrics(
            singleton["output"], batched["output"][:1]
        ),
        "local_replay_comparison": local_metrics,
        "classification": (
            "local_batch_sensitive_operator"
            if input_metrics["exact"] and not local_metrics["exact"]
            else "propagated_upstream_difference"
            if not input_metrics["exact"]
            else "locally_batch_invariant"
        ),
        "dispatched_aten_operations": dispatched,
    }
    write_json(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
