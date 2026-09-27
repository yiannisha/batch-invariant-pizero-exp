#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from experiments.common import (
    append_jsonl,
    implementation_context,
    tensor_metrics,
    utc_timestamp,
)


BATCH_SIZES = (1, 2, 4, 8, 16, 31, 32, 33, 64)
ATTENTION_CASES = {
    # Shapes observed in the frozen pretrained trace.  The leading dimension
    # is request batch times attention heads.
    "qkt": {
        "heads": 8,
        "left": (4, 256),
        "right": (256, 281),
        "source": "Action attention QK^T",
    },
    "pv": {
        "heads": 8,
        "left": (4, 281),
        "right": (281, 256),
        "source": "Action attention PV",
    },
    "prefill_qkt": {
        "heads": 8,
        "left": (277, 256),
        "right": (256, 277),
        "source": "Joint prefill attention QK^T",
    },
    "prefill_pv": {
        "heads": 8,
        "left": (277, 277),
        "right": (277, 256),
        "source": "Joint prefill attention PV",
    },
    "vision_qkt": {
        "heads": 16,
        "left": (256, 72),
        "right": (72, 256),
        "source": "SigLIP attention QK^T",
    },
    "vision_pv": {
        "heads": 16,
        "left": (256, 256),
        "right": (256, 72),
        "source": "SigLIP attention PV",
    },
}
OPERATIONS = ("conv2d", *ATTENTION_CASES)


def conv_case(batch_size: int, dtype: torch.dtype, seed: int, layout: str):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(
        batch_size, 3, 224, 224, generator=generator, device="cuda", dtype=dtype
    )
    if layout == "noncontiguous":
        storage = torch.empty(
            batch_size, 3, 224, 448, device="cuda", dtype=dtype
        )
        storage[..., ::2] = values
        values = storage[..., ::2]
    weight = torch.randn(
        1152, 3, 14, 14, generator=generator, device="cuda", dtype=dtype
    )
    bias = torch.randn(1152, generator=generator, device="cuda", dtype=dtype)
    return values, weight, bias


def attention_case(
    operation: str, batch_size: int, dtype: torch.dtype, seed: int, layout: str
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    case = ATTENTION_CASES[operation]
    heads = case["heads"]
    left_shape = (batch_size * heads, *case["left"])
    right_shape = (batch_size * heads, *case["right"])
    if layout == "transposed":
        left = torch.randn(
            left_shape[0], left_shape[2], left_shape[1],
            generator=generator, device="cuda", dtype=dtype,
        ).transpose(1, 2)
        right = torch.randn(
            right_shape[0], right_shape[2], right_shape[1],
            generator=generator, device="cuda", dtype=dtype,
        ).transpose(1, 2)
    elif layout == "noncontiguous":
        left_storage = torch.randn(
            left_shape[0], left_shape[1], left_shape[2] * 2,
            generator=generator, device="cuda", dtype=dtype,
        )
        right_storage = torch.randn(
            right_shape[0], right_shape[1], right_shape[2] * 2,
            generator=generator, device="cuda", dtype=dtype,
        )
        left, right = left_storage[..., ::2], right_storage[..., ::2]
    else:
        left = torch.randn(
            *left_shape, generator=generator, device="cuda", dtype=dtype
        )
        right = torch.randn(
            *right_shape, generator=generator, device="cuda", dtype=dtype
        )
    return left, right


def run_one(
    operation: str,
    implementation: str,
    batch_size: int,
    dtype: torch.dtype,
    seed: int,
    layout: str,
) -> dict:
    torch.manual_seed(seed)
    if operation == "conv2d":
        values, weight, bias = conv_case(batch_size, dtype, seed, layout)

        def calculate(x):
            return F.conv2d(x, weight, bias, stride=14)

        target_input, batched_input = values[:1], values
        input_shapes = [list(values.shape), list(weight.shape)]
        input_strides = [list(values.stride()), list(weight.stride())]
    else:
        left, right = attention_case(operation, batch_size, dtype, seed, layout)

        def calculate(pair):
            return torch.bmm(*pair)

        heads = ATTENTION_CASES[operation]["heads"]
        target_input, batched_input = (left[:heads], right[:heads]), (left, right)
        input_shapes = [list(left.shape), list(right.shape)]
        input_strides = [list(left.stride()), list(right.stride())]

    with implementation_context(implementation) as overrides:
        singleton = calculate(target_input)
        batched = calculate(batched_input)
        torch.cuda.synchronize()
    target_count = 1 if operation == "conv2d" else ATTENTION_CASES[operation]["heads"]
    invariance = tensor_metrics(singleton, batched[:target_count])

    with implementation_context("native"):
        native = calculate(batched_input)
        torch.cuda.synchronize()
    fidelity = tensor_metrics(native, batched)
    record = {
        "schema_version": 1,
        "experiment_id": str(uuid.uuid4()),
        "timestamp": utc_timestamp(),
        "operator": operation,
        "source_operation": (
            "SigLIP patch projection"
            if operation == "conv2d"
            else ATTENTION_CASES[operation]["source"]
        ),
        "dispatched_aten_operation": "aten::convolution"
        if operation == "conv2d"
        else "aten::bmm",
        "implementation": implementation,
        "batch_size": batch_size,
        "seed": seed,
        "dtype": str(dtype).removeprefix("torch."),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "layout": layout,
        "input_shapes": input_shapes,
        "input_ranks": [len(shape) for shape in input_shapes],
        "input_strides": input_strides,
        "input_contiguous": (
            [values.is_contiguous(), weight.is_contiguous()]
            if operation == "conv2d"
            else [left.is_contiguous(), right.is_contiguous()]
        ),
        "override_invocations": overrides.counts,
        "invariance": invariance,
        "fidelity_to_native": fidelity,
    }
    if seed == 0 and batch_size == 1 and dtype == torch.float32:
        if operation == "conv2d":
            reference = F.conv2d(
                values[:1].double(), weight.double(), bias.double(), stride=14
            )
        else:
            reference = torch.bmm(
                left[:target_count].double(), right[:target_count].double()
            )
        record["fidelity_to_fp64"] = tensor_metrics(reference, batched[:target_count])
    return record


def dispatch_report() -> list[dict]:
    report = []
    mm_inputs = (
        torch.randn(4, 281, device="cuda"),
        torch.randn(281, 256, device="cuda"),
    )
    bmm_inputs = (
        torch.randn(8, 4, 281, device="cuda"),
        torch.randn(8, 281, 256, device="cuda"),
    )
    convolution_inputs = (
        torch.randn(1, 3, 224, 224, device="cuda"),
        torch.randn(1152, 3, 14, 14, device="cuda"),
    )
    cases = {
        "rank2_mm": (lambda: torch.mm(*mm_inputs), mm_inputs, "mm"),
        "rank3_attention": (lambda: torch.bmm(*bmm_inputs), bmm_inputs, "bmm"),
        "siglip_projection": (
            lambda: F.conv2d(*convolution_inputs, stride=14),
            convolution_inputs,
            "convolution",
        ),
    }
    expected = {
        "rank2_mm": "aten::mm",
        "rank3_attention": "aten::bmm",
        "siglip_projection": "aten::convolution",
    }
    for name, (case, inputs, override_name) in cases.items():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            case()
            torch.cuda.synchronize()
        operators = sorted(
            {event.key for event in prof.key_averages() if event.key.startswith("aten::")}
        )
        if expected[name] not in operators:
            raise RuntimeError(f"{name} expected {expected[name]}, observed {operators}")
        with implementation_context("full_invariant") as overrides:
            case()
            torch.cuda.synchronize()
        if overrides.counts[override_name] == 0:
            raise RuntimeError(f"{name} silently bypassed invariant {override_name}")
        report.append(
            {
                "source_operation": name,
                "dispatched_aten_operation": expected[name],
                "tensor_ranks": [tensor.ndim for tensor in inputs],
                "shapes": [list(tensor.shape) for tensor in inputs],
                "strides": [list(tensor.stride()) for tensor in inputs],
                "dtype": str(inputs[0].dtype).removeprefix("torch."),
                "observed_operators": operators,
                "invariant_path_selected": True,
                "override_invocations": overrides.counts[override_name],
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/operator/raw.jsonl"))
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument(
        "--dtypes", nargs="+", choices=("float32", "bfloat16", "float16"), default=("float32", "bfloat16")
    )
    parser.add_argument("--layouts", nargs="+", default=("contiguous", "transposed", "noncontiguous"))
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=OPERATIONS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --resume")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and args.output.exists():
        with args.output.open() as stream:
            for line in stream:
                record = json.loads(line)
                completed.add(
                    (
                        record["operator"], record["implementation"],
                        record["batch_size"], record["dtype"],
                        record["seed"], record["layout"],
                    )
                )
    dispatch_path = args.output.with_name("dispatch_report.json")
    from experiments.common import write_json

    write_json(dispatch_path, dispatch_report())
    dtypes = {name: getattr(torch, name) for name in args.dtypes}
    written = 0
    for operation, implementation, batch_size, dtype_name, seed, layout in itertools.product(
        args.operations,
        ("native", "full_invariant"),
        args.batch_sizes,
        args.dtypes,
        range(args.seeds),
        args.layouts,
    ):
        # Transposed is meaningful for matrix operands; non-contiguous covers Conv2d.
        if operation == "conv2d" and layout == "transposed":
            continue
        key = (operation, implementation, batch_size, dtype_name, seed, layout)
        if key in completed:
            continue
        append_jsonl(
            args.output,
            [run_one(operation, implementation, batch_size, dtypes[dtype_name], seed, layout)],
        )
        written += 1
        if written % 100 == 0:
            print(f"completed {len(completed) + written} records", flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
