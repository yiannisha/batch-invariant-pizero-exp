from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# CUDA requires this to be present before the first cuBLAS handle is created.
# It does not enable deterministic algorithms by itself; the
# native_deterministic context does that explicitly.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from omegaconf import OmegaConf

# Pin rather than inherit framework-version defaults.  The primary policy
# configuration preserves cuDNN's TF32 convolution path while requiring IEEE
# FP32 for matmul; both choices are recorded in results/environment.json.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("highest")

from src.model.attention import attention_implementation
from src.model.vla.pizero import PiZeroInference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "fractal_beta.yaml"
IMPLEMENTATIONS = (
    "native",
    "native_deterministic",
    "existing_invariant_ops",
    "invariant_plus_patch_projection",
    "explicit_per_matrix_attention",
    "full_invariant",
)


@dataclass(frozen=True)
class Implementation:
    name: str
    operations: tuple[str, ...]
    attention: str = "native"
    deterministic: bool = False


IMPLEMENTATION_CONFIGS = {
    "native": Implementation("native", ()),
    "native_deterministic": Implementation(
        "native_deterministic", (), deterministic=True
    ),
    "existing_invariant_ops": Implementation(
        "existing_invariant_ops",
        ("mm", "addmm", "log_softmax", "mean"),
    ),
    "invariant_plus_patch_projection": Implementation(
        "invariant_plus_patch_projection",
        ("mm", "addmm", "log_softmax", "mean", "convolution"),
    ),
    "explicit_per_matrix_attention": Implementation(
        "explicit_per_matrix_attention",
        ("mm", "addmm", "log_softmax", "mean", "convolution"),
        attention="per_matrix",
    ),
    "full_invariant": Implementation(
        "full_invariant",
        (
            "mm",
            "addmm",
            "bmm",
            "convolution",
            "log_softmax",
            "mean",
        ),
    ),
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_sha(path: Path) -> str | None:
    return run_text(["git", "rev-parse", "HEAD"], path)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device | str = "cuda") -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    # Keep large operator tensors on their current device and transfer only
    # scalar statistics.  Trace snapshots and policy outputs are already on
    # CPU; operator qualification otherwise spent most of its time copying
    # multi-million-element batches to the host.
    reference = reference.detach().to(torch.float64, copy=True)
    candidate = candidate.detach().to(torch.float64, copy=True)
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "exact": False,
        }
    exact_mask = torch.eq(reference, candidate)
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    unequal = int((~exact_mask).sum().item())
    nonfinite = int((~finite).sum().item())
    difference = torch.where(finite, candidate - reference, torch.nan)
    finite_difference = difference[finite]
    if finite_difference.numel():
        absolute = finite_difference.abs()
        max_abs = float(absolute.max().item())
        mean_abs = float(absolute.mean().item())
        rmse = float(torch.sqrt(torch.mean(finite_difference.square())).item())
        difference_norm = torch.linalg.vector_norm(finite_difference)
        reference_norm = torch.linalg.vector_norm(reference[finite])
        relative_l2 = float(
            (difference_norm / reference_norm.clamp_min(1e-300)).item()
        )
    else:
        max_abs = mean_abs = rmse = relative_l2 = math.nan
    return {
        "shape_equal": True,
        "shape": list(reference.shape),
        "exact": bool(torch.equal(reference, candidate)),
        "unequal_element_count": unequal,
        "unequal_element_fraction": unequal / reference.numel() if reference.numel() else 0.0,
        "max_absolute_error": max_abs,
        "mean_absolute_error": mean_abs,
        "rmse": rmse,
        "relative_l2_error": relative_l2,
        "nonfinite_count": nonfinite,
    }


def flatten_tensors(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_tensors(item, name)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from flatten_tensors(item, f"{prefix}[{index}]")


def compare_traces(reference: dict, candidate: dict, sample_index: int = 0) -> dict:
    reference_tensors = dict(flatten_tensors(reference))
    candidate_tensors = dict(flatten_tensors(candidate))
    comparisons = []
    first_divergence = None
    for order, (name, reference_tensor) in enumerate(reference_tensors.items()):
        candidate_tensor = candidate_tensors.get(name)
        if candidate_tensor is None:
            metric = {"exact": False, "error": "missing candidate tensor"}
        elif reference_tensor.ndim and candidate_tensor.ndim:
            metric = tensor_metrics(
                reference_tensor[:1], candidate_tensor[sample_index : sample_index + 1]
            )
        else:
            metric = tensor_metrics(reference_tensor, candidate_tensor)
        entry = {"order": order, "trace_key": name, **metric}
        comparisons.append(entry)
        if first_divergence is None and not metric.get("exact", False):
            first_divergence = entry
    return {"first_divergence": first_divergence, "comparisons": comparisons}


def make_raw_inputs(
    config,
    count: int,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image_tokens = int(config.vision.config.num_image_tokens)
    sequence_length = int(config.max_image_text_tokens)
    input_ids = torch.full(
        (count, sequence_length), int(config.pad_token_id), dtype=torch.long
    )
    input_ids[:, :image_tokens] = int(config.image_token_index)
    input_ids[:, image_tokens] = torch.arange(1, count + 1) + 100
    attention_mask = torch.zeros((count, sequence_length), dtype=torch.long)
    attention_mask[:, : image_tokens + 1] = 1
    image_size = int(config.vision.config.image_size)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": torch.rand(
            count, 3, image_size, image_size, generator=generator, dtype=dtype
        )
        * 2
        - 1,
        "proprios": torch.rand(
            count,
            int(config.cond_steps),
            int(config.proprio_dim),
            generator=generator,
            dtype=dtype,
        )
        * 2
        - 1,
        "initial_action": torch.randn(
            count,
            int(config.horizon_steps),
            int(config.action_dim),
            generator=generator,
            dtype=dtype,
        ),
    }


def prepare_inputs(model: PiZeroInference, raw: dict, dtype: torch.dtype) -> dict:
    causal_mask, vlm_ids, proprio_ids, action_ids = (
        model.build_causal_mask_and_position_ids(raw["attention_mask"], dtype=dtype)
    )
    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
        causal_mask
    )
    return {
        "input_ids": raw["input_ids"],
        "pixel_values": raw["pixel_values"],
        "image_text_proprio_mask": image_text_proprio_mask,
        "action_mask": action_mask,
        "vlm_position_ids": vlm_ids,
        "proprio_position_ids": proprio_ids,
        "action_position_ids": action_ids,
        "proprios": raw["proprios"],
        "initial_action": raw["initial_action"],
    }


def index_inputs(inputs: dict[str, torch.Tensor], indices: list[int]) -> dict:
    return {key: value[indices] for key, value in inputs.items()}


def load_config(path: Path | str = DEFAULT_CONFIG):
    config = OmegaConf.load(path)
    if int(config.horizon_steps) != 4 or int(config.action_dim) != 7:
        raise RuntimeError("Fractal-Beta must predict four seven-dimensional actions")
    if int(config.num_inference_steps) != 10:
        raise RuntimeError("Fractal-Beta matching config must use ten flow steps")
    if int(config.act_steps) != 2:
        raise RuntimeError("Fractal-Beta controller must execute the first two actions")
    return config


def load_pretrained_policy(
    checkpoint: Path | str,
    *,
    config_path: Path | str = DEFAULT_CONFIG,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[PiZeroInference, Any]:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = load_config(config_path)
    model = PiZeroInference(config, use_ddp=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = payload["model"] if "model" in payload else payload
    state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/config mismatch: missing={missing}, unexpected={unexpected}"
        )
    del payload, state
    model.requires_grad_(False).eval().to(device=device, dtype=dtype)
    return model, config


class SelectiveInvariantOperators:
    """Register only the operator subset belonging to one causal ablation."""

    def __init__(self, operations: tuple[str, ...]):
        self.operations = operations
        self.library = None
        self.counts = {name: 0 for name in operations}

    def _counted(self, name: str, function):
        def call(*args, **kwargs):
            self.counts[name] += 1
            return function(*args, **kwargs)

        return call

    def __enter__(self):
        if not self.operations:
            return self
        from batch_invariant_ops import batch_invariant_ops as invariant_ops

        convolution_function = getattr(
            invariant_ops,
            "convolution_batch_invariant",
            invariant_ops.conv2d_batch_invariant,
        )

        def compatible_convolution(
            input, weight, bias, stride, padding, dilation, transposed,
            output_padding, groups,
        ):
            """Normalize PyTorch 2.8's one-element Conv2d parameter lists.

            PR #27 was authored against a dispatcher version that supplied
            two-element lists.  The arithmetic kernel is unchanged.
            """

            def pair(values):
                values = tuple(values)
                return values * 2 if len(values) == 1 else values

            return convolution_function(
                input,
                weight,
                bias,
                pair(stride),
                pair(padding),
                pair(dilation),
                transposed,
                pair(output_padding),
                groups,
            )

        functions = {
            "mm": invariant_ops.mm_batch_invariant,
            "addmm": invariant_ops.addmm_batch_invariant,
            "bmm": invariant_ops.bmm_batch_invariant,
            "convolution": compatible_convolution,
            "log_softmax": invariant_ops._log_softmax_batch_invariant,
            "mean": invariant_ops.mean_batch_invariant,
        }
        schemas = {
            "mm": "aten::mm",
            "addmm": "aten::addmm",
            "bmm": "aten::bmm",
            "convolution": "aten::convolution",
            "log_softmax": "aten::_log_softmax",
            "mean": "aten::mean.dim",
        }
        self.library = torch.library.Library("aten", "IMPL")
        for name in self.operations:
            self.library.impl(
                schemas[name], self._counted(name, functions[name]), "CUDA"
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.library is not None:
            self.library._destroy()
            self.library = None

    def assert_coverage(self) -> None:
        required = {"convolution", "bmm"}.intersection(self.operations)
        missing = sorted(name for name in required if self.counts[name] == 0)
        if {"mm", "addmm"}.intersection(self.operations) and not (
            self.counts.get("mm", 0) or self.counts.get("addmm", 0)
        ):
            missing.append("mm/addmm")
        if missing:
            raise RuntimeError(
                "invariant overrides were silently bypassed: " + ", ".join(missing)
            )


@contextlib.contextmanager
def implementation_context(name: str) -> Iterator[SelectiveInvariantOperators]:
    if name not in IMPLEMENTATION_CONFIGS:
        raise ValueError(f"unknown implementation {name!r}; choose {IMPLEMENTATIONS}")
    config = IMPLEMENTATION_CONFIGS[name]
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_tf32_cudnn = torch.backends.cudnn.allow_tf32
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    operators = SelectiveInvariantOperators(config.operations)
    try:
        with operators, attention_implementation(config.attention):
            yield operators
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32_matmul
        torch.backends.cudnn.allow_tf32 = previous_tf32_cudnn


def run_policy(
    model: PiZeroInference,
    inputs: dict[str, torch.Tensor],
    implementation: str,
    *,
    device: str = "cuda",
    trace: bool = False,
    trace_prefixes: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, dict | None, dict[str, int], float]:
    from src.utils.trace import trace_context

    moved = {key: value.to(device) for key, value in inputs.items()}
    trace_data = {} if trace else None
    synchronize(device)
    start = time.perf_counter()
    with implementation_context(implementation) as operators, torch.inference_mode():
        if trace:
            with trace_context(trace_data, include_prefixes=trace_prefixes):
                output = model(**moved)
        else:
            output = model(**moved)
        synchronize(device)
        elapsed = time.perf_counter() - start
        if implementation == "full_invariant":
            operators.assert_coverage()
    return output.detach().cpu(), trace_data, dict(operators.counts), elapsed


def append_jsonl(path: Path | str, records: Iterator[dict] | list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    temporary.replace(path)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
