"""Execution-mode control for attention products used by all model towers."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch


_ATTENTION_IMPLEMENTATION: ContextVar[str] = ContextVar(
    "attention_implementation", default="native"
)


@contextmanager
def attention_implementation(name: str):
    if name not in {"native", "per_matrix"}:
        raise ValueError(f"unsupported attention implementation: {name}")
    token = _ATTENTION_IMPLEMENTATION.set(name)
    try:
        yield
    finally:
        _ATTENTION_IMPLEMENTATION.reset(token)


def loop_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim == right.ndim == 2:
        return torch.mm(left, right)
    leading_shape = left.shape[:-2]
    left_flat = left.reshape(-1, left.shape[-2], left.shape[-1])
    right_flat = right.reshape(-1, right.shape[-2], right.shape[-1])
    output = torch.stack(
        [torch.mm(left_flat[index], right_flat[index]) for index in range(len(left_flat))]
    )
    return output.reshape(*leading_shape, left.shape[-2], right.shape[-1])


def attention_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if _ATTENTION_IMPLEMENTATION.get() == "per_matrix":
        return loop_bmm(left, right)
    return torch.matmul(left, right)
