"""Opt-in tensor tracing for the batch-invariance experiment."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch


_ACTIVE_TRACE: ContextVar[dict | None] = ContextVar("active_trace", default=None)
_TRACE_SCOPE: ContextVar[tuple[str, ...]] = ContextVar("trace_scope", default=())
_TRACE_INVOCATIONS: ContextVar[dict[str, int] | None] = ContextVar(
    "trace_invocations", default=None
)
_TRACE_PREFIXES: ContextVar[tuple[str, ...] | None] = ContextVar(
    "trace_prefixes", default=None
)


@contextmanager
def trace_context(
    trace_dict: dict | None = None,
    include_prefixes: tuple[str, ...] | None = None,
) -> Iterator[dict]:
    trace = {} if trace_dict is None else trace_dict
    trace_token = _ACTIVE_TRACE.set(trace)
    invocation_token = _TRACE_INVOCATIONS.set({})
    prefix_token = _TRACE_PREFIXES.set(include_prefixes)
    try:
        yield trace
    finally:
        _TRACE_PREFIXES.reset(prefix_token)
        _TRACE_INVOCATIONS.reset(invocation_token)
        _ACTIVE_TRACE.reset(trace_token)


@contextmanager
def trace_scope(name: str) -> Iterator[None]:
    """Qualify records made by a repeated model invocation.

    Flow steps and prefill execute the same modules repeatedly.  A scope plus
    an invocation suffix prevents later tensors from overwriting earlier ones.
    """

    token = _TRACE_SCOPE.set((*_TRACE_SCOPE.get(), name))
    try:
        yield
    finally:
        _TRACE_SCOPE.reset(token)


def get_trace() -> dict | None:
    return _ACTIVE_TRACE.get()


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        # Trace snapshots are only used after the model call for comparison.
        # Keeping them on the model device means every recorded activation is
        # retained in VRAM until the call finishes, which becomes prohibitive
        # for larger batch sizes. Copy directly to CPU instead.
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


class _GlobalTrace:
    def record(self, name: str, value: Any) -> None:
        trace = get_trace()
        if trace is not None:
            scope = _TRACE_SCOPE.get()
            qualified = ".".join((*scope, name)) if scope else name
            prefixes = _TRACE_PREFIXES.get()
            if prefixes is not None and not any(
                name.startswith(prefix) or qualified.startswith(prefix)
                for prefix in prefixes
            ):
                return
            invocations = _TRACE_INVOCATIONS.get()
            if invocations is None:
                invocations = {}
                _TRACE_INVOCATIONS.set(invocations)
            invocation = invocations.get(qualified, 0)
            invocations[qualified] = invocation + 1
            trace[f"{qualified}#invocation_{invocation}"] = _clone(value)


GLOBAL_TRACE = _GlobalTrace()
