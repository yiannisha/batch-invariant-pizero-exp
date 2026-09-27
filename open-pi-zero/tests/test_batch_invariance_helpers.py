import torch

from experiments.common import load_numerical_freeze
from experiments.heldout_invariance import completed_fidelity_keys
from src.model.paligemma.siglip import _UnfoldConv2d
from src.model.attention import attention_implementation, attention_matmul
from src.utils.trace import trace_context


def test_unfold_conv_matches_torch_conv2d():
    torch.manual_seed(0)
    reference = torch.nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
    candidate = _UnfoldConv2d(3, 5, kernel_size=3, stride=2, padding=1)
    candidate.load_state_dict(reference.state_dict())

    inputs = torch.randn(2, 3, 9, 11)
    torch.testing.assert_close(candidate(inputs), reference(inputs))


def test_unfold_conv_is_independent_of_batch_size():
    torch.manual_seed(0)
    conv = _UnfoldConv2d(3, 5, kernel_size=3, stride=2, padding=1)
    inputs = torch.randn(8, 3, 9, 11)

    reference = conv(inputs[:1])
    for batch_size in (1, 2, 4, 8):
        torch.testing.assert_close(reference, conv(inputs[:batch_size])[:1])


def test_trace_context_records_nested_tensors():
    value = torch.tensor([1.0])
    with trace_context() as trace:
        from src.utils.trace import GLOBAL_TRACE

        GLOBAL_TRACE.record("nested", {"value": value})

    assert torch.equal(trace["nested#invocation_0"]["value"], value)
    assert trace["nested#invocation_0"]["value"] is not value


def test_trace_context_preserves_repeated_invocations():
    with trace_context() as trace:
        from src.utils.trace import GLOBAL_TRACE

        GLOBAL_TRACE.record("repeated", torch.tensor([1]))
        GLOBAL_TRACE.record("repeated", torch.tensor([2]))

    assert trace["repeated#invocation_0"].item() == 1
    assert trace["repeated#invocation_1"].item() == 2


def test_trace_context_can_filter_large_model_traces():
    with trace_context(include_prefixes=("flow.",)) as trace:
        from src.utils.trace import GLOBAL_TRACE

        GLOBAL_TRACE.record("vision.hidden", torch.tensor([1]))
        GLOBAL_TRACE.record("flow.step_0.action_state", torch.tensor([2]))

    assert list(trace) == ["flow.step_0.action_state#invocation_0"]


def test_per_matrix_attention_matches_native():
    torch.manual_seed(0)
    left = torch.randn(2, 3, 4, 5)
    right = torch.randn(2, 3, 5, 6)
    expected = torch.matmul(left, right)
    with attention_implementation("per_matrix"):
        actual = attention_matmul(left, right)
    torch.testing.assert_close(actual, expected)


def test_completed_fidelity_keys_supports_resume(tmp_path):
    path = tmp_path / "fidelity.jsonl"
    path.write_text(
        '{"request_id":"request-1"}\n{"request_id":"request-2"}\n'
    )
    assert completed_fidelity_keys(path) == {"request-1", "request-2"}
    assert completed_fidelity_keys(tmp_path / "missing.jsonl") == set()


def test_numerical_freeze_validates_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    import hashlib
    import json

    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "numerical_implementation_frozen": True,
                "dirty_status": [],
                "checkpoint": {
                    "sha256": hashlib.sha256(b"checkpoint").hexdigest()
                },
            }
        )
    )
    assert load_numerical_freeze(freeze, checkpoint)[
        "numerical_implementation_frozen"
    ]
