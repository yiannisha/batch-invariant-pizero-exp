"""Measure whether a PiZero output depends on its batch neighbors.

One fixed reference sample is placed at index zero in every batch. The
remaining entries are freshly generated distractor samples. The output at
index zero is compared with the reference sample run alone, using the same
initial flow-matching noise. Differences therefore measure batch-dependent
behavior rather than different random sampling.

Example:

    uv run python scripts/run_sample.py --batch_sizes 1 2 4 8 --num_trials 10
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.model.vla.pizero import PiZeroInference


SEED = 42


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(config, num_samples: int, dtype: torch.dtype) -> dict:
    """Create diverse, fixed-shape synthetic model inputs."""
    num_image_tokens = int(config.vision.config.num_image_tokens)
    seq_len = int(config.max_image_text_tokens)
    input_ids = torch.full(
        (num_samples, seq_len), config.pad_token_id, dtype=torch.long
    )
    input_ids[:, :num_image_tokens] = config.image_token_index
    # Give each sample a different valid synthetic text token.
    input_ids[:, num_image_tokens] = torch.arange(1, num_samples + 1)

    attention_mask = torch.zeros((num_samples, seq_len), dtype=torch.long)
    attention_mask[:, : num_image_tokens + 1] = 1
    image_size = int(config.vision.config.image_size)
    pixel_values = torch.rand(
        num_samples, 3, image_size, image_size, dtype=dtype
    ) * 2 - 1
    proprios = torch.randn(
        num_samples, int(config.cond_steps), int(config.proprio_dim), dtype=dtype
    )
    initial_action = torch.randn(
        num_samples,
        int(config.horizon_steps),
        int(config.action_dim),
        dtype=dtype,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "proprios": proprios,
        "initial_action": initial_action,
    }


def add_masks(model, inputs: dict, dtype: torch.dtype) -> dict:
    causal_mask, vlm_ids, proprio_ids, action_ids = (
        model.build_causal_mask_and_position_ids(
            inputs["attention_mask"], dtype=dtype
        )
    )
    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
        causal_mask
    )
    return {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "image_text_proprio_mask": image_text_proprio_mask,
        "action_mask": action_mask,
        "vlm_position_ids": vlm_ids,
        "proprio_position_ids": proprio_ids,
        "action_position_ids": action_ids,
        "proprios": inputs["proprios"],
        "initial_action": inputs["initial_action"],
    }


def slice_batch(inputs: dict, start: int, end: int) -> dict:
    return {key: value[start:end] for key, value in inputs.items()}


def concat_batches(first: dict, second: dict) -> dict:
    return {
        key: torch.cat((first[key], second[key]), dim=0)
        for key in first
    }


def run_model(model, inputs: dict, device: torch.device) -> tuple[torch.Tensor, float]:
    inputs = {key: value.to(device) for key, value in inputs.items()}
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs)
    synchronize(device)
    return output.float().cpu(), time.perf_counter() - start


def metrics(reference: torch.Tensor, output: torch.Tensor) -> dict:
    difference = output - reference
    reference_flat = reference.reshape(reference.shape[0], -1)
    output_flat = output.reshape(output.shape[0], -1)
    difference_flat = difference.reshape(difference.shape[0], -1)
    reference_norm = torch.linalg.vector_norm(reference_flat, dim=1)
    difference_norm = torch.linalg.vector_norm(difference_flat, dim=1)
    cosine = torch.nn.functional.cosine_similarity(
        reference_flat, output_flat, dim=1, eps=1e-12
    )
    relative_l2 = difference_norm / reference_norm.clamp_min(1e-12)
    return {
        "max_abs_diff": difference.abs().max().item(),
        "mean_abs_diff": difference.abs().mean().item(),
        "rmse": torch.sqrt(torch.mean(difference.square())).item(),
        "relative_l2_mean": relative_l2.mean().item(),
        "relative_l2_max": relative_l2.max().item(),
        "cosine_similarity_mean": cosine.mean().item(),
        "output_abs_max": output.abs().max().item(),
    }


def main(args: argparse.Namespace) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    batch_sizes = sorted(set(args.batch_sizes))
    if any(batch_size < 1 for batch_size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    if args.num_trials < 1:
        raise ValueError("num_trials must be positive")

    config = OmegaConf.load(args.config)
    model = PiZeroInference(config, use_ddp=False)
    model.eval().to(device=device, dtype=dtype)
    print(f"Random model weights, seed={SEED}, device={device}, dtype={dtype}")

    # Keep this sample fixed for every batch size and trial.
    reference_raw_inputs = make_inputs(config, 1, dtype)
    reference_inputs = add_masks(model, reference_raw_inputs, dtype)
    reference_output, reference_elapsed = run_model(
        model, reference_inputs, device
    )

    results = []
    for batch_size in batch_sizes:
        trial_metrics = []
        trial_times = []
        for _ in range(args.num_trials):
            distractors = make_inputs(config, batch_size - 1, dtype)
            trial_inputs = add_masks(
                model,
                concat_batches(reference_raw_inputs, distractors),
                dtype,
            )
            output, elapsed = run_model(model, trial_inputs, device)
            trial_metrics.append(metrics(reference_output, output[:1]))
            trial_times.append(elapsed)

        metric_names = trial_metrics[0].keys()
        result = {
            "batch_size": batch_size,
            "reference_samples": 1,
            "distractor_samples": batch_size - 1,
            "num_trials": args.num_trials,
            "reference_seconds": reference_elapsed,
            "batch_seconds_mean": float(np.mean(trial_times)),
            "batch_seconds_std": float(np.std(trial_times)),
            "throughput_speedup_vs_reference": (
                reference_elapsed * batch_size / float(np.mean(trial_times))
            ),
            "metrics_mean": {
                name: float(np.mean([trial[name] for trial in trial_metrics]))
                for name in metric_names
            },
            "metrics_worst": {
                name: float(max(trial[name] for trial in trial_metrics))
                for name in metric_names
            },
        }
        results.append(result)
        print(json.dumps(result, sort_keys=False))

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/eval/bridge.yaml", help="PiZero config YAML"
    )
    parser.add_argument(
        "--batch_sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Distractor sets to test for each batch size",
    )
    parser.add_argument("--output", default="batch_invariance_results.json")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--use_bf16", action="store_true")
    main(parser.parse_args())
