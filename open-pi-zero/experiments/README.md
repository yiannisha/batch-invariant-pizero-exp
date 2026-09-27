# Pretrained π0 batch-invariance campaign

These commands run the campaign against the public Fractal-Beta checkpoint.
All numerical runs are eager. Tracing, qualification, and timing are separate
processes.

## Checkout and assets

Clone `thinking-machines-lab/batch_invariant_ops` beside
`batch-invariant-pizero` and check out PR #27 revision
`679da28bd038c8e3d7c161b0486cac9c54c4f967`. The integration shim only
normalizes PyTorch 2.8's one-element Conv2d dispatcher parameter lists; it does
not change kernel arithmetic.

Download
`allenzren/open-pi-zero/fractal_beta_step29576_2024-12-29_13-10_42.pt` and the
PaliGemma tokenizer. Then set:

```console
export PIZERO_CHECKPOINT=/path/to/fractal_beta_step29576_2024-12-29_13-10_42.pt
export PIZERO_TOKENIZER=/path/to/paligemma-3b-pt-224
export BATCH_INVARIANT_OPS_REPO=/path/to/batch_invariant_ops
export PYTHONPATH="$BATCH_INVARIANT_OPS_REPO:$PWD"
```

## Qualification and diagnostic sequence

```console
python experiments/record_environment.py --checkpoint "$PIZERO_CHECKPOINT"
python "$BATCH_INVARIANT_OPS_REPO/scripts/check_conv2d_bmm_batch_invariance.py"
python experiments/operator_campaign.py
python experiments/policy_ablation.py --checkpoint "$PIZERO_CHECKPOINT"
python experiments/local_operator_replay.py --checkpoint "$PIZERO_CHECKPOINT"
```

The policy ablation exposes exactly these named modes: `native`,
`native_deterministic`, `existing_invariant_ops`,
`invariant_plus_patch_projection`, `explicit_per_matrix_attention`, and
`full_invariant`.

Commit the implementation after diagnostic qualification, then freeze it:

```console
python experiments/freeze_implementation.py \
  --checkpoint "$PIZERO_CHECKPOINT" \
  --ops-repository "$BATCH_INVARIANT_OPS_REPO"
```

## Replay, held-out, and closed loop

On a host exposing NVIDIA graphics/Vulkan capabilities to SAPIEN:

```console
python experiments/prepare_replay.py \
  --checkpoint "$PIZERO_CHECKPOINT" --tokenizer "$PIZERO_TOKENIZER" \
  --dataset-root /large/persistent/replay
python experiments/heldout_invariance.py \
  --checkpoint "$PIZERO_CHECKPOINT" --manifest results/replay_manifest.json
python experiments/simpler_eval.py \
  --checkpoint "$PIZERO_CHECKPOINT" --tokenizer "$PIZERO_TOKENIZER" \
  --replay-manifest results/replay_manifest.json
python experiments/analyze_simpler.py
```

Both long campaigns append one complete record at a time and accept
`--resume`. Never combine held-out records from different freeze revisions.

## Performance and paper artifacts

```console
python experiments/benchmark_kernels.py
python experiments/benchmark_policy.py --checkpoint "$PIZERO_CHECKPOINT"
python experiments/benchmark_serving.py --checkpoint "$PIZERO_CHECKPOINT"
python experiments/generate_tables.py
python experiments/generate_figures.py
```

Policy benchmark defaults implement five sessions, 20 warm-ups per session,
and 200 timed calls per session. Timing paths do not enable tensor tracing.
Raw JSONL is authoritative; tables and figures are regenerated from it.
