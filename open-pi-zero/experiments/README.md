# Pretrained π0 batch-invariance campaign

These commands run the campaign against the public Fractal-Beta checkpoint.
All numerical runs are eager. Tracing, qualification, and timing are separate
processes.

## Checkout and assets

Clone `thinking-machines-lab/batch_invariant_ops` beside
`batch-invariant-pizero` and check out PR #27 revision
`679da28bd038c8e3d7c161b0486cac9c54c4f967`. On Triton 3.4, apply the recorded
integration commit `14dafb5fd84350dc796a96cbecb566ca6295593f`; it selects IEEE
input precision for FP32 Triton dot products after the PR diagnostic exposed
the changed default. The policy-side integration shim normalizes PyTorch 2.8's
one-element Conv2d dispatcher parameter lists.

The H100 numerical freeze remains bound to operator commit
`14dafb5fd84350dc796a96cbecb566ca6295593f`.  RTX PRO 6000 Blackwell runs use
the launch-only portability commit
`8fdbcada1401608622a07e1d35b645973deb905d` on top: it selects two Triton
pipeline stages when the device cannot fit the H100 three-stage shared-memory
launch, without changing tiles, reduction order, input precision, or
arithmetic.  Do not regenerate the H100 freeze for that launch-only change;
the RTX runtime records retain both revisions separately.

Download
`allenzren/open-pi-zero/fractal_beta_step29576_2024-12-29_13-10_42.pt` and the
PaliGemma tokenizer. Then set:

```console
export PIZERO_CHECKPOINT=/path/to/fractal_beta_step29576_2024-12-29_13-10_42.pt
export PIZERO_TOKENIZER=/path/to/paligemma-3b-pt-224
export BATCH_INVARIANT_OPS_REPO=/path/to/batch_invariant_ops
export SIMPLER_ROOT=/path/to/SimplerEnv
export SIMPLER_PYTHON=/path/to/simpler-python-3.10
export VULKAN_ICD=/etc/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH="$BATCH_INVARIANT_OPS_REPO:$PWD"
```

The policy environment used for the retained campaign is Python 3.12 with
PyTorch 2.8.0+cu128; SAPIEN 2.2.2 runs in the separate Python 3.10 interpreter
selected by `SIMPLER_PYTHON`.  The cross-process adapter keeps NumPy 1.x and
2.x payloads compatible.

Verify that the NVIDIA ICD is native before collecting simulator data:

```console
VK_DRIVER_FILES="$VULKAN_ICD" vulkaninfo --summary
```

CUDA base images can contain the NVIDIA ICD while omitting the generic GLVND
`libEGL.so.1` loader.  If ICD creation fails for that reason, install the
distribution-matched `libegl1` loader (or prepend an extracted copy to
`LD_LIBRARY_PATH`) and continue selecting the NVIDIA ICD explicitly.  Do not
replace it with Lavapipe: SAPIEN 2.2.2 requires extensions absent from that
CPU fallback.  The retained RTX loader copy has SHA256
`875ecbb2a07d60e32216c9f102965abc1e9cc1da7789558e2e9b8b9c107e231d`.

## Qualification and diagnostic sequence

```console
python experiments/record_environment.py --checkpoint "$PIZERO_CHECKPOINT"
python "$BATCH_INVARIANT_OPS_REPO/scripts/check_conv2d_bmm_batch_invariance.py"
python experiments/operator_campaign.py
python experiments/policy_ablation.py --checkpoint "$PIZERO_CHECKPOINT"
python experiments/batch_transformations.py --checkpoint "$PIZERO_CHECKPOINT"
python experiments/local_operator_replay.py --checkpoint "$PIZERO_CHECKPOINT"
python experiments/singleton_fidelity.py --checkpoint "$PIZERO_CHECKPOINT"
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
  --freeze results/numerical_freeze.json \
  --statistics config/fractal_statistics.json \
  --dataset-root /large/persistent/replay \
  --manifest results/replay_manifest.json \
  --episodes-per-task 25 --observations-per-episode 10 \
  --simpler-python "$SIMPLER_PYTHON" --simpler-root "$SIMPLER_ROOT" \
  --vulkan-icd "$VULKAN_ICD"
python experiments/heldout_invariance.py \
  --checkpoint "$PIZERO_CHECKPOINT" --manifest results/replay_manifest.json \
  --freeze results/numerical_freeze.json \
  --statistics config/fractal_statistics.json \
  --output results/heldout/invariance.jsonl \
  --fidelity-output results/heldout/singleton_fidelity.jsonl \
  --split heldout --batch-sizes 1 2 4 8 --resume
python experiments/simpler_eval.py \
  --checkpoint "$PIZERO_CHECKPOINT" --tokenizer "$PIZERO_TOKENIZER" \
  --replay-manifest results/replay_manifest.json \
  --freeze results/numerical_freeze.json \
  --statistics config/fractal_statistics.json \
  --output results/simpler/episodes.jsonl --initializations 50 \
  --simpler-python "$SIMPLER_PYTHON" --simpler-root "$SIMPLER_ROOT" \
  --vulkan-icd "$VULKAN_ICD" --resume
python experiments/analyze_simpler.py \
  --episodes results/simpler/episodes.jsonl \
  --output results/simpler/summary.json \
  --bootstrap-seed 20250401 --bootstrap-resamples 10000
```

Both long campaigns append one complete record at a time and accept
`--resume`. Never combine held-out records from different freeze revisions.
Run them sequentially on a single GPU.  The exhaustive held-out gate is
110,400 arrangement records and 2,400 singleton-fidelity records.  The paired
closed-loop gate is 800 unique task/initialization/condition episodes.  The
closed-loop evaluator creates `results/simpler/runtime.json` at process startup
and refuses resume if its source, checkpoint, replay, repository, device, or
condition identity changes.

## Performance and paper artifacts

```console
python experiments/benchmark_kernels.py
python experiments/benchmark_policy.py --checkpoint "$PIZERO_CHECKPOINT"
python experiments/benchmark_serving.py --checkpoint "$PIZERO_CHECKPOINT"
python experiments/generate_tables.py
python experiments/generate_figures.py
python experiments/generate_report.py
python experiments/audit_results.py
```

Policy benchmark defaults implement five sessions, 20 warm-ups per session,
and 200 timed calls per session. Timing paths do not enable tensor tracing.
Raw JSONL is authoritative; tables and figures are regenerated from it.
`audit_results.py` is the final exact-cardinality and provenance gate.  A
successful complete campaign writes `results/audit.json` with status
`complete_campaign_verified`; do not infer completion solely from a process
exit or the presence of generated figures.
