# Batch-invariant PiZero experiments

This repository studies a subtle inference failure mode: the output for one
example can change when unrelated examples are added to the same batch. It
contains a lightweight synthetic sample and a retained pretrained Fractal-Beta
campaign covering operator qualification, policy ablations, replay evaluation,
performance, serving, and paired SIMPLER closed-loop behavior.

The repository is deliberately an experiment, not a claim of cross-device or
cross-software bitwise determinism. Its numerical contract is narrower: on the
qualified software/hardware revisions, unrelated requests must not change a
fixed request's output. Environment, checkpoint, source, and runtime identities
are retained alongside the measurements.

## Repository map

- `PLAN.md` — the prespecified campaign plan and append-only execution log.
- `COMMIT_SHA_MAP.md` — maps commit IDs embedded in frozen provenance records
  to their equivalents after the authorship-metadata normalization.
- `batch_invariant_ops/` — the exact CUDA/Triton operator source used by the
  completed campaign, including the RTX launch adaptation.
- `open-pi-zero/` — the PiZero implementation and reproducible experiment.
- `open-pi-zero/experiments/` — pretrained campaign runners, analysis,
  artifact generation, and the exact final audit.
- `open-pi-zero/results/` — machine-readable measurements and generated report.
- `open-pi-zero/artifacts/` — generated tables and figures.
- `open-pi-zero/scripts/run_sample.py` — compares a fixed sample alone with
  that sample at index zero in larger batches and records intermediate errors.

External checkpoints, replay tensors, and SIMPLER assets are not vendored.
Their hashes/revisions and the commands that consume them are retained in the
experiment results and runbook.

The completed result set is committed, including deterministic compressed
archives of the large held-out and SIMPLER raw records. The corresponding
uncompressed JSONL files remain intentionally untracked; their SHA-256 hashes
and record counts are recorded in
[`open-pi-zero/results/archive_manifest.json`](open-pi-zero/results/archive_manifest.json).

## Read the results

The paper-facing result narrative is in
[`open-pi-zero/results/EXPERIMENT_REPORT.md`](open-pi-zero/results/EXPERIMENT_REPORT.md).
The machine-readable completion gate is
[`open-pi-zero/results/audit.json`](open-pi-zero/results/audit.json), whose
status is `complete_campaign_verified`. Generated tables and figures are under
[`open-pi-zero/artifacts/`](open-pi-zero/artifacts/).

## Run the lightweight sample

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run
the sample from the `open-pi-zero` project directory:

```console
cd open-pi-zero
uv sync --extra dev
uv run python scripts/run_sample.py --batch_sizes 1 2 4 --num_trials 10
```

Run the lightweight regression tests with `uv run pytest`.

The script automatically uses CUDA, Apple Silicon MPS, or CPU when available.
Use `--device cpu`, `--device mps`, or `--device cuda` to select one explicitly.
Results are written to `open-pi-zero/batch_invariance_results.json`.

The sample uses randomly initialized model weights and synthetic inputs. It is
intended to measure whether the output for a fixed sample changes when other
samples are added to its batch.

For the CUDA/Triton kernels, use a CUDA-capable environment. CPU and MPS are
useful for checking model wiring and the per-sample convolution, but the
operator shim becomes a no-op there. The experiment is not a performance or
numerical-equivalence benchmark: it measures sensitivity of sample zero to
batch composition under a fixed seed and fixed initial action noise.

See [`open-pi-zero/docs/batch-invariance.md`](open-pi-zero/docs/batch-invariance.md)
for the experiment protocol, interpretation, and known limitations.

## Reproduce the pretrained campaign

The full campaign has stricter checkpoint, environment, diagnostic/freeze,
held-out, paired-behavior, and artifact gates. The operator dependency is
already present at the repository root in the path expected by
`open-pi-zero/pyproject.toml`. See
[`open-pi-zero/experiments/README.md`](open-pi-zero/experiments/README.md) for
the exact sequence and external asset requirements. The generated answers to
the experiment's A–H research questions are in
[`open-pi-zero/results/EXPERIMENT_REPORT.md`](open-pi-zero/results/EXPERIMENT_REPORT.md),
and `open-pi-zero/results/audit.json` is the machine-readable completion gate.

Raw JSONL is authoritative. Tables, figures, summaries, and the report are
generated from it; missing measurements must remain unavailable rather than be
estimated or interpolated.
