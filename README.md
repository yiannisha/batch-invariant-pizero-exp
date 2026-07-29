# Batch-invariant PiZero

This repository contains the batch-invariance experiment for PiZero, along with
the model implementations used by the experiment.

## Run the sample

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
the sample from the `open-pi-zero` project directory:

```console
cd open-pi-zero
uv sync
uv run python scripts/run_sample.py --batch_sizes 1 2 4 --num_trials 10
```

The script automatically uses CUDA, Apple Silicon MPS, or CPU when available.
Use `--device cpu`, `--device mps`, or `--device cuda` to select one explicitly.
Results are written to `open-pi-zero/batch_invariance_results.json`.

The sample uses randomly initialized model weights and synthetic inputs. It is
intended to measure whether the output for a fixed sample changes when other
samples are added to its batch.
