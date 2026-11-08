# Exact-start paired SIMPLER campaign

The campaign finished on 2026-11-08 at 02:51 UTC. It used the frozen
Fractal-Beta checkpoint, four SIMPLER tasks, and 50 paired initializations per
task. Each initialization ran native singleton, native dynamic batching,
patched singleton, and patched dynamic batching twice, for 1,600 episodes.
Dynamic batches cycled through sizes 1, 2, 4, and 8 at successive policy calls.

## Success rates

| Task | Native singleton | Native dynamic | Patched singleton | Patched dynamic |
| --- | ---: | ---: | ---: | ---: |
| Pick Can | 49/50 (98%) | 48/50 (96%) | 49/50 (98%) | 49/50 (98%) |
| Move Near | 43/50 (86%) | 38/50 (76%) | 42/50 (84%) | 42/50 (84%) |
| Open Drawer | 23/50 (46%) | 23/50 (46%) | 27/50 (54%) | 27/50 (54%) |
| Close Drawer | 38/50 (76%) | 34/50 (68%) | 38/50 (76%) | 38/50 (76%) |
| Equal-task pooled total | 153/200 (76.5%) | 143/200 (71.5%) | 156/200 (78%) | 156/200 (78%) |

| Task | Native dynamic minus singleton | 95% paired bootstrap interval | Native singleton-only / dynamic-only successes |
| --- | ---: | ---: | ---: |
| Pick Can | -2 percentage points | [-6, 0] | 1 / 0 |
| Move Near | -10 percentage points | [-20, -2] | 5 / 0 |
| Open Drawer | 0 percentage points | [-14, 14] | 7 / 7 |
| Close Drawer | -8 percentage points | [-18, 0] | 5 / 1 |

Intervals use the analysis script's 10,000-resample paired percentile bootstrap.
They describe these task-specific samples; they are not adjusted for multiple
comparisons. The equal-task pooled totals are descriptive, not a substitute
for the task-specific paired comparisons.

## Replay and attribution checks

- All 200 blocks passed exact restoration checks and all four A/A duplicate
  episode checks.
- Native singleton and dynamic actions diverged in all 200 blocks. In every
  block, the first different action was computed at batch size greater than
  one from a byte-identical target policy input and an identical recorded
  trajectory up to that action.
- Native singleton and dynamic success differed in 26 blocks: 18
  singleton-only successes and 8 dynamic-only successes. All 26 disagreements
  passed the first-action attribution check.
- Patched singleton and dynamic actions, recorded trajectories, and outcomes
  matched exactly in all 200 blocks. Their paired success-rate difference was
  zero in every task.
- Patched dynamic had 156 successes, versus 143 for native dynamic and 153 for
  native singleton. The patch removes the batch-dependent mismatch in this
  campaign; it does not reproduce native singleton behavior on every start.

Open Drawer illustrates why paired outcomes matter: the two native conditions
both succeeded 23 times, but they disagreed on 14 individual starts.

The simulator does not expose every internal physics-engine variable. The
protocol rebuilds the physics scene before each restore and requires exact
observable state, initial input, and duplicate-episode reproducibility. The
claims above apply to the recorded hardware, software, model, and task setup.

## Audit files

- [summary.json](summary.json) contains per-task paired estimates and the
  first native action difference for every block.
- Each task directory contains `episodes.jsonl.gz` and `runtime.json`.
  The summary records the SHA-256 of each *uncompressed* episode file.
- [status.json](status.json) records `complete_verified`, and
  [replay_manifest_snapshot.json](replay_manifest_snapshot.json) identifies the
  companion observations used by dynamic batches.
