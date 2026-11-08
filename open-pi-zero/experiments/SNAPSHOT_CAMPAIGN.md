# Exact-start paired SIMPLER campaign

This follow-up tests the same Fractal-Beta checkpoint, tokenizer, four SIMPLER
tasks, 50 initialization seeds per task, and singleton/dynamic batch cycle as
the original campaign. The target policy input uses the same indexed noise
sequence. Dynamic batches cycle through sizes 1, 2, 4, and 8 and draw companion
requests from a retained SIMPLER replay manifest.

The original evaluator called `reset` once per condition. Object settling could
then produce slightly different starting poses. `simpler_snapshot_eval.py`
instead resets and settles once per initialization, captures a canonical scene
snapshot, and restores it before every condition. SAPIEN changes a few
quaternion bits on unpack, so the canonical state is defined **after** one
unpack. Every condition restores that same canonical state.

Each restore rebuilds the PhysX scene before loading the snapshot. This clears
contact caches that `Scene.pack()` does not serialize. Without rebuilding,
Move Near produced different object trajectories under identical starting
state and identical actions. The snapshot worker then checks exact equality of
simulation state, agent/controller state, task counters, elapsed steps, Python
and NumPy random-generator states, rendered observation, and instruction. The
evaluator also checks the full first policy input, including action noise.

Four conditions are run twice per initialization: native singleton, native
dynamic, patched singleton, and patched dynamic. A block is accepted only if
each duplicate exactly reproduces its action sequence, trajectory, success,
termination reason, and policy-call count. The analyzer requires all 200 blocks
to pass, and for any native success disagreement it requires the first differing
action to be computed from byte-identical target inputs with batch size greater
than one and identical recorded states before that action. Patched singleton
and dynamic must have identical actions, trajectories, and outcomes.

Run `prepare_replay.py` as documented in `README.md`, writing a separate
manifest and tensor directory. Then run one `simpler_snapshot_eval.py` process
per task with a distinct output directory and GPU. Each task output has a
`runtime.json` that binds checkpoint, replay manifest, tokenizer, source hashes,
environment revisions, and snapshot protocol. After all four task outputs are
complete, run:

```console
python experiments/analyze_snapshot_simpler.py \
  --episodes /path/to/results/{pick_can,move_near,open_drawer,close_drawer}/episodes.jsonl \
  --initializations 50 --output /path/to/results/summary.json
```

`--resume` skips only complete eight-condition blocks. A partial or failed A/A
block stops the campaign for investigation; it is never silently combined with
a newly generated snapshot. The results are valid for the recorded hardware,
software, checkpoint, and scene configuration.

## Verified video replays

`render_snapshot_comparison.py` replays the recorded environment actions from
one qualified block. It requires the original snapshot digests, every recorded
trajectory state and reward, and the final outcome to match. It also requires
the patched singleton and dynamic RGB frames to be identical before encoding.
The videos show the original experiment trajectories, not a new policy run.

- [Open Drawer start 3: native singleton vs dynamic](../../artifacts/open_drawer_start_03/open_drawer_start_03_native_singleton_vs_batched.mp4)
- [Open Drawer start 3: patched singleton vs dynamic](../../artifacts/open_drawer_start_03/open_drawer_start_03_patched_singleton_vs_batched.mp4)
- [Close Drawer start 5: native singleton vs dynamic](../../artifacts/close_drawer_start_05/close_drawer_start_05_native_singleton_vs_batched.mp4)
- [Close Drawer start 5: patched singleton vs dynamic](../../artifacts/close_drawer_start_05/close_drawer_start_05_patched_singleton_vs_batched.mp4)
