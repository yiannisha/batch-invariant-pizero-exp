# Experiment plan and execution log

Status: **completed and independently audited** on 2026-09-29. The final
machine-readable gate is `open-pi-zero/results/audit.json`; the generated
paper-facing report is `open-pi-zero/results/EXPERIMENT_REPORT.md`.

## Original objective

Complete the experimental campaign for the batch-invariant VLA paper. Core
campaigns began on NVIDIA H100; the native-Vulkan SIMPLER campaign and final
held-out work completed on NVIDIA RTX PRO 6000 Blackwell.

Relevant repositories:

- Main VLA repository:
  https://github.com/yiannisha/batch-invariant-pizero

- Batch-invariant operator library:
  https://github.com/thinking-machines-lab/batch_invariant_ops

- Conv2d/BMM additions:
  https://github.com/thinking-machines-lab/batch_invariant_ops/pull/27

The required batch-invariant kernels, including Conv2d and BMM, have already been implemented and tested.

Do **not** redesign the kernels unless integration with the complete pretrained policy exposes a real correctness or coverage bug.

The primary work is now experimental:

1. integrate the batch-invariant operators into the complete pretrained π0-style policy;
2. build a reproducible experiment harness;
3. establish the native policy's numerical dependence on batching;
4. establish whether the complete patched path achieves exact request-level batch invariance;
5. characterize where differences arise and how they propagate through iterative action generation;
6. run the diagnostic and held-out pretrained-policy campaigns;
7. run the paired SIMPLER closed-loop campaign;
8. benchmark operator, policy, and serving performance on the H100;
9. generate the raw records, tables, plots, and experiment summary required for the paper.

Do not stop after writing scripts. Run the experiments.

---

# 1. Scientific rules

This is an empirical campaign, not an attempt to obtain a preferred result.

Do not modify experimental conditions merely because a native result is smaller or larger than expected.

A numerical batch-invariance violation means:

> the same request produces any non-bit-identical output when only its batching arrangement changes.

Task-success disagreement is a separate behavioral metric.

Therefore all of the following are valid possible outcomes:

- native numerical differences + behavioral differences;
- native numerical differences + no behavioral differences;
- native differences only in some batch configurations;
- native operators that happen to produce equal output for particular shapes/configurations.

Do not search seeds, companions, batches, or observations merely to maximize native failures.

Run the predefined campaign and retain all outcomes.

Do not discard zero or negative results.

---

# 2. Inspect and understand the existing implementation

Before changing code, inspect both repositories and understand:

- current branch and commit;
- repository layout;
- inference entry point;
- model/checkpoint loading;
- image and language preprocessing;
- proprioceptive/state representation;
- action normalization and denormalization;
- flow-matching/action-generation loop;
- conditioning-cache behavior;
- current batch-invariant integration;
- existing tracing;
- diagnostic scripts;
- SIMPLER integration;
- tests and dependency management.

Inspect PR #27.

Confirm how the invariant implementation registers and reaches the relevant dispatched operators, including at least:

- `aten::mm`
- `aten::addmm`
- `aten::bmm`
- `aten::convolution`
- relevant reduction/log-softmax paths

Confirm the intended use of:

`set_batch_invariant_mode(...)`

Do not duplicate these operator implementations unnecessarily.

---

# 3. Record the exact experimental environment

Create a machine-readable environment record, e.g.

`results/environment.json`

Record:

- git SHA of `batch-invariant-pizero`
- git SHA/revision of `batch_invariant_ops`
- exact PR #27 revision used
- checkpoint identifier/hash
- hostname
- GPU model
- GPU memory
- NVIDIA driver
- CUDA runtime/toolkit
- cuDNN
- cuBLAS
- PyTorch
- Triton
- Python
- SIMPLER version
- dtype
- TF32 settings
- PyTorch matmul precision
- deterministic-algorithm state
- cuDNN benchmark setting
- compilation/eager mode
- relevant environment variables
- available clock/power information

Generate this automatically rather than recording it manually.

---

# 4. Establish complete pretrained-policy execution

Use the public **Fractal-Beta open-pi-zero checkpoint with its matching inference configuration**.

The experimental policy must be the actual pretrained model, not a random or synthetic substitute.

Verify and preserve:

- image preprocessing;
- language preprocessing;
- proprioceptive representation;
- normalization;
- action normalization;
- action denormalization;
- time embeddings;
- rotary settings;
- conditioning-cache behavior;
- solver/update rule;
- number of flow steps.

Verify from the actual implementation that the policy:

- predicts four seven-dimensional actions;
- uses ten flow steps;
- executes the first two predicted actions in the evaluated controller.

Do not blindly assume these properties if the repository/checkpoint disagrees; document any discrepancy.

First reproduce correct singleton execution before running batching experiments.

---

# 5. Define the experimental execution configurations

Expose explicit modes corresponding to:

1. `native`
2. `native_deterministic`
3. `existing_invariant_ops`
4. `invariant_plus_patch_projection`
5. `explicit_per_matrix_attention`
6. `full_invariant`

Map these conceptual modes onto the implementation that actually exists.

The experiment records must make the configuration unambiguous.

For deterministic native execution use the applicable controls, including:

- PyTorch deterministic algorithms;
- disabling convolution benchmarking where appropriate;
- applicable cuBLAS deterministic configuration.

---

# 6. Verify actual dispatch and invariant-operator activation

Do not infer coverage from Python function names.

Use profiling / dispatcher inspection to establish which ATen operators actually execute.

At minimum establish:

- rank-2 matrix multiplication reaches `aten::mm` where expected;
- higher-rank attention multiplication reaches `aten::bmm`;
- SigLIP visual projection reaches `aten::convolution`;
- invariant mode actually replaces the intended dispatched operations.

An experiment configuration must fail loudly if the expected invariant override is silently bypassed.

Create an operator-coverage report containing:

- source-level operation;
- dispatched ATen operation;
- tensor rank;
- shape;
- stride;
- dtype;
- invariant/native path selected.

---

# 7. Invocation-aware numerical tracing

Create a reusable tracer for comparing one fixed request between singleton and batched execution.

Trace keys must distinguish repeated invocations using information such as:

- module;
- layer;
- flow step;
- invocation index.

Do not overwrite repeated tensors.

Capture at least:

## Vision
- patch projection output;
- relevant vision-transformer outputs;
- final visual representation.

## Conditioning
- conditioning representations;
- conditioning-cache outputs.

## Attention
- Q/K/V where useful;
- `QKᵀ`;
- attention softmax;
- `PV`.

## Flow/action generation

For each of the ten flow steps capture:

- action state `a_k`;
- predicted velocity `v_k`;
- updated action state `a_{k+1}`.

## Final policy output

Capture:

- full pre-clipping predicted action chunk;
- returned/post-clipping action;
- denormalized action;
- executed first-two-action prefix.

For every tensor comparison compute:

- exact bitwise equality;
- unequal-element count/fraction;
- maximum absolute error;
- mean absolute error;
- RMSE;
- relative L2 error;
- nonfinite count.

Determine the **first divergent tensor/operator** for every diagnostic comparison.

Tracing must be disabled for performance benchmarks.

---

# 8. Local replay of divergence boundaries

When a tensor first diverges:

1. capture the exact singleton inputs to the producing operation;
2. replay that operation independently;
3. vary only its batching arrangement;
4. determine whether that operator itself is batch-sensitive;
5. distinguish local numerical divergence from propagated upstream error.

Store:

- operator;
- module/layer;
- flow step;
- invocation;
- shapes;
- tensor rank;
- dtype;
- strides;
- contiguity;
- dispatched ATen operation;
- exact and approximate error statistics.

This diagnostic is particularly important for `aten::mm` versus `aten::bmm`.

---

# 9. H100 operator qualification

The kernels are already implemented and tested. Treat this as qualification of the exact experimental environment, not a new kernel-development project.

First run the existing Conv2d/BMM diagnostic from PR #27.

Then test the **actual VLA shapes** observed during pretrained-policy inference.

At minimum include:

### SigLIP patch projection

Use the real model's patch-projection shape.

### Attention products

Test both:

- `QKᵀ`
- `PV`

### Action-query diagnostic shape

Explicitly include the paper's identified product:

`[BH, 4, 281] × [BH, 281, 256]`

and any other actual shapes observed during the policy run.

### Batch sizes

Where memory permits:

`B = {1, 2, 4, 8, 16, 31, 32, 33, 64}`

### Number of inputs

Run **100 seeded inputs per operator configuration**.

Use predefined seeds and retain them.

### Additional layouts

Exercise:

- transposed operands;
- partial tiles;
- non-contiguous operands;
- actual policy strides.

For every configuration independently measure:

## Batch invariance

Compare a fixed sample/matrix evaluated alone against the corresponding sample/matrix in larger batches.

## Fidelity

Compare patched results with the corresponding native implementation.

These are different properties.

For selected cases also compare with FP64 references.

---

# 10. Precision campaign

Use eager FP32 as the primary configuration.

Evaluate BF16 separately.

FP16 may also be evaluated at the operator level where appropriate.

For FP32 explicitly control and record TF32 behavior.

Do not rely silently on framework defaults.

Treat each precision/execution configuration as separately qualified.

---

# 11. Repeat the native Conv2d batch sweep

Run the native SigLIP-shaped convolution with the real model shape across the predefined batch-size sweep.

Measure the same sample against singleton execution.

Report:

- batch size;
- exact equality;
- maximum absolute difference;
- failure count over seeds;
- dtype;
- TF32 state;
- relevant library/environment metadata.

Then run the registered invariant Conv2d path over the same campaign.

Do not assume that the previous observed `B=32` transition must reproduce identically on the H100.

Record the actual H100 behavior.

---

# 12. Build the frozen SIMPLER replay dataset

Use four SIMPLER tasks:

1. picking a can;
2. moving near an object;
3. opening a drawer;
4. closing a drawer.

Collect:

- 25 episodes per task;
- 100 total episodes;
- 10 observations per episode;
- 1,000 total observations.

Include observations from unsuccessful episodes.

Freeze episode membership.

Use:

- 5 episodes/task as diagnostic;
- 20 episodes/task as held out.

This produces:

- 200 diagnostic observations;
- 800 held-out observations.

For every observation create **three explicit stored initial action-noise tensors**.

Do not rely solely on reseeding a global RNG.

Each request should have a stable identifier containing at least:

- task;
- episode;
- observation/time index;
- noise index.

Create a machine-readable frozen dataset manifest.

---

# 13. Diagnostic policy-level campaign

Use only the **200-observation diagnostic set** while debugging coverage.

Run the batching transformations and inspect the first divergence.

After known Conv2d/BMM coverage is active, audit the complete execution graph for any remaining batch-sensitive reduction/path.

Specifically investigate if encountered:

- normalizations;
- ordinary attention softmax;
- reductions;
- cache updates;
- framework substitutions.

For each new candidate:

1. locate the first divergence;
2. perform local replay with identical inputs;
3. establish whether it is genuinely batch-sensitive;
4. repair it only if necessary;
5. add a focused regression test.

Repeat until the intended full-invariant path produces exact request outputs over the diagnostic transformation matrix.

---

# 14. Freeze the numerical implementation

Once diagnostic coverage is complete:

1. commit the experimental/integration implementation;
2. record the exact SHA;
3. mark the numerical implementation frozen.

Do not change numerical implementation based on held-out results.

If the held-out campaign uncovers an actual implementation bug:

- document it;
- invalidate the previous held-out campaign;
- fix it;
- freeze a new revision;
- rerun the full held-out campaign.

Never selectively keep favorable results from different revisions.

---

# 15. Full batching-transformation matrix

For a fixed target request vary only batching arrangement.

Test:

## Batch size

Required:

`B = {1, 2, 4, 8}`

Also test:

`B = {16, 32}`

where memory permits.

## Target position

Where meaningful:

- first;
- middle;
- last.

## Companions

Use:

- duplicated companions;
- diverse/unrelated observations.

## Permutation

Execute the same request set under different orderings.

Restore canonical request ordering before comparisons.

## Partitioning

Execute the same logical request set under different batch partitions.

Again compare the same request after restoring output ordering.

For every comparison preserve:

- image/observation;
- language;
- proprioceptive state;
- all request-local state;
- stored initial action noise.

Only batching arrangement may vary.

---

# 16. Causal policy-level ablation

Run the complete ablation:

1. Native
2. Native + deterministic settings
3. Existing invariant operators
4. Operators + patch projection
5. Explicit per-matrix attention reference
6. Persistent BMM + complete audited invariant path

For each configuration record:

- request-level numerical violation rate;
- arrangement-level violation rate;
- maximum action error;
- first divergent boundary;
- pre-clipping action error;
- returned-action error;
- per-flow-step error.

This experiment must show how each intervention moves or eliminates the first divergent execution boundary.

---

# 17. Held-out numerical campaign

After freezing the implementation, evaluate all:

**800 held-out observations × 3 stored noise tensors**

giving:

**2,400 fixed request/noise pairs**

before batching transformations.

For each fixed request/noise pair run the complete applicable batching-transformation matrix.

A request-level violation occurs if **any tested batching arrangement** produces output not bit-identical to that implementation's singleton reference.

Report:

- exact violation count;
- request-level violation percentage;
- arrangement-level violation percentage where useful;
- maximum absolute action error;
- median error;
- p95/p99 error where meaningful;
- first-divergence distribution;
- pre-clipping differences;
- returned-action differences;
- differences across all ten flow steps.

After action denormalization separately report applicable:

- translation discrepancies;
- rotation discrepancies;
- gripper discrepancies.

Report both:

- full predicted action chunk;
- executed first-two-action prefix.

---

# 18. Patched-vs-native singleton fidelity

Separately compare:

`full_invariant singleton`

against:

`native singleton`

for the same requests/noise.

This measures the numerical change introduced by the repair itself.

Do not confuse this with batch invariance.

Report appropriate:

- exact equality;
- max absolute error;
- RMSE;
- relative L2;
- denormalized action differences.

Use selected FP64 references where appropriate.

---

# 19. Flow-step numerical propagation

For every relevant request analyze the ten iterative action-generation steps.

For every step calculate:

- exact-equality rate;
- median error;
- an explicitly labeled tail statistic;
- maximum error.

Generate plot-ready data for:

- native;
- existing invariant library;
- patch-projection/intermediate repair;
- complete audited invariant path.

Determine whether batching-induced numerical differences:

- amplify;
- remain stable;
- shrink;
- disappear;
- first arise only at later flow steps.

Never replace exact zero with invented epsilon solely for plotting purposes.

---

# 20. Paired closed-loop SIMPLER campaign

Run **50 matched initializations per task**.

Tasks:

- Pick can
- Move near
- Open drawer
- Close drawer

Conditions:

1. native singleton
2. native dynamic batching
3. patched singleton
4. patched dynamic batching

Total target:

`4 × 50 × 4 = 800 episodes`

Every matched episode must use:

- the same initial scene;
- the same indexed policy-noise sequence.

Dynamic batching cycles through:

`B = {1, 2, 4, 8}`

Companion requests should come from the frozen replay set.

Simulator stepping must be independent of inference wall-clock time so numerical effects are isolated from controller-delay effects.

For every episode store:

- task;
- initialization ID;
- execution condition;
- success/failure;
- termination reason;
- policy-call count;
- action sequence;
- relevant simulator trajectory/state.

---

# 21. Closed-loop behavioral measurements

Report:

- per-task success;
- paired success disagreements;
- trajectory differences;
- patched-vs-native singleton success difference;
- episode-level bootstrap 95% confidence intervals.

A native numerical violation does **not** require a task-success disagreement.

Zero task disagreements must be reported honestly if that is the result.

---

# 22. Trajectory-level analysis

Because binary task success may be insensitive to small continuous-action perturbations, characterize paired trajectory divergence where simulator state allows it.

Possible metrics include:

- end-effector positional difference;
- end-effector rotational difference;
- gripper-state difference;
- object-pose difference;
- maximum trajectory deviation;
- terminal-state deviation.

Only use metrics that can be extracted reliably from SIMPLER.

Define exact units and formulas.

Do not invent unavailable state.

---

# 23. Kernel performance benchmarks

Benchmark the actual policy-relevant shapes.

For attention compare at minimum:

1. native BMM
2. explicit per-matrix invariant MM reference
3. persistent invariant BMM

The most important invariant-performance comparison is:

`persistent invariant BMM`

versus

`explicit invariant per-matrix MM`

because both preserve the intended numerical schedule.

Measure:

- median latency;
- p95 latency;
- kernel-launch count where obtainable;
- throughput where meaningful.

Exclude tracing, compilation, and initialization.

Use correct CUDA synchronization.

Retain raw timing samples.

---

# 24. Complete-policy latency benchmark

Compare:

1. native dynamic batching
2. full invariant dynamic batching
3. serial singleton execution
4. fixed-size padded batching

For fixed-size padding:

- keep batch shape constant;
- keep request token shapes stable;
- exclude padded outputs;
- test target-position sensitivity;
- use its own canonical fixed-shape execution as its numerical reference.

For every configuration run:

- 5 independent sessions;
- 20 warm-up calls/session;
- 200 timed calls/session.

Synchronize at the action-output boundary.

Report:

- median latency;
- p95 latency;
- requests/second;
- peak allocated memory;
- kernel-launch count.

Timing must be trace-free.

---

# 25. Request-queue / serving benchmark

Create a reproducible request-queue benchmark comparing:

- native dynamic batching;
- invariant dynamic batching;
- serial singleton;
- fixed-size padded batching.

Sweep offered load over a range covering:

- underloaded;
- moderate occupancy;
- near saturation;
- saturation.

For every request record:

- arrival time;
- batch assignment;
- batch occupancy;
- service start;
- completion time;
- queueing latency;
- service latency;
- arrival-to-completion latency.

Report:

- throughput;
- median latency;
- p95 arrival-to-completion latency;
- batch occupancy distribution.

Verify numerical outputs under the same recorded schedules.

Mark configurations that fail the numerical contract.

---

# 26. Raw data structure

Use a structure approximately like:

```text
experiments/
    record_environment.py
    prepare_replay.py
    trace_policy.py
    local_operator_replay.py
    operator_campaign.py
    policy_ablation.py
    heldout_invariance.py
    simpler_eval.py
    benchmark_kernels.py
    benchmark_policy.py
    benchmark_serving.py
    generate_tables.py
    generate_figures.py

results/
    environment.json
    operator/
    diagnostic/
    heldout/
    simpler/
    performance/
    serving/
    summaries/

artifacts/
    tables/
    figures/
    logs/
```

Adapt to the repository's existing organization instead of restructuring unnecessarily.

Store measurements in machine-readable formats such as:

- JSONL
- JSON
- CSV
- Parquet

Publication tables must never be the only copy of the measurements.

---

# 27. Required metadata per experiment record

Where applicable include:

- experiment ID;
- timestamp;
- git SHA;
- checkpoint ID/hash;
- GPU/configuration;
- request ID;
- task;
- episode ID;
- observation index;
- noise ID;
- implementation;
- dtype;
- TF32 state;
- execution mode;
- batch size;
- target batch position;
- companion IDs;
- request ordering;
- partition ID;
- exact-equality result;
- unequal-element count;
- max absolute error;
- RMSE;
- relative L2;
- first divergent tensor/operator;
- latency measurements.

---

# 28. Generate paper-ready tables

Programmatically generate at least:

## Operator / visual projection table

Include:

- implementation;
- batch size;
- max singleton-vs-batched difference;
- failure count;
- precision configuration.

## Policy-level ablation table

Rows:

- Native
- Native + deterministic settings
- Existing invariant operators
- Operators + patch projection
- Explicit per-matrix reference
- Persistent BMM + audited path

Columns at minimum:

- Violations (%)
- Exact violation count
- Max action error

## SIMPLER table

Rows:

- Pick can
- Move near
- Open drawer
- Close drawer

Report:

- native dynamic-batching success
- patched dynamic-batching success
- native paired disagreements
- patched paired disagreements

Generate all table entries directly from retained measurements.

---

# 29. Generate paper-ready figures

At minimum produce:

## Flow-step propagation

X axis:

- flow step 0–10

Curves:

- native;
- existing invariant operators;
- patch-projection/intermediate repair;
- complete invariant path.

Plot an aggregate such as median plus a clearly labeled tail statistic.

Clearly distinguish exact zero from positive error.

## Serving tradeoff

Plot:

- throughput vs p95 arrival-to-completion latency

for:

- native dynamic batching;
- invariant dynamic batching;
- serial singleton;
- fixed-size padding.

Annotate whether each configuration satisfies the numerical contract.

All plots must be generated directly from retained data.

---

# 30. Statistical treatment

For closed-loop outcomes use paired analysis because initial conditions/noise are matched.

Compute episode-level bootstrap 95% confidence intervals.

Record:

- bootstrap procedure;
- random seed;
- number of bootstrap resamples.

Do not overinterpret small success differences.

For numerical batch invariance, use exact equality directly rather than statistical hypothesis testing.

Error distributions should be descriptive.

---

# 31. Performance safeguards

Performance experiments must:

- disable tracing;
- exclude compilation;
- exclude first-call initialization;
- synchronize CUDA correctly;
- avoid profiler overhead in timed regions;
- avoid dumping tensors in the timing path.

Numerical qualification and timing should be separate experimental runs.

The implementation being timed must otherwise match the qualified numerical implementation.

---

# 32. Final experimental report

Create:

`results/EXPERIMENT_REPORT.md`

It must answer:

## A. Does native inference violate request-level batch invariance?

Report:

- exact violating request count;
- percentage;
- affected batching transformations;
- magnitude distribution.

## B. Where does native execution first diverge?

Report frequency of first divergence at:

- visual patch projection;
- attention/BMM;
- any other discovered boundary.

## C. How do differences propagate through the flow solver?

Describe measured evolution across all ten flow steps.

## D. Does the full invariant path achieve exact action equality?

Report exact failure count.

If nonzero, classify every remaining failure by its first divergence.

## E. How faithful is invariant singleton inference to native singleton inference?

Report separately from batch invariance.

## F. Do numerical differences affect behavior?

Report separately:

- action divergence;
- trajectory divergence;
- paired success disagreement;
- success-rate differences.

Do not treat zero behavioral disagreements as a failed experiment.

## G. What is the computational cost?

Report:

- persistent invariant BMM vs explicit invariant MM loop;
- complete-policy latency difference;
- throughput difference;
- peak-memory difference;
- launch-count difference.

## H. How does invariant batching compare with practical alternatives?

Compare:

- invariant dynamic batching;
- serial singleton execution;
- fixed-size padding.

---

# 33. Acceptance criteria

The campaign is complete only when:

- the complete Fractal-Beta policy runs;
- actual ATen dispatch/override coverage has been verified;
- environment metadata is frozen;
- the operator qualification campaign uses the real VLA shapes;
- 100 seeded operator inputs/configuration have been evaluated;
- the 1,000-observation replay dataset has been created;
- the diagnostic split has been used to close any remaining execution-path gaps;
- the numerical implementation has been frozen before held-out evaluation;
- all 800 held-out observations × 3 noise tensors have been evaluated under the required batching transformations;
- causal policy-level ablations are complete;
- flow-step propagation measurements are complete;
- all 800 paired SIMPLER episodes are attempted/completed, or a genuine simulator blocker is precisely documented;
- H100 kernel benchmarks are complete;
- H100 complete-policy benchmarks are complete;
- serving/load measurements are complete;
- raw measurements are retained;
- tables and figures are generated automatically from raw records.

---

# 34. Execution behavior

Work autonomously.

Do not ask for permission after every implementation step.

Inspect the existing architecture and make the smallest reasonable additions.

Checkpoint experiment infrastructure in git.

If one experimental branch is blocked by an external dependency, continue all independent experiments.

Do not claim that an experiment ran unless it actually ran.

Never manufacture, interpolate, or estimate missing measurements.

Do not replace unexpected results with expected values.

The final repository state should make the full H100 experimental campaign reproducible with a small number of documented commands.

---

# 35. RTX PRO 6000 continuation log (2026-09-28)

The unfinished SIMPLER/replay work was moved to an RTX PRO 6000 Blackwell
Server Edition pod because the original H100 pod did not expose NVIDIA
graphics/Vulkan capabilities.  Completed H100 core campaigns remain
authoritative and must not be rerun.  No partial replay dataset was
transferred; the RTX replay will be generated from scratch.

## Transfer verification

- Main repository: `1fe465be27a773e14b5f37f4c0a8c0a553b7ce53` on `main`.
- Frozen numerical policy revision retained in
  `results/numerical_freeze.json`:
  `a2576bbac02fca0ba1868843475f4a644835bdf1`.
- Operator baseline: `14dafb5fd84350dc796a96cbecb566ca6295593f`;
  PR #27 base: `679da28bd038c8e3d7c161b0486cac9c54c4f967`.
- SIMPLER: `59ad9e1539042ed333fd8ebba1b0395f5662f0bd`;
  ManiSkill2_real2sim: `91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c`.
- Fractal-Beta checkpoint: 11,774,183,772 bytes, SHA256
  `1da40985d963ddf394fa63e7db5ddd4cd8082d5c6ce0b2a0cae56abd8b6ec0eb`.
- Tokenizer: `/workspace/pizero-assets/paligemma-tokenizer`.
- There was no replay directory or replay manifest in the transferred assets.

## Native Vulkan diagnosis and repair

The NVIDIA 595.91.07 user-space driver and valid ICD were already mounted,
but the CUDA base image omitted the generic GLVND `libEGL.so.1` loader.  This
made the ICD handshake return `VK_ERROR_INITIALIZATION_FAILED`.  A matching
Ubuntu 24.04 `libegl1` loader is retained under `.runtime/lib`, with SHA256
`875ecbb2a07d60e32216c9f102965abc1e9cc1da7789558e2e9b8b9c107e231d`.
Sourcing `.runtime/env.sh` selects `/etc/vulkan/icd.d/nvidia_icd.json` and
disables irrelevant implicit display layers.  Verified native device:

- Vulkan API: 1.4.329;
- driver: NVIDIA 595.91.07;
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition;
- device UUID: `63cbdbc3-ef09-5abb-8ec0-27c3d63f0b16`.

The stale `/usr/share/vulkan/icd.d/nvidia_icd.json` points at nonexistent
`libnvidia-vulkan.so.1` and is deliberately not used.

## SIMPLER smoke evidence

SAPIEN 2.2.2 runs in the isolated Python 3.10 simulator environment at
`/workspace/pizero-assets/simpler-venv`; the policy uses the recorded H100
software family (Python 3.12.3, PyTorch 2.8.0+cu128, Triton 3.4.0,
Transformers 4.57.1) at `/workspace/pizero-assets/policy-venv`.

Native Vulkan reset/render/step passed through the production cross-Python
worker for all four predefined environments:

- `google_robot_pick_horizontal_coke_can`: `pick coke can`;
- `google_robot_move_near_v0`: `move blue plastic bottle near pepsi can`;
- `google_robot_open_drawer`: `open bottom drawer`;
- `google_robot_close_drawer`: `close top drawer`.

Each returned a 512x640 uint8 overhead RGB image and completed a zero-action
step.  The headless GLFW warning is expected; Vulkan rendering is active.

The production `prepare_replay.py` path also completed a four-task end-to-end
smoke (one episode and one retained observation per task).  It produced 4
episodes, 4 serialized observations, and 12 deterministic request/noise pairs.
Policy-call counts were 40 (pick-can), 40 (move-near), 57 (open-drawer), and
57 (close-drawer); pick-can and close-drawer succeeded.  This smoke is stored
separately at `/workspace/pizero-assets/replay-smoke-rtx` and is not an input
to the full campaign.

## Blackwell launch-only operator adaptation

The original three-stage persistent Triton launch requires 133,120 bytes of
shared memory for the FP32 tile, while this GPU exposes 101,376 opt-in bytes
per block.  Operator commit
`8fdbcada1401608622a07e1d35b645973deb905d` selects two pipeline stages only
when the device cannot fit the H100 three-stage launch.  Tile sizes, reduction
order, IEEE input precision, and kernel arithmetic are unchanged.  The H100
baseline/freeze remains recorded separately rather than rewriting completed
campaign provenance.

Post-adaptation real-policy smoke evidence:

- checkpoint load: 13.24 GiB allocated VRAM;
- native singleton inference: finite `(1, 4, 7)` output;
- full-invariant singleton inference: finite `(1, 4, 7)` output with 1,470
  `mm`, 204 `addmm`, 700 `bmm`, 1 convolution, and 440 mean overrides;
- singleton versus batch-2 target under native execution: 23 unequal output
  elements, maximum absolute error `1.043081283569336e-07`;
- the same comparison under the full-invariant path: bit-identical, zero
  unequal elements.

The RTX machine record is `results/rtx_environment.json`.  Next execution
order is: generate the complete 100-episode/1,000-observation replay from
scratch, run held-out invariance, run all paired closed-loop SIMPLER episodes
with resume enabled, then regenerate analysis/audit/report artifacts.

## Full replay completed

The clean RTX replay campaign completed from scratch and passed the script's
cardinality assertion:

- 100 episodes (25 per task);
- 1,000 serialized observations (10 per episode);
- 200 diagnostic and 800 held-out observations;
- 3 stored noise tensors per observation, yielding 3,000 request/noise pairs;
- all 1,000 manifest tensor paths exist;
- tensor payload directory: `/workspace/pizero-assets/replay-rtx-full` (119 MiB);
- manifest: `results/replay_manifest.json` (2,433,837 bytes), SHA256
  `7b2e3cee5635112b885862865a879d4e4a089efb292b10d50fdbf5e222f1dd08`.

Replay-policy success counts are retained as observations rather than used to
filter membership: pick-can 25/25, move-near 18/25, open-drawer 16/25, and
close-drawer 22/25 (81/100 total).  Total policy calls were 4,850.  These are
replay-generation outcomes, not the paired closed-loop comparison.

## RTX diagnostic replay qualification

Before the exhaustive held-out run, the real replay writer was exercised on
24 fixed diagnostic request/noise pairs spanning all four tasks (15 pick-can
pairs and 3 pairs from one observation in each other task).  Every pair used
the full applicable B={1,2,4,8} transformation matrix, yielding a rectangular
1,104-record dataset: 552 native and 552 full-invariant arrangements.

- full-invariant: 552/552 bit-identical, zero violations, maximum absolute
  error exactly 0;
- native: 48/552 bit-identical and 504/552 violating arrangements, maximum
  absolute action error `0.00043824315071105957`;
- 24 separate patched-vs-native singleton fidelity records were retained.

The result files are
`results/diagnostic/simpler_invariance.jsonl` and
`results/diagnostic/simpler_singleton_fidelity.jsonl`.  The diagnostic was
used for coverage qualification rather than estimating held-out rates; the
required exhaustive estimate remains the 2,400-pair held-out campaign.

## Exhaustive held-out campaign in progress

The required 800 held-out observations x 3 stored noises were launched with
both native and full-invariant implementations over the complete B={1,2,4,8}
transformation matrix.  The expected output is 110,400 arrangement records
plus 2,400 singleton-fidelity records.  It runs resumably in detached tmux
session `pizero-heldout` using `.runtime/run_heldout.sh` and writes:

- `results/heldout/invariance.jsonl`;
- `results/heldout/singleton_fidelity.jsonl`;
- `results/heldout/runner.log`.

The session was confirmed active after its first complete pair (78
arrangement records and 1 fidelity record had been flushed at the check).
Do not start the closed-loop campaign on the same GPU until this process has
finished; its output is append-safe and `--resume` is enabled.

`results/heldout/runtime.json` separately binds the live evaluation to harness
revision `e12fc62cc2afb8236dd7bf4176e385863e2286f9`, the frozen numerical
baseline, RTX launch-only operator adaptation
`8fdbcada1401608622a07e1d35b645973deb905d`, checkpoint hash, replay-manifest
hash, RTX environment-record hash, and expected record cardinalities.  This
keeps the launch-only portability revision explicit without rewriting the
completed H100 numerical freeze.

Detached tmux session `pizero-continuation` runs
`.runtime/continue_after_heldout.sh`.  It waits for `pizero-heldout` to exit,
requires exact held-out cardinalities (110,400 arrangements and 2,400
fidelity records), and only then starts the resumable 800-episode paired
SIMPLER campaign.  On exact 800-episode completion it runs the seeded 10,000
resample bootstrap analysis.  A failed or incomplete held-out run fails the
gate instead of starting closed loop.  Status is logged in
`results/continuation.log`.

Detached tmux session `pizero-finalize` waits for the paired continuation to
exit, then independently requires 110,400 held-out arrangements, 2,400
fidelity records, 800 closed-loop episodes, and a nonempty bootstrap summary.
Only after those gates pass does it regenerate tables, figures, the final
report, and the machine-readable audit.  The final consumers were updated for
the RTX campaign: the flow plot and JSON summary aggregate all ten held-out
flow steps with median/p95/maximum errors; the report uses real held-out
singleton fidelity and paired behavioral results; and the audit treats the
H100 Vulkan blocker as historical while requiring the complete RTX outputs.
Status is logged in `results/finalize.log`.

The final audit also structurally verifies the 1,104-record four-task RTX
diagnostic replay, all ten held-out flow-step summaries, required report
sections A-H, and exact generated-table row counts.  Held-out numerical
results and the six-mode H100 causal ablation are emitted as separate tables
(`policy_heldout` and `policy_ablation`) so completing replay evaluation does
not overwrite or hide the causal-ablation deliverable.

Before closed-loop launch, the SIMPLER evaluator was extended to write
`results/simpler/runtime.json` at actual process startup.  The record binds the
evaluator/support/worker/analysis source hashes, policy repository revision,
frozen numerical and operator baselines, RTX launch-only operator revision,
checkpoint/replay/environment hashes, Vulkan and interpreter paths, GPU
identity, the SIMPLER and ManiSkill2_real2sim revisions, seed, task/condition
matrix, and expected 800-episode cardinality.  Resume refuses to mix any
changed identity field.  The final audit now verifies that provenance plus all
800 unique task/condition/initialization cells, condition-to-implementation
and batching mappings, deterministic seeds, bootstrap metadata, and finite
trajectory metrics.  A synthetic 800-row full-cardinality fixture passed this
audit before launch.  At `2026-09-28T03:49:18Z`, the held-out process remained
healthy at 100% GPU utilization with 2,616/110,400 arrangement records and
56/2,400 fidelity records; both gated downstream sessions were still alive.

A schema-aware partial-integrity audit at 2,739 arrangement records found
2,739 unique arrangement keys, 119 complete request/implementation groups of
23 records, one actively written partial group, unique fidelity request IDs,
and 1,359/1,359 bit-exact full-invariant records.  The same complete-matrix
check (1 singleton, 16 batch-position, 3 permutation, and 3 partition records;
physical sizes distributed 2/6/8/7 across B=1/2/4/8) was added to the final
audit and passed the retained 1,104-record diagnostic dataset.  At the current
mean serialized record size the two held-out outputs project to about 1.43 GiB,
with 129 GiB free, so local storage is not a campaign risk.

The SIMPLER checkout reports its ManiSkill submodule dirty only because
`jasmine_tea_3.jpg.002.jpg` has mode 0644 instead of the repository's 0755.
Its 1,172,246 content bytes are identical to `HEAD` (SHA256
`82c10803d9aac79582f33a41236ff6a04195b63273160e8c126089f3ada5fc69`),
so this is a transferred permission-bit difference rather than a changed
rendering asset.  It is preserved rather than reverted while live work runs.

The durable experiment runbook was brought in sync with the executed RTX
campaign in policy commit `050553b1e512bce4b24bd4a8aeea821f103e0ba4`.
It now distinguishes the frozen H100 operator baseline from the Blackwell
launch-only revision, documents the native-ICD/GLVND requirement and retained
loader hash, gives the complete cross-Python replay/held-out/SIMPLER commands,
states exact cardinality gates, and identifies `results/simpler/runtime.json`
and `results/audit.json` as provenance/completion evidence.  The pod-local
loader binary remains outside git; the repository records how to obtain the
distribution-matched loader rather than committing a system shared library.

A figure/report audit found and closed a scope-loss bug that would have
replaced the required four-path flow figure with a two-path held-out figure.
Policy commit `b9c1d9d52039da637cd4a1327a9b19195783dc08` now generates two explicitly
scoped artifacts: `flow_step_propagation.png` contains native, existing
invariant operators, patch-projection repair, and complete invariant paths for
the frozen B=2 causal diagnostic (mean and maximum over action-state elements),
while `flow_step_propagation_heldout.png` contains exhaustive native/full
median and p95 curves.  Both retain true zeros.  The diagnostic summary is
machine-readable at `results/diagnostic/flow_step_summary.json`, is checked
against the raw trace, and the final audit requires both figures and summaries.
The final report now retains the intermediate causal paths alongside held-out
aggregate evidence.  Both plots were rendered and visually inspected; a
temporary-tree A-H report generation test also passed without altering the
retained report.

A second report-coverage pass closed remaining presentation gaps in commit
`a021159c5250fbfd463ebe706ac8234a1e5d7b9a`: category frequencies are now
reported for the retained first-divergence traces with their limited scope
stated; trajectory position, quaternion-geodesic rotation, and gripper formulas
and units are explicit; maximum and aligned-terminal metrics are emitted; all
six persistent-vs-explicit kernel comparisons include latency, throughput, and
launch deltas; and the complete-policy invariant-vs-native comparison derives
latency, throughput, peak-memory, and launch differences.  The final audit
requires these report elements.  A final-like report using the synthetic
800-episode/10,000-bootstrap fixture passed after the test caught and corrected
a loop-variable shadowing defect before retained output generation.

Closed-loop record metadata was completed before launch in commit
`29d8d36acd5c0444aca6222c469d827d1fd66d60`.  Every episode will now retain a
UUID experiment ID, stable request/episode ID, dtype, matmul/cuDNN TF32 states,
and eager execution mode.  Every executed action retains its full request
ordering in addition to target position, indexed noise ID, companion IDs,
normalized/environment action, and reward.  The final audit verifies these
fields, every dynamic B={1,2,4,8} cycle position, action finiteness and width,
policy-call coverage, and the action-count/trajectory-length relationship.
An enriched synthetic 800-episode campaign passed the real 10,000-resample
analysis and this stricter audit before the live watcher can invoke it.
Commit `d597ea57c7164d788fb470d5e89c47748f346a30` additionally verifies that
native and patched singleton pairs, and native and patched dynamic pairs, use
identical batch positions, indexed noise IDs, companions, and request ordering
at every policy call shared by both trajectories.

Tokenizer content provenance was added in commit
`4e4b80dc0e4328a4f50e7229f07b27da9b6d5597`.  Closed-loop startup now records
and resume-validates a recursive manifest of the eight functional tokenizer
files (21,856,089 bytes, aggregate SHA256
`64516a95741d80f99eb0a1202f8dd703f0a5da46f76742804c53a60dbb354580`).
Mutable Hugging Face `.cache` metadata is explicitly excluded.  The final
audit recomputes the tree and the report surfaces its aggregate hash; this adds
content identity without changing the already-bound live replay manifest.

Native graphics provenance was made machine-verifiable in commit
`702a9c72f4461f5d182d5fb9b4a8cb5c829534ac`.  SIMPLER startup now records
the resolved ICD manifest (140 bytes, SHA256
`8bf904abc1c5d6428a1f27cc5d4a3c9783d15231f17ce1ef8d25e9cab1f60d1c`),
the repaired GLVND EGL loader (72,312 bytes, SHA256
`875ecbb2a07d60e32216c9f102965abc1e9cc1da7789558e2e9b8b9c107e231d`),
and the loader/capability environment.  The final audit recomputes both file
identities and requires explicit NVIDIA ICD selection plus disabled implicit
layers; the generated report includes both hashes.

All eight pre-existing helper tests passed under the production policy
environment by direct execution (the environment intentionally lacks the
optional `pytest` runner).  Commit
`bfcb0bf7c700fe5107a5787705e0dbc8a6073747` adds and passes a ninth regression
test proving that mutable `.cache` contents do not change the functional asset
tree manifest.

Commit `add4132fbed5b2fb02ee9aec7fc49f8e9a26230e` makes closed-loop startup run
and retain `vulkaninfo --summary`; final audit requires graphics capability and
the exact native RTX device name, API 1.4.329, NVIDIA 595.91.07 driver, and
device UUID already recorded above.  The synthetic full-cardinality audit
continues to pass with this device-level graphics identity.

The serving-tradeoff generator was made paper-readable in commit
`941d11362094ab4bcc2f47a7fa70be70e648ed41`: it now draws one curve per
configuration, labels numerical-contract pass/fail explicitly, uses a distinct
failure marker, and annotates offered load at each point.  The regenerated
figure was visually inspected against all 20 retained serving summaries; the
large measured latency range is preserved.

Because the exhaustive raw JSONL projects to about 1.42 GiB while this
repository has no Git LFS, commit `679d55634c1b16da887d6eac9828caa669e7dd58`
adds a streaming deterministic gzip packager and verifier.  A live-sample
measurement projects roughly 76 MiB at gzip level 1 (level 6 is used for the
final archive).  After all writers and the exact audit finish, it will retain
lossless commit-safe copies plus `results/archive_manifest.json` containing
raw/compressed sizes, raw line counts, and both SHA256 identities.  A
1,000-record fixture produced byte-identical archives across rebuilds and
passed streamed decompression verification.  The live raw files remain the
authoritative analysis inputs and were not touched.

Top-level documentation was corrected in commit
`2414244c6c46c715f902b5a83c6f4487256b5c29`: it no longer describes the
repository as synthetic-only or claims simulator/serving work is excluded.
The lightweight sample and the full pretrained campaign are now clearly
separated, with direct links to the runbook, retained results, generated A-H
report, and machine-readable audit gate.

A second read-only live-prefix integrity checkpoint was taken after the
held-out arrangement stream crossed 10,000 records.  The immutable prefix
contained 10,119 parseable records with 10,119 unique composite arrangement
keys: 439 complete `(request_id, implementation)` groups passed the exact
23-record matrix (1 singleton, 16 batch-position, 3 permutation, and 3
partition records; physical B={1,2,4,8} counts 2/6/8/7), and the only partial
group was the actively written full-invariant group with 22 unique records.
All prefix records matched the frozen numerical-policy, operator, and
checkpoint identities.  All 4,619 physical-nonsingleton full-invariant
comparisons were bit-exact with maximum error zero.  The corresponding native
prefix had 4,620/4,620 violations and maximum absolute error
`0.008353471755981445`.  The 219 singleton-fidelity rows had 219 unique request
IDs; none was bit-exact to native singleton, and their maximum normalized
absolute error was `4.506111145019531e-05`.  These are retained as observed
prefix results rather than used to select or alter the still-running campaign.
The writer advanced to 10,126 arrangements and 220 fidelity rows during the
audit, while all three tmux sessions and the original GPU-owning PID remained
healthy.

A third read-only live-prefix integrity checkpoint was taken after the
held-out arrangement stream crossed 20,000 records.  The immutable prefix
captured at `2026-09-28T06:29:46Z` contained 20,132 parseable records and
20,132 unique composite arrangement keys.  Its 875 complete
`(request_id, implementation)` groups all passed the exact 23-record matrix
(1 singleton, 16 batch-position, 3 permutation, and 3 partition records;
physical B={1,2,4,8} counts 2/6/8/7).  The only incomplete group was the
actively written full-invariant group, with 7 rows and 7 unique arrangement
keys; there were no malformed or overfull groups.  Every arrangement and all
437 fidelity records matched the numerical-policy, frozen-operator, and
checkpoint identities bound by the held-out runtime record and numerical
freeze.  All 9,182 physical-nonsingleton full-invariant comparisons were
bit-exact with maximum error zero.  Native had 9,198/9,198 violations with
maximum absolute error `0.008353471755981445`.  The fidelity prefix contained
437 unique request IDs, zero bit-exact patched-vs-native singleton pairs, and
maximum normalized absolute error `4.506111145019531e-05`.  These remain an
unfiltered observational checkpoint, not a basis for selecting or changing
the campaign.  The append-only writer advanced to 20,139 arrangements during
the audit, while all three tmux sessions and the original GPU-owning PID
remained healthy.

A fourth read-only live-prefix integrity checkpoint was taken after the
held-out arrangement stream crossed the 25% boundary.  The immutable prefix
captured at `2026-09-28T07:39:29Z` contained 27,734 parseable records and
27,734 unique composite arrangement keys.  Its 1,205 complete
`(request_id, implementation)` groups all passed the canonical exact
23-record matrix (1 singleton, 16 batch-position, 3 permutation, and 3
partition records; physical B={1,2,4,8} counts 2/6/8/7).  The only incomplete
group was the actively written full-invariant group, with 19 rows and 19
unique arrangement keys; there were no duplicate, malformed, invalid, or
overfull groups.  Every arrangement and all 602 fidelity records matched the
frozen numerical-policy, operator, and checkpoint identities.  All 12,659
physical-nonsingleton full-invariant comparisons were bit-exact with maximum
error zero.  Native had 12,663/12,663 violations with maximum absolute error
`0.008353471755981445`.  The fidelity prefix contained 602 unique request
IDs, zero bit-exact patched-vs-native singleton pairs, and maximum normalized
absolute error `4.506111145019531e-05`.  These observations were retained
without selecting or changing the ongoing campaign.  The append-only writer
advanced to 27,737 arrangements during the audit, while all three tmux
sessions and the original GPU-owning PID remained healthy.

A fifth read-only live-prefix integrity checkpoint was taken at the planned
50,000-record boundary.  The immutable first-50,000-line prefix, audited at
`2026-09-28T11:03:42Z`, contained 50,000 parseable records and 50,000 unique
composite arrangement keys.  Its 2,173 complete
`(request_id, implementation)` groups all passed the canonical exact
23-record matrix (1 singleton, 16 batch-position, 3 permutation, and 3
partition records; physical B={1,2,4,8} counts 2/6/8/7).  The only incomplete
group was the actively written full-invariant group, with 21 rows and 21
unique arrangement keys; there were no malformed, duplicate, invalid
complete, or overfull groups.  Every arrangement and all 1,090 records in the
fixed fidelity prefix matched the frozen numerical-policy, operator, and
checkpoint identities.  All 22,825 physical-nonsingleton full-invariant
comparisons were bit-exact with maximum error zero.  Native had
22,827/22,827 violations with maximum absolute error
`0.008353471755981445`.  The fidelity prefix contained 1,090 unique request
IDs, zero bit-exact patched-vs-native singleton pairs, and maximum normalized
absolute error `5.370378494262695e-05`.  The increased fidelity maximum is
retained as observed evidence and was not used to select, alter, or restart
the campaign.  The append-only writer advanced to 50,185 arrangements during
the audit, while all three tmux sessions and the original GPU-owning PID
remained healthy.

A sixth read-only live-prefix integrity checkpoint was taken at the exact 50%
boundary.  The immutable first-55,200-arrangement and first-1,200-fidelity
prefixes, audited at `2026-09-28T11:50:52Z`, were both fully parseable.  All
55,200 composite arrangement keys were unique, and the prefix comprised
exactly 1,200 request IDs and 2,400 complete `(request_id, implementation)`
groups.  Every group passed the canonical 23-record matrix (1 singleton, 16
batch-position, 3 permutation, and 3 partition records; physical
B={1,2,4,8} counts 2/6/8/7), with no partial, malformed, duplicate, invalid,
or overfull group.  Every arrangement and fidelity row matched the frozen
numerical-policy SHA `a2576bbac02fca0ba1868843475f4a644835bdf1`, operator
SHA `14dafb5fd84350dc796a96cbecb566ca6295593f`, and checkpoint SHA256
`1da40985d963ddf394fa63e7db5ddd4cd8082d5c6ce0b2a0cae56abd8b6ec0eb`.
All 25,200 physical-nonsingleton full-invariant comparisons were bit-exact
with maximum error zero.  Native had 25,200/25,200 violations with maximum
absolute error `0.008353471755981445`.  The fidelity prefix contained exactly
1,200 unique request IDs matching the arrangement request set, zero bit-exact
patched-vs-native singleton pairs, and maximum normalized absolute error
`5.370378494262695e-05`; the maximum was retained without selection or
alteration.  The low-priority streaming audit completed in 6.5 seconds while
the append-only writer advanced to 55,312 arrangements and 1,202 fidelity
rows, and all three tmux sessions and their original PIDs remained healthy.

A seventh read-only live-prefix integrity checkpoint was taken at the exact
75% boundary after both counters crossed it at `2026-09-28T16:02:23Z`.  The
immutable first-82,800-arrangement and first-1,800-fidelity prefixes were
fully parseable and passed a low-priority streaming audit in 8.722 seconds.
All 82,800 composite arrangement keys were unique; the prefix contained
exactly 1,800 request IDs and 3,600 complete
`(request_id, implementation)` groups.  Every group passed the canonical
23-record matrix (1 singleton, 16 batch-position, 3 permutation, and 3
partition records; physical B={1,2,4,8} counts 2/6/8/7), with no malformed,
duplicate, invalid, partial, or overfull group.  Every arrangement and
fidelity row matched numerical-policy SHA
`a2576bbac02fca0ba1868843475f4a644835bdf1`, frozen-operator SHA
`14dafb5fd84350dc796a96cbecb566ca6295593f`, and checkpoint SHA256
`1da40985d963ddf394fa63e7db5ddd4cd8082d5c6ce0b2a0cae56abd8b6ec0eb`.
All 37,800 physical-nonsingleton full-invariant comparisons were bit-exact
with maximum error zero.  Native had 37,800/37,800 violations and maximum
absolute action error `0.14864274859428406`.  The fidelity prefix contained
exactly 1,800 unique request IDs matching the arrangement request set, zero
bit-exact patched-vs-native singleton pairs, and maximum normalized absolute
error `0.00021225214004516602`.

The two increased maxima were retained and investigated rather than filtered.
The native maximum belongs to
`open_drawer/episode-013/observation-03/noise-2`: its complete 46-record
neighborhood is canonical and finite, the native B=4 batch-position records
repeat the `0.14864274859428406` effect across placements and companion types,
and all 21 corresponding full-invariant nonsingleton records remain exactly
equal with zero error.  The fidelity maximum belongs to the distinct request
`open_drawer/episode-020/observation-06/noise-2`; its normalized and
denormalized comparisons are finite, shape-equal, and have zero nonfinite
elements, while its complete full-invariant arrangement group is also exact.
Thus the observations are valid numerical sensitivity evidence, not record
corruption or an invariant-path failure.  By `2026-09-28T16:04:21Z` the live
append-only writer had advanced to 83,134 arrangements and 1,807 fidelity
rows, and all three tmux sessions and their original PIDs remained healthy.

The live RTX heldout campaign crossed its exact 85% cardinality thresholds
(93,840 arrangement rows and 2,040 fidelity rows) without interruption.  At
`2026-09-28T17:42:16Z` the append-only outputs contained 93,920 arrangements
and 2,041 fidelity rows.  Across the preceding 30-minute read-only monitoring
window the same worker PID 19045 remained live at approximately 424% CPU with
15,406 MiB VRAM allocated, both downstream watcher sessions retained their
original live PIDs (19328 and 19632), and `/workspace` retained 126 GiB free.
No worker, watcher, output, or evaluation was restarted or duplicated.

The same uninterrupted campaign crossed its exact 90% cardinality thresholds
(99,360 arrangement rows and 2,160 fidelity rows) during the next sustained
read-only monitoring window.  At `2026-09-28T18:33:08Z` the append-only
outputs contained 99,522 arrangements and 2,163 fidelity rows.  Worker PID
19045 remained live at approximately 424% CPU with 15,406 MiB VRAM allocated,
the continuation and finalization watchers retained PIDs 19328 and 19632, and
all three tmux sessions remained live with 126 GiB free on `/workspace`.  No
worker, watcher, output, or evaluation was restarted or duplicated.

The exhaustive heldout campaign completed normally at
`2026-09-28T20:15:32Z` with exactly 110,400 arrangement records and 2,400
singleton-fidelity records, after which its tmux session exited and the
already-running continuation watcher launched the paired real SIMPLER
campaign.  An independent low-priority heldout-only audit parsed the complete
immutable outputs and verified 2,400 unique request IDs, 4,800 canonical
23-record `(request_id, implementation)` groups, and exact frozen provenance
for every arrangement.  All 50,400 physical-nonsingleton full-invariant
comparisons were bit-exact with maximum error zero; native had
50,400/50,400 violations with maximum absolute action error
`0.14864274859428406`.  No patched-vs-native singleton pair was bit-exact,
and the final maximum normalized fidelity error was
`0.00029343366622924805`.

That increased fidelity maximum was retained and investigated rather than
filtered.  It belongs to
`close_drawer/episode-010/observation-09/noise-2`: all 28 normalized elements
differ but are finite (relative L2 `0.00021551875838269393`), and its complete
46-record arrangement neighborhood has two canonical 23-record groups with
the frozen policy/operator/checkpoint provenance.  All 21 corresponding
full-invariant nonsingletons are exact with zero error, while all 21 native
nonsingletons violate invariance with local maximum `0.06984400749206543`;
there are no nonfinite values.  This is valid singleton/native numerical
sensitivity evidence rather than malformed data or an invariant-path
failure.

The paired real SIMPLER campaign started at `2026-09-28T20:15:30Z` under
continuation PID 19328.  Its runtime manifest records 800 expected episodes,
50 initializations per task, four paired native/patched singleton/dynamic
conditions, seed 20250311, replay SHA256
`7b2e3cee5635112b885862865a879d4e4a089efb292b10d50fdbf5e222f1dd08`,
numerical-policy SHA `a2576bbac02fca0ba1868843475f4a644835bdf1`, frozen
operator SHA `14dafb5fd84350dc796a96cbecb566ca6295593f`, checkpoint SHA256
`1da40985d963ddf394fa63e7db5ddd4cd8082d5c6ce0b2a0cae56abd8b6ec0eb`,
and native NVIDIA Vulkan ICD `/etc/vulkan/icd.d/nvidia_icd.json` on the RTX PRO
6000 Blackwell.  By `2026-09-28T20:18:58Z`, 6/800 real episodes were retained;
the policy and SIMPLER worker processes held 15,406 MiB and 515 MiB GPU memory
respectively, GPU utilization was 99%, and both continuation and finalization
watchers remained live.

At `2026-09-28T20:33:36Z`, the real SIMPLER output reached 40/800 episodes.
An independent live-prefix audit taken immediately afterward parsed 41 rows
without error and found 41 unique `(task, initialization_id, condition)` keys,
ten complete four-condition initialization blocks plus only the currently
in-progress block, zero nonfinite values, and exact checkpoint, numerical-policy,
and frozen-operator provenance on every row.  The campaign continued normally
to 42/800 by `2026-09-28T20:34:26Z`; evaluator PID 65344, worker PID 65656,
continuation PID 19328, and finalizer PID 19632 all remained live with stable
15,406 MiB / 515 MiB policy/simulator GPU allocations and 126 GiB free disk.

A second live SIMPLER prefix audit at 62 rows (15 complete `pick_can`
initialization blocks plus the active partial block) found zero record-schema or
per-action scheduling failures and zero native-vs-patched paired request-schedule
mismatches.  The complete blocks provisionally contained two genuine outcome-
discordant initializations: ID 1 succeeded in every condition except
`patched_singleton`, and ID 2 failed in both singleton conditions and
`native_dynamic` but succeeded in `patched_dynamic`.  These rows were retained.
All recorded initial low-dimensional trajectory states were bit-identical across
the four paired resets, while action divergence could begin on the first policy
call despite identical indexed noise/scheduling (and two examined blocks remained
action-identical throughout), supporting closed-loop numerical/rendering
sensitivity rather than reset or pairing corruption.  No inference is made from
this incomplete prefix; the prespecified full 800-episode paired bootstrap remains
the behavioral result.  The healthy campaign reached 64/800 at
`2026-09-28T20:43:36Z`.

The immutable first-80-row SIMPLER prefix crossed the exact 10% boundary and
was independently audited as 20 complete `pick_can` initialization blocks: 80
unique task/initialization/condition keys, exactly 20 rows per condition, zero
schema or frozen-provenance failures, zero nonfinite values, and zero paired
native-vs-patched request-schedule mismatches.  Provisional success counts in
this deliberately unfiltered prefix were 19/20 for each native condition,
18/20 for `patched_singleton`, and 20/20 for `patched_dynamic`; the same two
previously investigated blocks were the only discordant blocks.  A separate
live-process provenance check confirmed that both evaluator and simulator
actually carried `VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json`,
`VK_LOADER_LAYERS_DISABLE=~implicit~`, and
`NVIDIA_DRIVER_CAPABILITIES=all`; their resolved executables were Python 3.12
and 3.10 respectively, all four evaluator source hashes still matched the
launch manifest, and the retained Vulkan summary contained the exact RTX PRO
6000 Blackwell, Vulkan 1.4.329, and NVIDIA 595.91.07 identities.  The campaign
remained healthy at 83/800 at `2026-09-28T20:51:57Z` with no resume events.

The immutable first-120-row prefix was then audited at the exact 30-complete-
block boundary: all 120 composite keys were unique, all four conditions had 30
rows, frozen provenance and record schemas were exact, all values were finite,
and paired request schedules had zero mismatches.  Success counts were 29/30
for each native condition, 28/30 for `patched_singleton`, and 30/30 for
`patched_dynamic`; initialization IDs 1 and 2 remained the only discordant
blocks, so no new discordance appeared between rows 81 and 120.  A whole-file
snapshot taken moments later correctly observed the next initialization as an
ordinary two-row partial block while the append-only evaluator advanced, not as
an integrity failure.

The first closed-loop task, `pick_can`, completed at the exact 200/800 boundary
at `2026-09-28T21:42:00Z`.  Its 200 rows comprise all 50 initialization IDs and
exactly 50 rows per condition, with 200 unique composite keys, exact frozen
provenance, finite trajectories/actions, and zero native-vs-patched request-
schedule mismatches.  Success counts were 49/50 `native_singleton`, 49/50
`native_dynamic`, 48/50 `patched_singleton`, and 49/50 `patched_dynamic`, giving
the prespecified point estimates 0.00 native dynamic-minus-singleton, +0.02
patched dynamic-minus-singleton, and -0.02 patched-minus-native singleton before
the scheduled full-campaign bootstrap.  Three initialization blocks were
discordant: IDs 1 and 2 described above, plus ID 49, where only
`patched_dynamic` failed.  ID 49 retained all 80 actions and 81 states; its
patched dynamic-vs-singleton action difference began at policy call 0 with
maximum normalized difference `1.019736684858799`, and end-effector trajectories
separated by at most `0.09413562085353881` m and `0.21536100954367568` rad.

An unfiltered reset audit identified a SIMPLER limitation that must qualify the
behavioral interpretation.  All four conditions received the same seed and
predefined episode ID, all static reset fields matched, and the recorded initial
robot/EEF state was bit-identical in all 50 blocks.  However, the environment
drops the object from the seeded pose and settles physics for 1.5 seconds, so the
returned post-settle object pose was byte-identical in only 1/50 blocks.  The
within-block maximum pairwise settled-pose variation had median/maximum
translation `4.8817010727501864e-06`/`8.963258336119213e-05` m and
median/maximum rotation `2.9937119410221702e-05`/`0.002749904562089016` rad.
Discordant ID 49 had only `1.0000000000287557e-06` m and
`8.792089194377266e-06` rad variation (47th and 45th of 50 when ranked largest
first), so its disagreement is not explained by unusually large reset jitter,
but post-settle simulator nondeterminism remains a disclosed paired-evaluation
confound rather than being hidden.  At the task transition, the evaluator
normally replaced the pick-can worker with PID 69761 for
`google_robot_move_near_v0`; the same evaluator/watcher chain remained live and
the campaign advanced to 216/800 without a resume event.

At the immutable 240/800 boundary, a cross-task audit covered all 200
`pick_can` rows plus ten complete `move_near` blocks.  All 240 composite keys
were unique; all groups were complete; frozen provenance, trajectory shape, and
finiteness checks passed; and native-vs-patched request schedules still had zero
mismatches.  The deliberately unfiltered first-ten `move_near` success counts
were 7/10 `native_singleton`, 4/10 `native_dynamic`, 5/10
`patched_singleton`, and 8/10 `patched_dynamic`, with seven discordant blocks.
These are incomplete-prefix observations, not replacements for the prescribed
50-initialization bootstrap.

The same reset limitation is visible in `move_near`, where two objects settle
after the identical seeded scene request.  Across its first ten complete blocks,
all static reset fields and initial robot/EEF states matched exactly.  Maximum
within-block source-object translation/rotation variation reached
`0.0009087953565022463` m / `0.010605405901099164` rad; target-object maxima
were the same, with medians `3.2393017087094895e-05` m and
`0.0008585960286123565` rad.  Outcome discordance occurred under both small and
larger pose variation.  Accordingly, the final paired SIMPLER result must be
reported as the observed end-to-end effect under repeated same-seed scene
construction, with post-settle physics nondeterminism disclosed as a limitation,
not as a pure causal estimate of inference batching alone.  The healthy live run
continued to 244/800 at `2026-09-28T22:00:22Z`.

The second closed-loop task, `move_near`, completed normally at the exact
400/800 campaign boundary.  An independent immutable-prefix audit found 400
unique keys, 100 complete initialization groups, exact initialization-ID
coverage, exact frozen provenance, finite actions/trajectories, and zero paired
request-schedule mismatches.  Complete `move_near` successes were 39/50
`native_singleton`, 34/50 `native_dynamic`, 38/50 `patched_singleton`, and
40/50 `patched_dynamic`; 20/50 blocks had at least one outcome disagreement.
The prespecified point estimates before the scheduled bootstrap are therefore
-0.10 native dynamic-minus-singleton, +0.04 patched dynamic-minus-singleton,
and -0.02 patched-minus-native singleton.

The full 50-block `move_near` reset audit retained exact static reset fields and
bit-identical initial robot/EEF state in all blocks.  For the settled source
object, maximum within-block pairwise translation variation had
median/p95/maximum `2.078282181183009e-06` / `0.0003055689007242743` /
`0.0009087953565022463` m, while rotation had
`2.4184836211744445e-05` / `0.00855102064649915` /
`0.010605405901099164` rad.  The corresponding target-object values were
`2.298594621019995e-06` / `0.0002872197459681788` /
`0.0009087953565022463` m and `2.6253809463921508e-05` /
`0.008564511763336917` / `0.010605405901099164` rad.  These observations are
retained as the simulator-settling limitation described above.  At the normal
task boundary, evaluator PID 65344 replaced the move-near worker with PID 73878
for `google_robot_open_drawer`; the campaign reached 403/800 without a resume
event while both orchestration watchers remained live.

The third closed-loop task, `open_drawer`, completed normally at the exact
600/800 campaign boundary at `2026-09-29T01:08:57Z`.  The immutable-prefix
audit found 600 unique composite keys, 150 complete four-condition blocks,
exact frozen provenance, finite actions and trajectories, consistent
trajectory lengths, and zero paired request-schedule mismatches.  Complete
`open_drawer` successes were 25/50 `native_singleton`, 27/50
`native_dynamic`, 22/50 `patched_singleton`, and 22/50 `patched_dynamic`,
giving the prespecified point estimates +0.04 native dynamic-minus-singleton,
0.00 patched dynamic-minus-singleton, and -0.06 patched-minus-native singleton
before the scheduled full-campaign bootstrap.  Twelve initialization blocks
were outcome-discordant: IDs 0, 8, 12, 15, 22, 26, 27, 34, 36, 41, 47, and
48.  In contrast to the two object-settling tasks, all 50 `open_drawer` blocks
had byte-identical `reset_info` and initial trajectory state across all four
conditions, so the object-settling qualification does not apply to this task.
At the normal transition, the same evaluator replaced worker PID 73878 with
PID 79117 for `google_robot_close_drawer`; both orchestration watchers remained
live and no resume event occurred.

## Final completion and retention

The paired real SIMPLER campaign completed normally at exactly 800/800 episodes
at `2026-09-29T03:13:18Z`.  The independent immutable full-file audit found 800
unique `(task, initialization_id, condition)` keys, 200 complete four-condition
blocks, exactly 200 records per task and condition, exact frozen checkpoint,
numerical-policy, and operator provenance, finite actions/trajectories,
consistent trajectory lengths, zero paired request-schedule mismatches, and no
resume events.  The final `close_drawer` success counts were 36/50
`native_singleton`, 37/50 `native_dynamic`, 38/50 `patched_singleton`, and
38/50 `patched_dynamic`, giving point estimates +0.02 native
dynamic-minus-singleton, 0.00 patched dynamic-minus-singleton, and +0.04
patched-minus-native singleton.  Its four outcome-discordant initialization
IDs were 4, 7, 20, and 25, while all 50 returned resets and recorded initial
states were byte-identical across conditions.

The predeclared seeded 10,000-resample matched-initialization bootstrap then
completed and retained `results/simpler/summary.json`.  Native
dynamic-minus-singleton estimates (95% bootstrap intervals) were 0.00
`pick_can` `[0.00, 0.00]`, -0.10 `move_near` `[-0.22, 0.02]`, +0.04
`open_drawer` `[-0.06, 0.14]`, and +0.02 `close_drawer` `[-0.04, 0.10]`.
Patched dynamic-minus-singleton estimates were +0.02 `[-0.04, 0.10]`, +0.04
`[-0.08, 0.16]`, 0.00 `[0.00, 0.00]`, and 0.00 `[0.00, 0.00]` in the same
task order.  Patched-minus-native singleton estimates were -0.02
`[-0.06, 0.00]`, -0.02 `[-0.16, 0.12]`, -0.06 `[-0.18, 0.08]`, and +0.04
`[0.00, 0.10]`.  These are the complete prespecified outcomes; none were
filtered or replaced based on direction or magnitude.

A reproducible reset-pairing analysis was retained at
`results/simpler/reset_audit.json` and is now required by the final audit and
report.  It independently binds to the 800-row episode SHA-256
`0caa7b7e33b89f8d8acf9e580e836904aece750eebb321d26e677f050345d59f`.
All four tasks had exact static reset fields and bit-identical recorded initial
robot/EEF state in 50/50 blocks.  Raw returned reset records were exact in 1/50
`pick_can`, 3/50 `move_near`, and 50/50 for both drawer tasks.  Full
per-initialization object-pose variations are retained, and the generated
report therefore qualifies the object-task comparisons as end-to-end effects
under repeated same-seed scene construction with post-reset physics
nondeterminism, while the drawer-task comparisons have exact reset pairing.

The cardinality-gated finalizer regenerated the held-out and diagnostic flow
summaries, all tables, all three figures, `results/EXPERIMENT_REPORT.md`, and
`results/audit.json`; all figures were visually inspected against their source
tables.  The strengthened independent audit reports
`complete_campaign_verified`, verifies the exact RTX/Vulkan/repository/source
provenance, all result cardinalities and schemas, the full SIMPLER/bootstrap
structure, the reset qualification, and the deterministic archives.  The
nine repository helper tests also passed by direct invocation under the
production policy environment (the optional pytest and ruff runners are not
installed in this pod).

The three authoritative raw JSONLs remain present locally and are excluded
from Git only by their exact paths.  Deterministic gzip level-6 archives were
built twice with `mtime=0`, were byte-identical across rebuilds, and passed
streamed decompression/source verification.  `results/archive_manifest.json`
retains 113,600 total raw records and both identities: held-out invariance
`e337116406c4a7d641cd03c160c045a1520cb5046ed4a200028924619bfcb241`
raw / `7f801503424ae8640e5b181bb1af61507014e815c788dfe281b4066ee09822ae`
gzip; singleton fidelity
`5cca18880e387e8102e7fb957176969e1b8ddc3d5cce50241ee8b0d09d97f1ea`
raw / `6d6f5e70d223707874dc74f146355fc2c400ec4d4d72cc4f42cf309e7239f3f4`
gzip; and SIMPLER episodes
`0caa7b7e33b89f8d8acf9e580e836904aece750eebb321d26e677f050345d59f`
raw / `f9444145f1e44c063245d10bca266362ec79a12f2125dcac800d7074e25b2e0f`
gzip.  The three archives occupy 76,200,878 bytes versus 1,660,750,213 raw
bytes, with 126 GiB of pod storage still free.

The retained final repository state was committed on `main` as
`49948971dec2dfed1a06ad5a1d7897adc72133b2` (`retain completed held-out and
SIMPLER campaigns`).  The main worktree was clean immediately afterward;
`git fsck --no-dangling`, archive verify-only, the exact raw cardinality check,
and the machine-readable audit gate all passed.  The external operator checkout
remained at launch-adaptation commit
`8fdbcada1401608622a07e1d35b645973deb905d`; it has no source changes (only a
generated untracked Python bytecode cache from validation).

The final requirement-by-requirement review against Section 33 passed: the
complete pretrained policy and actual dispatch evidence are retained; the
environment and numerical implementation are frozen; 72,000 real-shape
operator records cover 720 configurations (100 seeds each); the 1,000-
observation replay and diagnostic/held-out split counts are exact; all causal
ablations and ten flow steps are present; all 800 paired closed-loop episodes
and seeded bootstrap results are present; kernel, complete-policy, and serving
measurements have 3,600, 4,000, and 4,000 raw records respectively; all raw
measurements remain local and losslessly archived; and every required table,
figure, report section, provenance assertion, and machine-readable audit gate
is present in the clean committed repository state.
