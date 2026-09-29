# Experimental report

This report is generated from the retained machine-readable measurements. Missing campaigns are marked unavailable rather than estimated.

## Provenance

- Policy revision: `a2576bbac02fca0ba1868843475f4a644835bdf1`
- Operator revision: `14dafb5fd84350dc796a96cbecb566ca6295593f`
- RTX launch-only operator adaptation: `8fdbcada1401608622a07e1d35b645973deb905d`
- RTX evaluation harness revision: `e12fc62cc2afb8236dd7bf4176e385863e2286f9`
- SIMPLER evaluator revision: `2414244c6c46c715f902b5a83c6f4487256b5c29`
- SIMPLER evaluator source SHA-256: `ca8fc8731cf4685dc3217947805bfab9f14607539270ad3bc904001860b35791`
- SIMPLER revision: `59ad9e1539042ed333fd8ebba1b0395f5662f0bd`
- ManiSkill2_real2sim revision: `91d154bfd864577f8d2e80f3fc2f8b4d9df9ae5c`
- Tokenizer tree SHA-256: `64516a95741d80f99eb0a1202f8dd703f0a5da46f76742804c53a60dbb354580`
- NVIDIA Vulkan ICD manifest SHA-256: `8bf904abc1c5d6428a1f27cc5d4a3c9783d15231f17ce1ef8d25e9cab1f60d1c`
- GLVND EGL loader SHA-256: `875ecbb2a07d60e32216c9f102965abc1e9cc1da7789558e2e9b8b9c107e231d`
- Checkpoint SHA-256: `1da40985d963ddf394fa63e7db5ddd4cd8082d5c6ce0b2a0cae56abd8b6ec0eb`
- RTX GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Evaluation scope used below: **held-out replay**

## A. Does native inference violate request-level batch invariance?

Yes within the held-out replay campaign: 2400/2400 requests violated (100.000%), and 50400/50400 non-singleton arrangements violated (100.000%). The maximum normalized action error was 0.148642749.
Tested transformations=['batch_position', 'partition', 'permutation'], batch sizes=[2, 4, 8], target positions=['first', 'last', 'middle'], companions=['diverse', 'duplicate']. Across arrangement-level maximum errors: median=7.43195415e-07, p95=0.000219002366, p99=0.00068221949.
In the separate same-request-set transformation matrix, native failed 5/10 aggregate comparisons: all five partition changes failed, while all five restored permutations were exact.

### Separate BF16 policy campaign

- `native`: 18/18 arrangement violations; maximum error 0.0087890625.
- `native_deterministic`: 18/18 arrangement violations; maximum error 0.0087890625.
- `existing_invariant_ops`: 6/18 arrangement violations; maximum error 0.001953125.
- `invariant_plus_patch_projection`: 6/18 arrangement violations; maximum error 0.00390625.
- `explicit_per_matrix_attention`: 0/18 arrangement violations; maximum error 0.
- `full_invariant`: 0/18 arrangement violations; maximum error 0.

## B. Where does native execution first diverge?

First-divergence category frequencies across the one retained frozen B=2 diverse trace per implementation mode (descriptive mode counts, not a held-out incidence estimate): visual patch projection=0/6, attention/BMM=3/6, other boundary=2/6, no divergence=1/6.
- `existing_invariant_ops`: prefill.step_0.action_joint_model.layer_0.post_attn.attn_outputs_final#invocation_0.vlm
- `explicit_per_matrix_attention`: prefill.step_0.action_joint_model.layer_0.post_attn.attn_outputs_final#invocation_0.vlm
- `full_invariant`: none (all traced tensors exact)
- `invariant_plus_patch_projection`: prefill.step_0.action_joint_model.layer_0.post_attn.attn_outputs_final#invocation_0.vlm
- `native`: proprio_embeds#invocation_0
- `native_deterministic`: proprio_embeds#invocation_0
Native local replay at `proprio_encoder` used bit-identical inputs and reproduced a max error of 1.1920929e-07 through aten::addmm, aten::as_strided, aten::linear, aten::reshape, aten::t, aten::transpose, aten::view.
The VLM layer-0 output projection replay was locally batch-sensitive in both the patch-projection and explicit-attention modes (max 0.000442504883), but exact for the tested full-invariant input.

## C. How do differences propagate through the flow solver?

- `native`:
  - step 1: exact 0/50400, median=1.1920929e-07, p95=2.69465148e-05, maximum=0.000189065933.
  - step 2: exact 0/50400, median=2.38418579e-07, p95=5.31012192e-05, maximum=0.000384926796.
  - step 3: exact 0/50400, median=2.98023224e-07, p95=7.93993473e-05, maximum=0.000597268343.
  - step 4: exact 0/50400, median=3.57627869e-07, p95=0.0001053527, maximum=0.000836059451.
  - step 5: exact 0/50400, median=4.47034836e-07, p95=0.000132020563, maximum=0.00112195313.
  - step 6: exact 0/50400, median=5.36441803e-07, p95=0.000160613284, maximum=0.00147402287.
  - step 7: exact 0/50400, median=5.96046448e-07, p95=0.000180710107, maximum=0.00185739994.
  - step 8: exact 0/50400, median=6.85453415e-07, p95=0.000216573477, maximum=0.00289389491.
  - step 9: exact 0/50400, median=7.4505806e-07, p95=0.0002449058, maximum=0.0100307167.
  - step 10: exact 0/50400, median=7.4505806e-07, p95=0.000219158828, maximum=0.148642749.
- `full_invariant`:
  - step 1: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 2: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 3: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 4: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 5: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 6: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 7: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 8: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 9: exact 50400/50400, median=0, p95=0, maximum=0.
  - step 10: exact 50400/50400, median=0, p95=0, maximum=0.
The frozen B=2 diverse diagnostic trace supplies the causal intermediate paths not run across the exhaustive held-out set (mean and maximum below are over the 28 action-state elements at each step):
- `native` step:mean/max — 1:mean 2.36489411e-08/max 5.96046448e-08, 2:mean 5.26862485e-08/max 1.1920929e-07, 3:mean 7.34081758e-08/max 1.78813934e-07, 4:mean 1.00682623e-07/max 2.5331974e-07, 5:mean 1.26992485e-07/max 2.83122063e-07, 6:mean 1.44571199e-07/max 3.12924385e-07, 7:mean 1.6078619e-07/max 3.42726707e-07, 8:mean 1.63014712e-07/max 3.57627869e-07, 9:mean 1.44554568e-07/max 3.57627869e-07, 10:mean 1.19990935e-07/max 4.47034836e-07
- `existing_invariant_ops` step:mean/max — 1:mean 2.98023224e-08/max 1.1920929e-07, 2:mean 9.8786716e-08/max 3.57627869e-07, 3:mean 1.53202564e-07/max 4.76837158e-07, 4:mean 2.10179548e-07/max 5.96046448e-07, 5:mean 2.64096473e-07/max 7.74860382e-07, 6:mean 3.22703272e-07/max 1.07288361e-06, 7:mean 3.82490855e-07/max 1.13248825e-06, 8:mean 4.22421311e-07/max 1.1920929e-06, 9:mean 4.40681885e-07/max 1.60932541e-06, 10:mean 5.67308494e-07/max 2.27987766e-06
- `invariant_plus_patch_projection` step:mean/max — 1:mean 2.98023224e-08/max 1.1920929e-07, 2:mean 9.8786716e-08/max 3.57627869e-07, 3:mean 1.53202564e-07/max 4.76837158e-07, 4:mean 2.10179548e-07/max 5.96046448e-07, 5:mean 2.64096473e-07/max 7.74860382e-07, 6:mean 3.22703272e-07/max 1.07288361e-06, 7:mean 3.82490855e-07/max 1.13248825e-06, 8:mean 4.22421311e-07/max 1.1920929e-06, 9:mean 4.40681885e-07/max 1.60932541e-06, 10:mean 5.67308494e-07/max 2.27987766e-06
- `full_invariant` step:mean/max — 1:mean 0/max 0, 2:mean 0/max 0, 3:mean 0/max 0, 4:mean 0/max 0, 5:mean 0/max 0, 6:mean 0/max 0, 7:mean 0/max 0, 8:mean 0/max 0, 9:mean 0/max 0, 10:mean 0/max 0

## D. Does the full invariant path achieve exact action equality?

In the held-out replay records, 0/50400 non-singleton arrangements failed exact equality; maximum error was 0.
The full path also had 0/10 failures across restored permutations and partitioning at B=2/4/8/16/32.

## E. How faithful is invariant singleton inference to native singleton inference?

Across 2400 held-out request/noise pairs, 0 were bit-identical. Maximum absolute error=0.000293433666, median=5.06639481e-07, p95=2.50339508e-06. This patched-vs-native singleton comparison is distinct from request-level batch invariance.

## F. Do numerical differences affect behavior?

- Action divergence: measured in the held-out replay numerical campaign; 50400/50400 native arrangements differed.
The paired closed-loop campaign completed 800/800 episodes.
Trajectory formulas: position is the Euclidean norm of paired end-effector XYZ differences in meters; rotation is the unit-quaternion geodesic 2*arccos(|q1 dot q2|) in radians; gripper is absolute scalar difference in simulator-native units. States are aligned by step index through the shorter trajectory, and terminal denotes the last state of that aligned prefix.
Reset-pairing qualification: all four conditions used the same indexed seed, static reset fields matched in 50/50 blocks for every task, and the recorded initial robot/end-effector state was bit-identical in 50/50 blocks for every task. Returned raw reset records were byte-identical in 1/50 `pick_can`, 3/50 `move_near`, 50/50 `open_drawer`, and 50/50 `close_drawer` blocks.
The non-exact object-task records contain small returned post-reset object-pose differences. For `pick_can`, maximum within-block pairwise translation variation had median/p95/maximum 4.88170107e-06/7.01113589e-05/8.96325834e-05 m, and rotation variation had median/p95/maximum 2.99371042e-05/0.00216682288/0.00274990456 rad.
For `move_near`, the corresponding source-object translation and rotation median/p95/maximum values were 2.07828218e-06/0.000305568901/0.000908795357 m and 2.41848384e-05/0.00855102065/0.0106054059 rad; the target-object values were 2.29859462e-06/0.000287219746/0.000908795357 m and 2.62537906e-05/0.00856451176/0.0106054059 rad.
Accordingly, object-task estimates are observed end-to-end effects under repeated same-seed scene construction and retain post-reset physics nondeterminism as a disclosed confound; the two drawer tasks had exact returned resets and do not carry that qualification. Full per-initialization evidence is retained in `results/simpler/reset_audit.json`.
- `close_drawer` success rates: native singleton=0.720, native dynamic=0.740, patched singleton=0.760, patched dynamic=0.760.
  - `native_dynamic_minus_singleton`: paired estimate=0.020, 95% bootstrap CI=[-0.040, 0.100], disagreements=3/50.
  - `patched_dynamic_minus_singleton`: paired estimate=0.000, 95% bootstrap CI=[0.000, 0.000], disagreements=0/50.
  - `patched_minus_native_singleton`: paired estimate=0.040, 95% bootstrap CI=[0.000, 0.100], disagreements=2/50.
  - `native_dynamic_minus_singleton` trajectory differences across matched pairs: maximum position=0.231427493 m, rotation=0.728212737 rad, gripper=0.831999421 native units; maximum aligned-terminal position=0.136049877 m, rotation=0.711981555 rad, gripper=0.828373671 native units.
  - `patched_dynamic_minus_singleton` trajectory differences across matched pairs: maximum position=0 m, rotation=5.16191366e-08 rad, gripper=0 native units; maximum aligned-terminal position=0 m, rotation=4.21468485e-08 rad, gripper=0 native units.
  - `patched_minus_native_singleton` trajectory differences across matched pairs: maximum position=0.260783082 m, rotation=0.760984646 rad, gripper=0.831925511 native units; maximum aligned-terminal position=0.260783082 m, rotation=0.723424638 rad, gripper=0.827834964 native units.
- `move_near` success rates: native singleton=0.780, native dynamic=0.680, patched singleton=0.760, patched dynamic=0.800.
  - `native_dynamic_minus_singleton`: paired estimate=-0.100, 95% bootstrap CI=[-0.220, 0.020], disagreements=9/50.
  - `patched_dynamic_minus_singleton`: paired estimate=0.040, 95% bootstrap CI=[-0.080, 0.160], disagreements=10/50.
  - `patched_minus_native_singleton`: paired estimate=-0.020, 95% bootstrap CI=[-0.160, 0.120], disagreements=13/50.
  - `native_dynamic_minus_singleton` trajectory differences across matched pairs: maximum position=0.364824868 m, rotation=0.610219983 rad, gripper=0.828519873 native units; maximum aligned-terminal position=0.284828183 m, rotation=0.577589371 rad, gripper=0.825659275 native units.
  - `patched_dynamic_minus_singleton` trajectory differences across matched pairs: maximum position=0.283038983 m, rotation=0.675620949 rad, gripper=0.831656326 native units; maximum aligned-terminal position=0.187383913 m, rotation=0.675620949 rad, gripper=0.82676506 native units.
  - `patched_minus_native_singleton` trajectory differences across matched pairs: maximum position=0.380118292 m, rotation=0.887172915 rad, gripper=0.829012985 native units; maximum aligned-terminal position=0.367446482 m, rotation=0.887172915 rad, gripper=0.825897932 native units.
- `open_drawer` success rates: native singleton=0.500, native dynamic=0.540, patched singleton=0.440, patched dynamic=0.440.
  - `native_dynamic_minus_singleton`: paired estimate=0.040, 95% bootstrap CI=[-0.060, 0.140], disagreements=6/50.
  - `patched_dynamic_minus_singleton`: paired estimate=0.000, 95% bootstrap CI=[0.000, 0.000], disagreements=0/50.
  - `patched_minus_native_singleton`: paired estimate=-0.060, 95% bootstrap CI=[-0.180, 0.080], disagreements=11/50.
  - `native_dynamic_minus_singleton` trajectory differences across matched pairs: maximum position=0.23418019 m, rotation=0.963149088 rad, gripper=0.832009315 native units; maximum aligned-terminal position=0.212133172 m, rotation=0.911173996 rad, gripper=0.831482649 native units.
  - `patched_dynamic_minus_singleton` trajectory differences across matched pairs: maximum position=0 m, rotation=4.21468485e-08 rad, gripper=0 native units; maximum aligned-terminal position=0 m, rotation=2.98023224e-08 rad, gripper=0 native units.
  - `patched_minus_native_singleton` trajectory differences across matched pairs: maximum position=0.237661079 m, rotation=0.881367386 rad, gripper=0.832008958 native units; maximum aligned-terminal position=0.237661079 m, rotation=0.819961756 rad, gripper=0.830790997 native units.
- `pick_can` success rates: native singleton=0.980, native dynamic=0.980, patched singleton=0.960, patched dynamic=0.980.
  - `native_dynamic_minus_singleton`: paired estimate=0.000, 95% bootstrap CI=[0.000, 0.000], disagreements=0/50.
  - `patched_dynamic_minus_singleton`: paired estimate=0.020, 95% bootstrap CI=[-0.040, 0.100], disagreements=3/50.
  - `patched_minus_native_singleton`: paired estimate=-0.020, 95% bootstrap CI=[-0.060, 0.000], disagreements=1/50.
  - `native_dynamic_minus_singleton` trajectory differences across matched pairs: maximum position=0.205426665 m, rotation=0.92054944 rad, gripper=0.573763847 native units; maximum aligned-terminal position=0.0980979568 m, rotation=0.591221249 rad, gripper=0.570108533 native units.
  - `patched_dynamic_minus_singleton` trajectory differences across matched pairs: maximum position=0.118300902 m, rotation=0.366052758 rad, gripper=0.8267802 native units; maximum aligned-terminal position=0.118300902 m, rotation=0.366052758 rad, gripper=0.595608354 native units.
  - `patched_minus_native_singleton` trajectory differences across matched pairs: maximum position=0.204700728 m, rotation=0.930653427 rad, gripper=0.826784968 native units; maximum aligned-terminal position=0.11795902 m, rotation=0.612695846 rad, gripper=0.826784968 native units.

## G. What is the computational cost?

- B=1 `explicit_invariant_mm`: median 0.352256 ms, p95 0.442396787 ms, 9 profiled device launches.
- B=1 `persistent_invariant_bmm`: median 0.0708960034 ms, p95 0.0770255979 ms, 1 profiled device launches.
- B=2 `explicit_invariant_mm`: median 0.659248024 ms, p95 0.674624011 ms, 17 profiled device launches.
- B=2 `persistent_invariant_bmm`: median 0.0719359964 ms, p95 0.0786752027 ms, 1 profiled device launches.
- B=4 `explicit_invariant_mm`: median 1.27337605 ms, p95 1.28275037 ms, 33 profiled device launches.
- B=4 `persistent_invariant_bmm`: median 0.0719999969 ms, p95 0.078536002 ms, 1 profiled device launches.
- B=8 `explicit_invariant_mm`: median 2.50150394 ms, p95 2.51059679 ms, 65 profiled device launches.
- B=8 `persistent_invariant_bmm`: median 0.0723359995 ms, p95 0.0794272013 ms, 1 profiled device launches.
- B=16 `explicit_invariant_mm`: median 4.95697618 ms, p95 4.97467055 ms, 129 profiled device launches.
- B=16 `persistent_invariant_bmm`: median 0.111920003 ms, p95 0.117095998 ms, 1 profiled device launches.
- B=32 `explicit_invariant_mm`: median 9.91953564 ms, p95 11.3564837 ms, 258 profiled device launches.
- B=32 `persistent_invariant_bmm`: median 0.192400001 ms, p95 0.213707203 ms, 1 profiled device launches.
Persistent-vs-explicit invariant kernel differences:
- PV B=1: persistent median latency is 4.96862987x faster (0.281359997 ms lower), throughput is 4.96862987x, and launches are reduced by 8 (9 to 1).
- PV B=2: persistent median latency is 9.16436912x faster (0.587312028 ms lower), throughput is 9.16436912x, and launches are reduced by 16 (17 to 1).
- PV B=4: persistent median latency is 17.6857792x faster (1.20137605 ms lower), throughput is 17.6857792x, and launches are reduced by 32 (33 to 1).
- PV B=8: persistent median latency is 34.5817292x faster (2.42916794 ms lower), throughput is 34.5817292x, and launches are reduced by 64 (65 to 1).
- PV B=16: persistent median latency is 44.2903507x faster (4.84505617 ms lower), throughput is 44.2903507x, and launches are reduced by 128 (129 to 1).
- PV B=32: persistent median latency is 51.5568378x faster (9.72713564 ms lower), throughput is 51.5568378x, and launches are reduced by 257 (258 to 1).
- `native_dynamic`: median 371.636594 ms, p95 381.204572 ms, 10.7632027 requests/s, peak 14578105856 bytes, launches 13234.
- `invariant_dynamic`: median 958.731715 ms, p95 959.763234 ms, 4.17217866 requests/s, peak 14578105856 bytes, launches 11835.
- `serial_singleton`: median 943.998932 ms, p95 977.700788 ms, 4.23729293 requests/s, peak 14342974464 bytes, launches 48956.
- `fixed_size_padded`: median 528.893172 ms, p95 539.357372 ms, 7.56296397 requests/s, peak 14885527552 bytes, launches 13133.
Complete-policy invariant-vs-native dynamic differences at logical B=4: median latency +587.095121 ms (2.57975595x native), throughput -6.59102405 requests/s (0.387633567x), peak allocation +0 bytes, and profiled launches -1399 (13234 to 11835).

## H. How does invariant batching compare with practical alternatives?

- `fixed_size_padded`: 5 offered-load points; numerical contract satisfied at 5/5 points. load 1: 1.02644042 req/s, p95 962.871138 ms; load 4: 3.83091186 req/s, p95 1003.15271 ms; load 8: 7.87350179 req/s, p95 1019.83688 ms; load 12: 9.42261338 req/s, p95 992.025745 ms; load 20: 14.4574829 req/s, p95 3034.23986 ms
- `invariant_dynamic`: 5 offered-load points; numerical contract satisfied at 5/5 points. load 1: 1.02406754 req/s, p95 1556.10944 ms; load 4: 3.80957091 req/s, p95 1918.94896 ms; load 8: 6.07893351 req/s, p95 9165.78916 ms; load 12: 6.16108414 req/s, p95 11576.3892 ms; load 20: 6.15285527 req/s, p95 21368.9617 ms
- `native_dynamic`: 5 offered-load points; numerical contract satisfied at 0/5 points. load 1: 1.02795608 req/s, p95 398.40406 ms; load 4: 3.87315422 req/s, p95 517.455813 ms; load 8: 8.03554515 req/s, p95 700.337085 ms; load 12: 9.59801533 req/s, p95 781.104568 ms; load 20: 15.2955505 req/s, p95 2580.07768 ms
- `serial_singleton`: 5 offered-load points; numerical contract satisfied at 5/5 points. load 1: 1.02802002 req/s, p95 424.294189 ms; load 4: 3.87315493 req/s, p95 1712.16368 ms; load 8: 4.48759425 req/s, p95 19723.9973 ms; load 12: 4.35116976 req/s, p95 23861.0127 ms; load 20: 4.48711509 req/s, p95 32425.7847 ms

## Operator qualification

Retained 72000 operator records. The full-invariant path had 0 batch-invariance failures across 36000 records.
Dispatcher evidence covers: `rank2_mm`→`aten::mm` (registered_invariant_override), `rank3_attention`→`aten::bmm` (registered_invariant_override), `siglip_projection`→`aten::convolution` (registered_invariant_override), `rmsnorm_mean`→`aten::mean` (registered_invariant_override), `qualified_log_softmax`→`aten::log_softmax` (registered_invariant_override), `attention_softmax`→`aten::softmax` (audited_native_batch_local).
