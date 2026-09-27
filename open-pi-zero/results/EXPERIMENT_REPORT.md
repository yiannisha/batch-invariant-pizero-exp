# Experimental report

This report is generated from the retained machine-readable measurements. Missing campaigns are marked unavailable rather than estimated.

## Provenance

- Policy revision: `a2576bbac02fca0ba1868843475f4a644835bdf1`
- Operator revision: `14dafb5fd84350dc796a96cbecb566ca6295593f`
- Checkpoint SHA-256: `1da40985d963ddf394fa63e7db5ddd4cd8082d5c6ce0b2a0cae56abd8b6ec0eb`
- Evaluation scope used below: **synthetic diagnostic**

## A. Does native inference violate request-level batch invariance?

Yes within the synthetic diagnostic campaign: 1/1 requests violated (100.000%), and 18/18 non-singleton arrangements violated (100.000%). The maximum normalized action error was 1.80006027e-05.
Tested transformations=['batch_position'], batch sizes=[2, 4, 8], target positions=['first', 'last', 'middle'], companions=['diverse', 'duplicate']. Across arrangement-level maximum errors: median=5.66244125e-07, p95=1.80006027e-05, p99=1.80006027e-05.
The required 2,400-request held-out result is unavailable because the replay dataset could not be collected in this container; this diagnostic result must not be read as a held-out rate.

## B. Where does native execution first diverge?

- `existing_invariant_ops`: prefill.step_0.action_joint_model.layer_0.post_attn.attn_outputs_final#invocation_0.vlm
- `explicit_per_matrix_attention`: prefill.step_0.action_joint_model.layer_0.post_attn.attn_outputs_final#invocation_0.vlm
- `full_invariant`: none (all traced tensors exact)
- `invariant_plus_patch_projection`: prefill.step_0.action_joint_model.layer_0.post_attn.attn_outputs_final#invocation_0.vlm
- `native`: proprio_embeds#invocation_0
- `native_deterministic`: proprio_embeds#invocation_0

## C. How do differences propagate through the flow solver?

- `native` step:error — 1:5.96046448e-08, 2:1.1920929e-07, 3:1.78813934e-07, 4:2.5331974e-07, 5:2.83122063e-07, 6:3.12924385e-07, 7:3.42726707e-07, 8:3.57627869e-07, 9:3.57627869e-07, 10:4.47034836e-07
- `existing_invariant_ops` step:error — 1:1.1920929e-07, 2:3.57627869e-07, 3:4.76837158e-07, 4:5.96046448e-07, 5:7.74860382e-07, 6:1.07288361e-06, 7:1.13248825e-06, 8:1.1920929e-06, 9:1.60932541e-06, 10:2.27987766e-06
- `invariant_plus_patch_projection` step:error — 1:1.1920929e-07, 2:3.57627869e-07, 3:4.76837158e-07, 4:5.96046448e-07, 5:7.74860382e-07, 6:1.07288361e-06, 7:1.13248825e-06, 8:1.1920929e-06, 9:1.60932541e-06, 10:2.27987766e-06
- `full_invariant` step:error — 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0

## D. Does the full invariant path achieve exact action equality?

In the synthetic diagnostic records, 0/18 non-singleton arrangements failed exact equality; maximum error was 0.

## E. How faithful is invariant singleton inference to native singleton inference?

For the frozen synthetic diagnostic singleton: exact=False, max absolute error=1.4603138e-06, RMSE=5.78366959e-07, relative L2=1.47285234e-06. This fidelity comparison is distinct from batch invariance.

## F. Do numerical differences affect behavior?

- Action divergence: measured in the synthetic diagnostic numerical campaign; 18/18 native arrangements differed.
- Trajectory divergence: unavailable; no SIMPLER episode could reach `env.reset`.
- Paired success disagreement: unavailable; 0/800 planned episodes were executed.
- Success-rate differences: unavailable; no behavioral outcome is inferred from the numerical results.
The container exposed CUDA compute but not the NVIDIA graphics/Vulkan ICD required by SAPIEN 2.2.2; native rendering segfaulted and the Lavapipe fallback lacked a required Vulkan extension. Full evidence is retained in `results/simpler/blocker.json`.

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
- `native_dynamic`: median 371.636594 ms, p95 381.204572 ms, 10.7632027 requests/s, peak 14578105856 bytes, launches 13234.
- `invariant_dynamic`: median 958.731715 ms, p95 959.763234 ms, 4.17217866 requests/s, peak 14578105856 bytes, launches 11835.
- `serial_singleton`: median 943.998932 ms, p95 977.700788 ms, 4.23729293 requests/s, peak 14342974464 bytes, launches 48956.
- `fixed_size_padded`: median 528.893172 ms, p95 539.357372 ms, 7.56296397 requests/s, peak 14885527552 bytes, launches 13133.

## H. How does invariant batching compare with practical alternatives?

- `fixed_size_padded`: 5 offered-load points; numerical contract satisfied at 5/5 points. load 1: 1.02644042 req/s, p95 962.871138 ms; load 4: 3.83091186 req/s, p95 1003.15271 ms; load 8: 7.87350179 req/s, p95 1019.83688 ms; load 12: 9.42261338 req/s, p95 992.025745 ms; load 20: 14.4574829 req/s, p95 3034.23986 ms
- `invariant_dynamic`: 5 offered-load points; numerical contract satisfied at 5/5 points. load 1: 1.02406754 req/s, p95 1556.10944 ms; load 4: 3.80957091 req/s, p95 1918.94896 ms; load 8: 6.07893351 req/s, p95 9165.78916 ms; load 12: 6.16108414 req/s, p95 11576.3892 ms; load 20: 6.15285527 req/s, p95 21368.9617 ms
- `native_dynamic`: 5 offered-load points; numerical contract satisfied at 0/5 points. load 1: 1.02795608 req/s, p95 398.40406 ms; load 4: 3.87315422 req/s, p95 517.455813 ms; load 8: 8.03554515 req/s, p95 700.337085 ms; load 12: 9.59801533 req/s, p95 781.104568 ms; load 20: 15.2955505 req/s, p95 2580.07768 ms
- `serial_singleton`: 5 offered-load points; numerical contract satisfied at 5/5 points. load 1: 1.02802002 req/s, p95 424.294189 ms; load 4: 3.87315493 req/s, p95 1712.16368 ms; load 8: 4.48759425 req/s, p95 19723.9973 ms; load 12: 4.35116976 req/s, p95 23861.0127 ms; load 20: 4.48711509 req/s, p95 32425.7847 ms

## Operator qualification

Retained 28800 operator records. The full-invariant path had 0 batch-invariance failures across 14400 records.
