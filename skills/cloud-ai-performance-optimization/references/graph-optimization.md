# Performance Graph Optimization and Tuning

Use this reference after the baseline/configurator evidence identifies a graph,
memory, or scheduling opportunity. Do not apply transforms speculatively. The
main skill owns the workflow gate; `cloud-ai-model-conversion` owns transform
implementation; `cloud-ai-accuracy-tuning` owns numerical acceptance.

## Pre-compilation checks

Before flag tuning, inspect the export and ONNX graph. Fix structural problems
first:

- Prefer Conv/FC/projection output channels in multiples of 32 (at least 8),
  keep Conv input depth at least 32 where possible, avoid channel counts that
  are not divisible by 8, and keep grouped channels-per-group a multiple of 32.
- BatchNorm should be fused into Conv; Cast/Quantize nodes should not break a
  linear-to-activation fusion; in-place updates and unnecessary Transposes should
  not prevent buffering or multicast.
- Avoid unnecessarily wide fan-out. Many parallel branches consume synchronization
  resources; consolidate common pre-branch work when it preserves semantics.
- Shape/axis/index inputs that control graph structure must be constant; symbolic
  dimensions must be bound with `-onnx-define-symbol`.
- Runtime `NonZero`, `Loop`, and `If` on hot paths require an explicit static or
  supported replacement where possible.
- Check ONNX operator support before graph surgery. `Einsum`,
  `ScatterElements`, and pooling `ceil_mode` each have backend-specific
  constraints; use the operator-support and graph-surgery references for the
  exact supported forms. Use a supported ONNX op or numerically equivalent
  decomposition before considering a custom NSP kernel.
- Inspect dilation, stride/padding, pooling, and Resize constraints for HMX
  eligibility and SDK support. For 2D Conv, HMX requires equal x/y dilation
  values and dilation <= 2; when dilation > 1, stride must be 1. With
  non-zero padding, keep stride <= kernel size; restrict stride > kernel size
  to zero-padding cases. Interleave/deinterleave helpers do not prove that a
  dilated Conv reached HMX; check for `aicconvhvx` in top ops.
- Budget per-core weight tiles, active/residual tensors, embedding tables, and
  retained state for VTCM. Use `-load-embedding-tables` for externalized large
  tables and `-retained-state` for persistent state where applicable. A tensor
  slice can fit even when the complete tensor cannot; use the actual working set
  and spill evidence.

## Tuning order

Run each candidate from a fresh production compile and compare against the same
input, core count, batch, instance count, and stability conditions.

The compiler reference owns exact syntax and valid ranges for the complete tuning
surface, including `-mdts-mos`, `-vtcm-working-set-limit-ratio`,
`-size-split-granularity`, `-allocator-dealloc-delay`,
`-aic-enable-depth-first`, `-aic-depth-first-mem`, `-multicast-weights`,
`-allow-mxint8-mdp-io`, `-mos`, and `-ols`. This reference only supplies the
conditions under which those controls are worth testing.

1. **MOS:** start with the compiler default (omit `-mos`), then `-mos=1`, then
   higher values only when runtime DMA and compile-log DDR evidence show a weight
   or tiling problem. The default is a required data point.
2. **OLS:** test only when D32 or size-split behavior makes it relevant. It raises
   VTCM use and must not be paired with depth-first, which forces OLS to 1.
3. **Depth-first:** use only for activation-spill evidence; test the documented
   8 MB and 32 MB budgets separately with
   `-aic-enable-depth-first -aic-depth-first-mem=8` and then
   `-aic-enable-depth-first -aic-depth-first-mem=32`. Require runtime
   `DMAIssue` above the diagnostic threshold, activation-spill evidence, and
   `constant only` DDR traffic below roughly 50%; otherwise investigate weight
   placement or compute/synchronization causes first. Depth-first forces OLS to 1.
4. **Multicast:** verify whether multicast is already active before adding the
   flag; use it for shared weights, not as a generic performance switch.
5. **MXFP6 MatMul:** use only for MatMul-heavy, weight-DDR-bound models. Recheck
   accuracy and compare DDR traffic; in multi-QPC pipelines, validate both the
   stage and the complete pipeline. A stage may keep the flag only when every
   stage output passes its per-tensor threshold and the complete pipeline still
   reproduces the accepted output.
6. **SDP/MDP:** use the `cloud-ai-qaic-compile` reference for partition syntax and
   compare the full end-to-end workload, not one isolated QPC.

Skip any candidate whose condition is not met. Do not recommend an undocumented
flag or infer a bottleneck from one aggregate counter.

## Graph candidate classification

Start the CPU-execution audit with the export script (graph truncation or wrapper
modules), compiler output (unsupported-op or partition messages), and prior model
card limitations. Verify each prior unsupported/High classification against the
current ONNX operator-support table.

For every candidate, ask whether it targets a named source-level module, method,
or weight tensor. If yes, use a `BasePytorchTransform` in a copied export script
and re-export. If no, use a `BaseOnnxTransform` against the final ONNX. Check
existing transforms with:

```bash
python base_pytorch_transform.py --list
python base_onnx_transform.py --list
```

Prototype ONNX-only changes in `optimize_graph.py` against the Step 1
`<model>_fp16ready.onnx`, writing `<model>_v2_fp16ready.onnx`. The input is already
simplified and FP16-clipped, so the prototype must not add `OnnxSimTransform` or
`FP16ClipTransform`. Record the per-op node-count delta from `run_pipeline()`.

For PyTorch-level changes, copy the export script to `export_<model>_v2.py` and
insert transforms at their required lifecycle point: before construction, after
model loading, or before `torch.onnx.export`. Re-export to regenerate the final
ONNX and CPU reference outputs.

Consolidate accepted changes into the self-contained `_v2.py` production export
script in the order: correctness fixes, optimization transforms, then
`FP16ClipTransform`. Do not import transforms from `optimize_graph.py`.

Keep model-specific transforms in the model workspace. Mark a transform for later
base-library promotion only when it targets a model class, contains no
model-specific constants, shapes, or node names, and no-ops when absent.

For a custom-op candidate, first confirm that the ONNX operator or an equivalent
supported decomposition cannot meet the measured requirement. Then use
`cloud-ai-custom-ops-kernels` for a measured fusion or specialized attention/
composite block, not as an automatic response to an unfamiliar graph.

For each candidate record the bottleneck, evidence/cycle share, why it is
inefficient on the NSP, the concrete ONNX- or PyTorch-level fix, estimated impact,
and effort. Implement observed Low and Medium candidates from highest impact to
lowest. Stop only for a measured rejection or genuine external blocker.

## Deployment and I/O checks

- Pre-load models to device DDR rather than reloading them for every request;
  this avoids repeating the model-download phase in request latency.
- Keep models that fit entirely in fast memory there when practical to eliminate
  steady-state DDR access.
- Pass contiguous inputs (`np.ascontiguousarray`) to QAIC and match input layout
  and dtype exactly (for example, NCHW versus NHWC and RGB versus BGR).

## Validation and stability

After every graph or precision change:

1. Run the accuracy skill's per-output comparison against matched artifacts.
2. Recompile the production QPC with the same workload configuration.
3. Re-run the relevant latency or throughput measurement and compare the selected
   metric plus headroom.
4. Run the 60-second stability test without profiling flags.
5. Keep only changes that pass accuracy, performance, and stability; revert the
   rest and record the reason.

Validate an ONNX-only prototype against `<model>_v2_fp16ready.onnx`. For a
PyTorch-level or consolidated candidate, rerun `export_<model>_v2.py` and validate
its regenerated `model_fp16ready.onnx`. Recompile with the same production flags
and workload configuration used for the baseline.

Run latency and throughput winners separately when their configurator rows differ.
For throughput, continue batch-specific exports when the winner is at the top of
the searched batch range until the documented plateau or latency budget stop
signal occurs. Use the batch-size reference for the exact configurator commands.

## INT8 boundary

Enter INT8 only after explicit user authorization. If validated FP16 already has
the required headroom and no memory-pressure constraint, record the decision and
skip calibration. Otherwise use `cloud-ai-accuracy-tuning` for PGQ and return here
only to benchmark the accuracy-approved INT8 QPC.
