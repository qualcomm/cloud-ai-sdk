---
name: cloud-ai-accuracy-tuning
description: Use when validating FP16 or INT8 output accuracy, debugging CPU-versus-QAIC differences, fixing mixed-precision policies, comparing stochastic outputs, or investigating accuracy changes across static shapes on Cloud AI 100.
---

# Cloud AI 100 Accuracy Tuning

## Overview

Use this skill to decide whether a prepared ONNX model and Cloud AI 100 QPC are
accurate enough to advance, and to choose the least invasive recovery path when
they are not. It covers ONNX fidelity, FP16, user-requested INT8, selective
FP16/FP32 mixed precision inside INT8 graphs,
stochastic and discrete outputs, static-shape MOS effects, and narrowly justified
CPU boundaries.

Use `cloud-ai-model-conversion` for export, graph repair, and baseline QPC
compilation. Use `cloud-ai-performance-optimization` for compiler sweeps,
benchmarking, and opstats. This skill owns the accuracy gates between those steps.

## Non-negotiable accuracy contract

- Prove framework-to-ONNX Runtime fidelity before comparing any QPC.
- Compare only matching artifacts: framework, ORT, QAIC outputs, real input
  bundle, QPC, and conversion manifest must identify the same final ONNX revision.
- Compare shapes before numbers. Judge every output independently; one failed
  output fails the model even when an aggregate score passes.
- Never dismiss an incorrect output as a "known limitation."
- FP16 is the default. Start INT8 only when the user explicitly requests INT8 or
  higher-throughput quantization work; preserve the validated FP16 baseline.
- After every graph, precision, profile, MOS, or boundary change, revalidate with
  the same artifact and input contract.
- Report failed experiments and retain the last validated production artifact.

## Decision guide

| Evidence | Required action |
|---|---|
| Framework and final ONNX differ | Stop before QPC diagnosis; repair export using [ONNX baseline validation](references/onnx-baseline-validation.md). |
| Artifacts or manifests identify different graph/input revisions | Regenerate the mismatched artifacts; do not interpret similarity metrics. |
| Any FP16 output is below cosine 0.99 | Fail FP16, isolate that output, and apply the least invasive proven fix. |
| Compiler reports FP16 weight underflow | Apply `FP16ClipTransform`, export a new final ONNX, recompile, and repeat all gates before INT8. |
| One static shape fails while nearby shapes pass | Rule out a precision tie, then run the one-variable [MOS control](references/output-comparison.md#static-shape-mos-control). |
| Sampled outputs differ | Fix seed, shape, and upstream-tensor controls using [stochastic comparison](references/output-comparison.md#stochastic-outputs). |
| Argmax/token/box order differs on near-equal scores | Use [ULP-aware discrete comparison](references/output-comparison.md#near-tied-discrete-outputs); a tie is not an accuracy pass. |
| User requests INT8 | Run [PGQ viability and calibration](references/profile-guided-quantization.md); do not skip the FP16 baseline. |
| INT8 dataset accuracy drops more than 2pp or misses accuracy on a small sensitive region | Use `fp16sweep`, then the [mixed-precision workflow](references/profile-guided-quantization.md#mixed-precision-override-workflow). |
| PGQ and mixed precision are insufficient | Use [AIMET](references/aimet.md); CLE is the documented data-free fallback when calibration data is unavailable. |
| A deterministic, bounded, non-learned boundary is proven | Consider [narrow CPU offload](references/output-comparison.md#narrow-cpu-offload-evidence-gate) only after every evidence gate passes. |

Model type suggests likely effort, not permission to skip gates. Classification
CNNs often stop at PGQ; detection and segmentation commonly need mixed precision;
transformer, OCR, and attention models may need AIMET. INT8 can provide substantial
throughput gains, including 2–3× on suitable models, but measured accuracy and
throughput decide whether to ship it.

## Ordered workflow

### Validate artifact identity and ONNX fidelity

Read [ONNX baseline validation](references/onnx-baseline-validation.md) and run
both its numerical and input-pipeline checks.

Requirements:

- use the exact final ONNX passed to `qaic-compile`;
- generate framework goldens after every pre-export framework transform;
- use real representative inputs and the matching conversion manifest;
- require every output shape to match before numerical comparison;
- require framework-to-ORT cosine `>= 0.9999`; and
- require max absolute error `< 1e-3`. If cosine passes but this error fails,
  treat it as an outlier and apply `FP16ClipTransform`; if cosine also fails,
  troubleshoot the export.

If the original framework is unavailable, use the reference's reduced-confidence
comparison between the pre-simplification and final ONNX and record that limitation.
Do not proceed until both numerical fidelity and preprocessing sanity pass.

### Validate the FP16 QPC

Compile the pure FP16 baseline using the
[model-conversion compile procedure](../cloud-ai-model-conversion/references/compiling-to-qpc.md),
then run the same real inputs through CPU ONNX Runtime and the QPC using
`qaic.Session` or `qaic-runner`. For dataset-scale top-1/top-5, mAP, or mIoU,
use `qaic-acc-evaluator.py -platform-tag aic_fp16` from
[qaic-pytools](references/qaic-pytools.md).

For every output:

1. Verify name, shape, and dtype.
2. Calculate cosine similarity, MSE, and max absolute error.
3. Require cosine `>= 0.99`. MSE is secondary because output scale changes its
   meaning.
4. Mark the complete model FAIL if any output fails.

Before changing the graph, rule out layout, dtype, NCHW/NHWC, RGB/BGR,
scale/mean/std, non-contiguous arrays, masks, valid lengths, cache/state tensors,
and model-defined shape formulas. Use `np.ascontiguousarray` where required.

Autoregressive models also require full token-for-token decode validation after
the per-output gate. A step-0 result with zero past-KV cannot prove cache wiring.
Follow [Autoregressive decode validation](references/output-comparison.md#autoregressive-decode-validation).

Do not start throughput acceptance or INT8 while FP16 fails.

### Recover FP16 with the least invasive proven change

If underflow is demonstrated, use the export pipeline's `FP16ClipTransform`;
never substitute an ad hoc clamp script. Produce a new final ONNX, matching
goldens and manifest, and a fresh QPC, then restart validation.
The normal and standalone/rescue procedures are in the
[model-conversion export pipeline](../cloud-ai-model-conversion/references/exporting-models.md#export-pipeline);
verify `FP16ClipTransform` remains after all model-specific transforms.

For other failures:

1. Treat the first low-cosine tensor as a boundary, not a guilty operation.
2. Compare immediate inputs and outputs.
3. Use `qaic-acc-analyzer.py -snooping layerwise` or sparse intermediate outputs.
4. Follow [cost-aware semantic bisection](references/output-comparison.md#cost-aware-semantic-bisection)
   with captured failing tensors and one hypothesis per compile.
5. Promote only a proven precision override, reformulation, export repair, or
   custom on-device op through the normal export path and full-graph gate.

Read `cloud-ai-architecture` before choosing a hardware-sensitive remedy.
Preserve HMX/HVX fusion where possible and validate placement with the compiler
precision audit, `AIC non-HMX MAC estimate`, and opstats.

Whole-graph FP32 is not a valid target, diagnostic, or fallback on HMX. Do not
wait for a long-running whole-network FP32 compile.

If required outputs are absent or mismatched, correct `graph.output` before
compilation; numerical comparison cannot validate tensors the graph did not export.

### Run INT8 only with explicit authorization

Read [PGQ and mixed precision](references/profile-guided-quantization.md) before
starting. It owns calibration sourcing and splits, `IO-files` JSON, two-pass PGQ
commands, viability opstats, schema sweeps, precision policy validation, and
external profiles.

The required order is:

1. Preserve the validated FP16 accuracy and throughput baselines.
2. Use 10-20 representative domain images for the viability check.
3. Compile one INT8 QPC, capture fallback warnings, benchmark, and compare FP16
   and INT8 opstats.
4. When an op is over `1.3×` slower and shows mixed types, pin from the nearest
   upstream precision boundary through the op. Never pin only the regressed op.
5. If throughput remains `<=` FP16 after grouped pinning, stop and ship FP16.
6. Run a separate INT8 performance sweep. The production target is
   `>= 1.5×` FP16; a result above `1.0×` but below `1.5×` may continue only
   with the gap recorded.
7. Use `qaic-acc-evaluator.py` and `pgq_sweep` on 20 calibration and 20 test
   images to rank compiler-supported schemas. Never ship the 20-image profile.
8. Regenerate the profile from the full non-overlapping calibration split and
   evaluate every item in the held-out test split.

Calibration and test data never overlap. Use the model's evaluation domain, not
preprocessing similarity. The standard 1,000-image split is 500/500 for
classification and 750/250 for detection, segmentation, transformer, OCR,
attention, and other models.

Report INT8 results regardless of outcome. Use CPU FP32 as the formal reference;
after FP16 is validated, FP16 QPC outputs may isolate FP16-to-INT8 drift if the
model card records that choice.

### Recover INT8 with mixed precision

Try the [qaic-pytools `fp16sweep`](references/qaic-pytools.md#fp16sweep-workflow)
first. Its flag syntax is non-obvious; do not infer it from memory. If the graph
is unsupported or the result is insufficient, use the manual selection procedure
in the PGQ reference.

Every `node_precision.yaml` name must resolve to the intended first output in the
exact final ONNX. A missing name or wrapper-op match invalidates the policy.
Regenerate the PGQ profile after every policy change, recompile, benchmark first,
then validate accuracy.

| Model type | Logits/scores cosine | Boxes/spatial cosine |
|---|---:|---:|
| Classification CNN | `>= 0.99` | Not applicable |
| Detection or segmentation | `>= 0.97` | `>= 0.95` |
| Transformer or attention | `>= 0.97` | `>= 0.95` |

These thresholds correspond to no more than approximately 1–2pp mAP degradation
for detection and 1pp top-1 degradation for classification.

Use the minimum FP16 node set that passes:

| Result | Action |
|---|---|
| Accuracy passes and throughput `>= 1.5×` FP16 | Accept the QPC. |
| Accuracy passes and throughput `< 1.5×` FP16 | Reduce FP16 pins and rebenchmark. |
| Accuracy fails and throughput `>= 1.5×` FP16 | Add a proven sensitive group and revalidate. |
| Accuracy fails and throughput `< 1.5×` FP16 | Escalate to AIMET. |

For AIMET, read [AIMET](references/aimet.md) for CLE, AdaRound, sensitivity/AMP,
and conversion to the v0.5.0 YAML required by
`qaic-compile -external-quantization`. If external encodings omit a tensor, the
compiler falls back to FP16; report and correct or explicitly accept that fallback.

## Edge-case acceptance rules

- **Static shape:** change only `-mos` and the fresh binary directory in the
  control compile. A passing MOS is part of that shape's contract only.
- **Stochastic output:** require the same immediate seed, static shape, and
  captured upstream tensors. Otherwise use deterministic boundaries and a
  distributional or task metric.
- **Near-tied discrete output:** record first-mismatch scores and margin against
  the format ULP at that magnitude. A sub-ULP tie is neither a pass nor a defect;
  rerun the complete sequence with a tie-tolerant metric.
- **CPU offload:** allow only deterministic, bounded, non-learned boundary work
  after framework-boundary, captured-suffix, end-to-end, and bridge-cost gates.
  Never offload learned Conv, MatMul, attention, recurrent, decoder, generator,
  or residual blocks to bypass failed accuracy.

The complete procedures and required evidence are in
[Output comparison and FP16 localization](references/output-comparison.md).

## Completion evidence

Record:

- hashes or manifest identity for final ONNX, real inputs, and QPC;
- per-output names, shapes, dtypes, cosine, MSE, max absolute error, and status;
- full autoregressive or tie-tolerant sequence result when applicable;
- exact compile commands, precision policy, profile, MOS, and static shape;
- calibration and test source/split with proof of no overlap;
- FP16 and candidate throughput for every INT8/mixed-precision decision; and
- accepted artifact or rollback to the last validated FP16 QPC.

## Related skills

- `cloud-ai-model-conversion`: export, graph repair, `FP16ClipTransform`, final
  ONNX, conversion manifest, and baseline QPC compilation.
- `cloud-ai-performance-optimization`: core/batch/compiler sweeps, benchmarking,
  canonical profiling, and opstats.
- `cloud-ai-qaic-compile`: authoritative compiler flag semantics.
- `cloud-ai-qaic-runner` and `cloud-ai-qaic-python`: QPC execution and output
  capture.
- [qaic-pytools](references/qaic-pytools.md): dataset evaluation, layerwise
  analysis, `fp16sweep`, comparisons, histograms, and metric plugins.
- `cloud-ai-architecture`: HMX/HVX, precision placement, tiling, and fusion.
- `cloud-ai-custom-ops-kernels`: faithful custom on-device operations when no
  standard ONNX formulation can preserve the required semantics.
