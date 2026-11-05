---
name: cloud-ai-model-conversion
description: Use when exporting a CNN, vision, text, or audio / ASR model to ONNX, or compiling ONNX or TensorFlow source to a Cloud AI 100 QPC with `qaic-compile`. Not for LLMs, VLMs, diffusion, embeddings, or rerankers.
---

# Cloud AI 100 Model Conversion and Compilation

## Overview

This skill produces reproducible ONNX and Cloud AI 100 QPC artifacts for
supported models, or compiles supported TensorFlow GraphDef/SavedModel sources
directly to QPC. It covers export, graph preparation, compiler inputs, baseline
compilation, and sanity validation.

## When to Use

Use this skill when the task involves:

- Exporting a PyTorch, TensorFlow, Hugging Face, Ultralytics, Detectron2, or
  community-repository model to ONNX.
- Repairing symbolic shapes, dynamic shape inputs, unsupported operators, or
  framework export behavior.
- Compiling prepared ONNX, frozen TensorFlow GraphDef, or TensorFlow SavedModel
  source to a Cloud AI 100 QPC with `qaic-compile`.
- Preparing compiler input manifests, PGQ calibration inputs, or conversion
  provenance artifacts.
- Producing a baseline FP16 QPC before accuracy or performance tuning.

## When Not to Use

- **Generative LLM, VLM, diffusion, embedding, or reranker onboarding:** use
  `cloud-ai-llm-onboarding`.
- **Accuracy diagnosis or FP16/INT8 policy:** use `cloud-ai-accuracy-tuning`.
- **Throughput, latency, compiler-flag, or opstats tuning:** use
  `cloud-ai-performance-optimization`.
- **Specific `qaic-compile` flags:** use `cloud-ai-qaic-compile`.
- **Custom C++/HVX kernels:** use `cloud-ai-custom-ops-kernels`.
- **Host-side QAIC runtime code:** use `cloud-ai-qaic-cpp` or
  `cloud-ai-qaic-python`.

## Non-Negotiable Rules

- Keep the user-selected model source and framework fixed throughout export.
  Never silently substitute a different implementation because it exports more
  easily.
- On the ONNX route, preserve a symbolic `batch_size` axis in the raw export.
  Resolve it at compilation with `-onnx-define-symbol`; do not bake batch 1
  into reusable transforms.
- On the ONNX route, make non-batch dimensions concrete or deliberately named
  for compile-time resolution. Shape, axis, index, and reduction inputs that
  control graph structure must be constant initializers.
- If a compiled dimension is an upper bound for real input length, pass and
  consume `input_lengths` or `attention_mask`. Do not treat padded positions as
  valid content.
- On the ONNX route, put every graph modification in the export script's
  transform pipeline. Do not patch the raw ONNX manually between export and
  `run_pipeline()`.
- For the ONNX route, generate the framework golden after pre-export
  transforms and generate `cpu_output.raw` from the final `_fp16ready.onnx`
  passed to `qaic-compile`. Do not compile until the ONNX fidelity gate passes.
- For direct TensorFlow compilation, keep the original TensorFlow framework
  golden and use the TensorFlow source as the CPU reference; do not invent an
  ONNX artifact only to satisfy the ONNX route.
- Always include `FP16ClipTransform` after model-specific transforms on the
  ONNX route.
- Compile into a fresh `-aic-binary-dir`; verify the generated binary and scan
  the compile log for accidental FP32 compilation.
- Write `conversion_manifest.json` so the source format, inputs, QPC, compile
  command, log, output names, and precision settings can be matched later.

## Route to the Right Reference

| Need | Read |
|---|---|
| Choose an export source, export a decoder, use slice mode, trim outputs, or build the export pipeline | [Exporting models](references/exporting-models.md) |
| Resolve symbolic shapes, constant shape inputs, padded lengths, or attention masks | [QAIC model requirements](references/qaic-model-requirements.md) |
| Simplify ONNX, replace unsupported ops, patch framework code, or audit graph operators | [ONNX graph surgery](references/onnx-graph-surgery.md) |
| Compile ONNX or TensorFlow to QPC and validate compiler output | [Compiling to QPC](references/compiling-to-qpc.md) |
| Prepare RAW/JSON inputs or PGQ calibration data | [Input and calibration data](references/input-and-calibration-data.md) |
| Run CPU/QPC sanity checks, enforce fidelity, or write the conversion manifest | [Validation and manifest](references/validation-and-manifest.md) |

## End-to-End Workflow

1. Select the source and choose either the direct TensorFlow compilation path
   or the ONNX conversion path. If direct TensorFlow compilation fails, use the
   TF-to-ONNX fallback. Preserve a symbolic batch axis on ONNX exports.
2. On the ONNX route, apply the shape, masking, and constant-input
   requirements before graph surgery. On the direct TensorFlow route, verify
   concrete input shapes and selected output names in the source graph.
3. On the ONNX route, run the export script's transform pipeline and verify
   final graph operators. On the direct TensorFlow route, inspect the graph or
   SavedModel signature before compiling.
4. Prepare compiler inputs, compile the final `_fp16ready.onnx` or the direct
   TensorFlow source, and inspect precision, binary, spill, and warning signals.
5. Run CPU and QPC sanity checks, enforce the route-specific fidelity gate,
   and write the conversion manifest.
6. Hand off to accuracy tuning or performance optimization only after the
   baseline artifacts are internally consistent.

## Completion Criteria

A conversion is ready for downstream work when:

- For the ONNX route, the export script records the source, transforms,
  framework golden, and final `_fp16ready.onnx`; the final ONNX passes its
  framework-to-ORT fidelity gate and has the expected outputs and compile-time
  shape contracts.
- For the TensorFlow direct compilation branch, the source is a supported
  frozen GraphDef or SavedModel with concrete input shapes and explicit output
  names, and its framework golden is recorded.
- `qaic-compile` produces a QPC using the intended FP16 path and compiler inputs.
- CPU and QPC sanity outputs have matching shapes and acceptable basic
  comparison results.
- The QPC directory contains a complete conversion manifest.

## Common Mistakes

- On the ONNX route, baking `batch_size=1` into a reusable export or transform
  to silence an importer error.
- Adding an all-true mask to a padded variable-length input, or declaring a
  mask that the model never consumes.
- On the ONNX route, applying raw ONNX edits outside `run_pipeline()`.
- On the ONNX route, comparing QPC output with a framework golden from an
  unpatched model or with `cpu_output.raw` generated from the pre-transform
  ONNX.
- Reusing an existing QPC output directory or accepting a valid-looking FP32
  QPC because compilation returned a non-zero shell status.
- Treating a QPC execution smoke test as an accuracy acceptance decision.

## Related Skills

- `cloud-ai-model-onboarding`: end-to-end onboarding workflow around this
  conversion path.
- `cloud-ai-accuracy-tuning`: formal FP16/INT8 and output-fidelity validation.
- `cloud-ai-performance-optimization`: throughput, latency, and compiler tuning.
- `cloud-ai-qaic-compile`: detailed compiler flag and operator-support reference.
- `cloud-ai-custom-ops-kernels`: NSP custom operators when no supported ONNX
  replacement exists.
- `cloud-ai-qaic-runner` and `cloud-ai-qaic-python`: QPC execution alternatives.
