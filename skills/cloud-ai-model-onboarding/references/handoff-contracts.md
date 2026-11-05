# Model Onboarding Handoff Contracts

This reference contains only invariants that cross skill ownership boundaries.
The model-onboarding `SKILL.md` owns order and gates; the linked owner skills own
the detailed procedures.

## Workspace and source contract

- All artifacts live below the model-specific workspace directory.
- Before the source lock, inspect any model-family variants. When no variant is
  specified, select a transparent, target-appropriate baseline from the
  source-supported choices and record the available variants, selected variant,
  and variant-selection rationale in the workspace provenance and model card.
  Do not select the smallest variant solely for faster export or compilation.
  Once selected, the variant is part of the source/checkpoint contract.
- The selected model source, framework, checkpoint, source revision, and task
  remain fixed through onboarding. A source change requires user approval.
- The Dockerfile establishes the cloned community source before the export script
  imports it. Architecture reimplementation or inline source copies are invalid.
- Export and QAIC SDK containers are separate; only the host model workspace is
  shared.

Owner: `cloud-ai-env-setup` for device/container setup and
`cloud-ai-model-conversion/references/exporting-models.md` for source/export
provenance.

## Stage-count contract

The Step 1 source inventory is authoritative. For N stages:

1. Export N role-suffixed ONNX files and record role, source path, and filename.
2. Compile N QPC directories independently using
   `qpc_<role>_<cores>c_mos<N>/` naming.
3. Confirm exactly N `programqpc.bin` files before accuracy validation.
4. Bind every role's QPC in `predict.py`; never silently omit a stage.

If the model uses SDP, MDP, or tensor-slice (TS) partitioning, its partition
configuration and inter-device precision choices are part of the handoff. Route
syntax to `cloud-ai-qaic-compile`; validate the complete multi-device workload.

Slice exports for deep transformers are debug accelerators only. Full-model export,
compilation, accuracy, and performance remain mandatory.

Owner: `cloud-ai-model-conversion` for export and compile procedures;
`cloud-ai-vision-demo/references/template-routing-and-integration.md` for
multi-stage runtime integration.

The source inventory must consider multiple role-named artifacts, distinct
model-loading paths, and README/inference-pipeline composition when deciding N.
Use source inspection to find separate loading paths, for example:

```bash
grep -rE "(from_pretrained|torch\.load|InferenceSession)" src/ \
  | grep -v "#"
```

Two or more calls using different model paths indicate multiple stages.

## Iterative-stage boundary

For diffusion/flow-matching sampling, autoregressive or encoder-decoder decode,
and other host-driven loops, separate:

- loop-carried inputs: evolving latent/hidden state, step/time, or token position;
- loop-invariant inputs: token/reference/conditioning embeddings, valid-length
  masks, positional/RoPE tables, and classifier-free-guidance conditioning.

Compute loop-invariant inputs once and pass them to the repeated QPC. The QPC
contains only the per-iteration update rather than the complete model `forward()`.

## Conversion and accuracy provenance

`conversion_manifest.json` must tie together the final ONNX revision, framework
golden outputs, ONNX Runtime outputs, `cpu_output.raw`, real input bundle, QPC,
compile log, and selected output names/shapes. Shapes match before numerical
comparison. Every output tensor is judged independently.

The framework-to-ONNX fidelity gate is cosine similarity >= 0.9999 per output;
the QAIC FP16 and graph-optimization gates use the thresholds defined by
`cloud-ai-accuracy-tuning` and the optimization workflow. Do not mix artifacts
from different export or transform revisions, and never label a failed tensor a
known limitation.

Owner: `cloud-ai-model-conversion/references/validation-and-manifest.md` and
`cloud-ai-accuracy-tuning`.

## Precision and production-QPC contract

FP16 is the clean baseline. INT8/PGQ requires explicit user authorization and a
passing FP16 baseline. After every compile, the onboarding record proves:

- `Compiling model with FP32 precision` is absent from `compile.log`;
- `programqpc.bin` exists for every stage; and
- `/opt/qti-aic/tools/qaic-qpc extract` confirms the intended IO precision.

FP16 underflow warnings require the export pipeline's `FP16ClipTransform`, a new
ONNX export, and a recompile. Diagnostic/statistics QPCs are never substituted
for production benchmark QPCs. The selected production QPC is the one whose
manifest, accuracy result, performance result, and stability result all match.

Owner: `cloud-ai-model-conversion/references/compiling-to-qpc.md`,
`cloud-ai-qaic-compile`, `cloud-ai-accuracy-tuning`, and
`cloud-ai-performance-optimization`.

## Demo path and fixture contract

The demo uses paths relative to the script/workspace and binds the selected
production QPC, not an arbitrary compile directory. It copies the approved
template, keeps `predict_base.py` unchanged, and discovers QPC IO names/shapes
from validated artifacts.

Acceptance requires a real task-representative fixture. Random `input.raw`, a
generic fixture, empty detections, or a synthetic image cannot pass detection,
two-stage, OCR, or known-answer validation. The one-line smoke-test JSON is the
model-card Usage Examples source.

Owner: `cloud-ai-vision-demo` and its smoke-test and template-integration
references.

## Model-card provenance contract

`MODEL_CARD.md` is copied from `MODEL_CARD_TEMPLATE.md` and records the source,
revision, task, precision, every measured batch, selected QPC, accuracy per
output, benchmark conditions, stability result, compile/profile evidence, graph
surgery, unsupported/custom ops, candidate base transforms, artifact paths, and
real smoke-test JSON. It is the final deliverable, not an optional summary.

Benchmark rows preserve their producer and units: use configurator `results.csv`
throughput directly, and retain both direct-runner QPC executions/sec and the
compiled-batch-normalized completed inference units/sec.

Owner: `MODEL_CARD_TEMPLATE.md`, with evidence supplied by every preceding step.
