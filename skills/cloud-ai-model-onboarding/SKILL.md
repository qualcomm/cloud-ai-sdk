---
name: cloud-ai-model-onboarding
description: Use when onboarding, converting, compiling, validating, optimizing, or demonstrating a CNN, vision, text/audio classification, TTS, or ASR model for Cloud AI 100 with qaic-compile and QPC artifacts. Not for LLMs, VLMs, diffusion, embeddings, or rerankers.
---

# Cloud AI 100 Model Onboarding

## Purpose and completion rule

This is the end-to-end contract for supported non-generative models. It owns
stage ordering, gates, artifact handoffs, and completion criteria; linked skills
own detailed export, compiler, accuracy, performance, demo, and hardware work.
Do not declare completion until `MODEL_CARD.md` exists and all five done criteria
pass. Use `cloud-ai-llm-onboarding` for LLMs, VLMs, diffusion, embeddings, and
rerankers; use `cloud-ai-model-conversion` alone for conversion-only work.

## Critical rules

- SDK tools exist only at `/opt/qti-aic/exec/` and `/opt/qti-aic/tools/`. Check
  them with `ls`; never use `which`, `find`, or host-wide searches.
- Write the Dockerfile before the export script. Clone community source in the
  Dockerfile; never inline or reimplement the architecture.
- Complete the model-family variant selection gate before fixing the Step 1
  source, framework, checkpoint, revision, and task. After that lock, if
  supported remediation is exhausted, report failure; never silently substitute.
- Create `<model_name>/` first. Keep every ONNX, QPC, log, profile, script,
  input, and `MODEL_CARD.md` below it.
- Keep the conversion manifest, final ONNX, framework/ORT outputs,
  `cpu_output.raw`, real input bundle, QPC, and compile evidence matched.
- FP16 is the baseline. INT8/PGQ is user-requested only.
- Never pass a failed gate, missing stage/QPC, output mismatch, or failed real
  fixture by calling it an approximation or known limitation.

## Handoff contract

Read [handoff-contracts.md](references/handoff-contracts.md) first. It defines
the cross-step stage-count, artifact, precision, production-QPC, demo-path, and
model-card provenance invariants.

## Step 0: Environment and workspace

Confirm device visibility and start the QAIC SDK container with
`cloud-ai-env-setup`. Export uses a separate short-lived `docker compose run
export` container; only the host model workspace is shared. Read
`cloud-ai-architecture` and `references/cloud_ai_soc_architecture.md`, and read
`references/transformer_architecture.md` before attention-model export. Create
`<model_name>/`, record source/framework/task/revision/device, detect multi-stage
or iterative loops, and read the model-card template plus owner skills. Do not
write the export script before the Dockerfile and source inventory.

## Step 0.5: Model-family variant selection

Before locking the source and checkpoint, determine whether the requested model
family has materially different capacity, resolution, backbone, or deployment
variants. When no variant is specified, inspect the source-supported choices
and select using the quality target, latency/throughput target, memory budget,
input-shape constraints, and target accelerator capacity. Do not choose the
smallest variant solely because it is quicker to export or compile. When the
requirements do not establish a preference, select and record a transparent,
target-appropriate baseline assumption.
Record the model family, available variants, selected variant, and selection
rationale in workspace provenance and the model card. For a single-variant
source, record that the gate is not applicable. The selection is part of the
source lock and cannot change without user approval.

## Step 1: Export ONNX and prove fidelity

Use `cloud-ai-model-conversion` and its [exporting models](../cloud-ai-model-conversion/references/exporting-models.md), [model requirements](../cloud-ai-model-conversion/references/qaic-model-requirements.md), [ONNX graph surgery](../cloud-ai-model-conversion/references/onnx-graph-surgery.md), [input and calibration data](../cloud-ai-model-conversion/references/input-and-calibration-data.md), and [validation and manifest](../cloud-ai-model-conversion/references/validation-and-manifest.md) references.

- Treat the source as multi-stage when it has multiple role-named
  `.onnx`/`.pt`/`.pth`/`.safetensors` artifacts, distinct model-loading paths,
  or README/inference-pipeline composition. Record role, source path, and
  `<model>_<role>.onnx` before writing the export script; export every stage.
- For diffusion/flow-matching, autoregressive, encoder-decoder, or other loops,
  export only the per-iteration state update. Compute embeddings, masks, and
  position tables once; pass them with loop-carried state into the repeated QPC.
- Use a real task-representative input and preserve `input.raw` for ONNX
  fidelity, QAIC accuracy, and throughput. A synthetic input can make failures
  invisible. Use `export_params=True`, `do_constant_folding=True`, and regenerate
  framework goldens after pre-export transforms when applicable.
- Padded variable-length/time dimensions require per-request `input_lengths` or
  `attention_mask`; an all-true mask or no mask is a silent padding bug. Read the
  transformer architecture reference before attention-model export.
- Check supported ONNX operators first, then use `BaseOnnxTransform`/
  `BasePytorchTransform`. Copy both libraries, use `torch.onnx.export` with
  `model.eval()` and the export pipeline's `run_pipeline()`, and never patch raw
  ONNX between export and the pipeline. A custom op also requires static shapes,
  deterministic control flow, and a suitable NSP profile; use it only after
  supported replacements are ruled out. Measured fusion and specialized
  attention/composite blocks remain valid custom-op use cases.

Do not advance to Step 2 until shapes match and framework-to-ONNX/ORT cosine is
>= 0.9999 for every output. Any output below 0.9999 is a hard failure. Required
artifacts are final ONNX per stage, framework/ORT outputs, `framework_output_0.raw`
when applicable, `cpu_output.raw`, real inputs, and `conversion_manifest.json`.

## Step 2: Compile QPCs and validate precision

Read `cloud-ai-qaic-compile` before any `qaic-compile` command; use [compiling to QPC](../cloud-ai-model-conversion/references/compiling-to-qpc.md) and [validation and manifest](../cloud-ai-model-conversion/references/validation-and-manifest.md).

- A 4-layer slice can shorten debugging for 12+ homogeneous transformer layers,
  but full-model export, compile, accuracy, and performance remain mandatory.
- For N recorded stages, produce N role-suffixed QPCs, sweep each independently,
  and confirm N `programqpc.bin` files. SDP/MDP/TS partitioning and
  `-mdts-mos`/inter-device precision decisions belong to `cloud-ai-qaic-compile`;
  validate the complete multi-device workload.
- Compile clean FP16 with `-convert-to-fp16` and `-aic-perf-metrics`. After every
  compile, scan for `Compiling model with FP32 precision`, confirm the binary,
  and use `/opt/qti-aic/tools/qaic-qpc extract` on
  `networkdesc.bin_dir/networkdesc.json`, checking every `io_transformed.type`.
  A FloatTy-only audit is FP32. Repeat for initial, core, MOS, and Step 5 builds.
- FP16 underflow requires `FP16ClipTransform`, a fresh ONNX export, and recompile.
  Unsupported-op order is existing transform, new transform subclass, then
  custom NSP op. Use `base_*_transform.py --list`; use `gen_custom_op_package.py`
  for new packages. `/opt/qti-aic/examples/apps/custom-op/` is structural
  guidance and `known-custom-op/` is interface reference only.

Do not advance with a missing binary/stage, FP32 QPC, or underflow warning. If
compile.log, support matrix, all applicable transforms, and custom-kernel paths
have been exhausted, stop with a Step 2 failure report listing reason, attempts,
and conclusion. Record QPC paths and compile logs in the manifest.

## Step 3: Validate per-output accuracy

Read `cloud-ai-accuracy-tuning` before comparison. Verify the manifest first;
QPC, final ONNX, framework/ORT outputs, and real inputs must match revisions.
Shapes must match before numeric comparison. Judge every output independently;
one failing tensor fails the model, and “known limitation” is not a valid excuse.
The QAIC FP16 gate is the accuracy skill's per-output threshold (baseline quick
reference: cosine >= 0.99). If a slice was used, validate slice then full model.
INT8/PGQ, mixed precision, stochastic/discrete outputs, and CPU offload follow
that skill's gates. Do not advance to Step 4 until accuracy passes.

## Step 4: Validate throughput, latency, and stability

Read `cloud-ai-performance-optimization`; record target, `NSP_TOTAL`, cores, batch, instances,
activation count, set size, warmup, duration, input, QID, and selected metric.
Prepare the benchmark ledger before selecting a winner: record producer,
compiled batch, activation count, raw QPC executions/sec when measured by direct
`qaic-runner`, and completed inference units/sec. Keep configurator `results.csv`
throughput as reported; normalize only direct-runner rates by compiled batch.
Use its `qaic-runner`/`qaic.Session` workflow and measure `-mos`, `-ols`, and
`-multicast-weights` only when evidence gates justify them. Run latency and
throughput winners separately when rows differ. For throughput, select only a row
where `cores × instances == NSP_TOTAL` and rank the full-SoC rows by aggregate
completed `Throughput (inf/sec)`; never select a partial-SoC row. If a custom op was introduced,
compare end-to-end with and without it; retain it only when
interpreter output, hardware accuracy, performance, and stability justify it.
Otherwise revert it or assess CPU offload through the owning gates and document
the decision.

## Step 5: Evidence-driven graph optimization

This step is optional. Opstats must already be collected through the performance
skill's diagnostic procedure; if no actionable pattern remains, do not create
`optimize_graph.py` or `_v2.py` and record “no graph optimizations applicable.”
Otherwise use `cloud-ai-performance-optimization/references/graph-optimization.md`
and `cloud-ai-model-conversion/references/onnx-graph-surgery.md`:

1. Audit CPU execution first; check the ONNX support table before prior
   classifications or custom-op investigation.
2. Classify candidates Low (existing ONNX rewrite), Medium (new ONNX/PyTorch
   transform), or High (retraining, broad calibration, or substantial custom-op
   work). Implement observed Low and Medium candidates unless a measured result
   or external blocker rules them out; defer High candidates only with evidence.
3. Prototype ONNX changes in `optimize_graph.py` and PyTorch changes in copied
   `export_<model>_v2.py`; do not edit the shared base. Check `base_*_transform.py
   --list` and support for every replacement op first.
4. Consolidate `_v2.py` as correctness fixes, optimization transforms, then
   `FP16ClipTransform`. The ONNX prototype runs on an already simplified/clipped
   ONNX and must not add `OnnxSimTransform` or `FP16ClipTransform`.
5. Re-run per-output accuracy (optimization gate cosine >= 0.99), isolate/revert
   failures, then recompile and re-benchmark. Record transforms and every
   `# CANDIDATE FOR base_onnx_transform.py` in the model card; promotion is later.

## Step 6: Generate the relocatable demo

Use `cloud-ai-vision-demo` and its [template routing](../cloud-ai-vision-demo/references/template-routing-and-integration.md), [smoke tests](../cloud-ai-vision-demo/references/smoke-tests-and-known-answer-validation.md), and [detection/segmentation assertions](../cloud-ai-vision-demo/references/detection-and-segmentation.md) references. Copy the correct template; do not write `predict.py` from scratch or modify `predict_base.py` (`visualize()`, `_nms()`, and `COLORS` remain shared). Bind the Step 5 QPC, or Step 4 QPC if Step 5 was skipped; discover names/shapes from `qaic_output/`, `session.model_output_names`, or `model_output_shape_dict`. Produce `demo_config.json` with validated input/output shape/dtype, normalization, task, output names, and QPC paths. Keep paths relative. For multi-stage models bind every role-suffixed QPC with the required resource/program-group integration. A random `input.raw`, generic fixture, empty `"boxes": []`, or synthetic image cannot pass detection, two-stage, or OCR smoke validation; use a real scene and preserve the one-line JSON/annotated output for the model card.

## Step 7: Write the model card

Copy `MODEL_CARD_TEMPLATE.md` to `<model_name>/MODEL_CARD.md` and fill every
section, including exact `$MODEL_SOURCE`, source revision, task, architecture,
precision, tags, and date. Record every measured batch, `-a 1 -S 1` standard
latency and throughput, per-output MSE/cosine, QPC/program binary,
profiles, inputs, HMX/non-HMX MAC, barriers, DDR write/spill/DMAIssue, unsupported
ops, graph surgery, custom-op evidence, rollback, and real Step 6 JSON. Scan
`export_*.py` for `CANDIDATE FOR`; do not fill that section from memory.
Populate the model-card tables from the benchmark ledger so every direct-runner
row retains its raw QPC executions/sec and its completed inference units/sec.

## Done criteria

All five are required: (1) QAIC FP16 outputs pass accuracy per output; (2) the
use-case throughput target passes with pipeline headroom; (3) sustained
`qaic-runner` stability passes without errors/device resets; (4) demo contract
and real smoke/known-answer validation pass; and (5) complete `MODEL_CARD.md`
with real smoke JSON and its validation checklist.

## Related skills

- `cloud-ai-env-setup` - device/container setup.
- `cloud-ai-model-conversion` - export, requirements, graph surgery, QPC, manifest.
- `cloud-ai-qaic-compile` - `qaic-compile` flags, support, SDP/MDP/TS.
- `cloud-ai-accuracy-tuning` - ONNX/FP16/INT8 numerical gates.
- `cloud-ai-performance-optimization` - benchmark, opstats, optimization, stability.
- `cloud-ai-architecture` - hardware and transformer context.
- `cloud-ai-custom-ops-kernels` - measured NSP fusions/specialized blocks.
- `cloud-ai-vision-demo` - `predict.py`, `demo_config.json`, real fixtures.
