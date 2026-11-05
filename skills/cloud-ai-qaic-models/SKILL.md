---
name: cloud-ai-qaic-models
description: Use when adding, migrating, refactoring, or reviewing canonical qaic_models recipes for Cloud AI 100 models in this repository
---

# Cloud AI QAIC Models

Use this skill for canonical recipes under `qaic_models/`. A recipe exports and
compiles caller-owned artifacts, provides CPU and QAIC inference, exposes a
demo, and documents/tests one validated model contract.

## Before coding

Establish model facts first: tensors, layouts, preprocessing, outputs,
postprocessing, artifacts, compile flags, samples, and golden results.
Do not invent a missing fact; route it to model conversion, compile, accuracy,
or environment guidance.

## Non-negotiables

- Keep implementation in `qaic_models/`, tests in `tests/qaic_models/`, and user guidance in the recipe README; do not import `models.*`.
- Keep CPU reference on `CPUPredictor` and QAIC inference on `QAICPredictor`; never make CPU code depend on QAIC sessions.
- Keep recipe semantics local. Promote code to `qaic_models/base` only after a real second consumer proves reuse.
- Keep `model.yaml` catalog-only; Python owns behavior and runtime policy.
- Write generated ONNX/QPC files only beneath a caller-provided work directory; never package or commit model binaries.
- Keep validated graph surgery in recipe-local `onnx_transforms.py`/`pytorch_transforms.py`; `export.py` composes their explicit order.
- Keep prediction schemas consistent across postprocess, demo, README, and tests.

## Route the work

| Need | Read |
|---|---|
| Environment, model facts, work directories, extras | [environment and artifacts](references/environment-and-artifacts.md) |
| Package layout, predictors, preprocessing, postprocessing, YAML | [recipe structure and runtime](references/recipe-structure-and-runtime.md) |
| Export, artifact inspection, compile, transforms | [export and transforms](references/export-and-transforms.md) |
| Demo, README, unit and hardware tests | [demo, README, and testing](references/demo-readme-and-testing.md) |
| Triton repository, batching, backend, output schema | [Triton serving](references/triton-serving.md) |

## Lifecycle

Facts → package boundary → CPU/QAIC runtime → preprocessing/postprocessing →
export/transform/compile → demo and README → hardware-free tests → optional
hardware smoke test. Validate the final deployable artifact, not an earlier
raw export. Keep generated artifacts outside recipe source.

## Stop conditions

Stop when a recipe invents model facts, reads bundled artifacts, puts model
behavior in YAML, subclasses `QAICPredictor` for CPU, duplicates base classes,
rewrites graphs inline in `export.py`, or changes validated postprocessing
without matching golden behavior.
