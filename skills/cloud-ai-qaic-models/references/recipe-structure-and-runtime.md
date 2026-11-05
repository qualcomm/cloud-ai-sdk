# Recipe Structure and Runtime

## Package boundary

Use the simplest hierarchy that matches reuse:

```text
qaic_models/<domain>/<task>/<model>/
qaic_models/<domain>/<provider>/<family>/<task>/
```

Keep shared code at the narrowest proven boundary: task, family, provider,
then base. Generic predictor lifecycle belongs in
`qaic_models/base/predict_base.py`; session helpers, preprocessing helpers,
compile helpers, ONNX transform infrastructure, PyTorch primitives, and
Triton config generation likewise use their existing base modules.

## Recipe modules

- `__init__.py`: explicit stable exports and `Model` alias for a single default task.
- `model.py`: public factory, CPU/QAIC predictors, artifact resolution, and validated runtime wiring.
- `preprocess.py`: exact input conversion and metadata.
- `postprocess.py`: output validation, interpretation, and JSON schema.
- `visualize.py`: optional task-local rendering or output writing.
- `export.py`: export, transform, compile, and optional Triton generation.
- `pytorch_transforms.py` / `onnx_transforms.py`: optional recipe-specific graph changes.
- `triton_postprocess.py` / `triton_model.py`: optional Python-backend implementations.
- `demo.py`, `model.yaml`, `README.md`: user CLI, catalog metadata, and runbook.

## Runtime contracts

Use `<PublicName>OnnxCPU(..., CPUPredictor)` for CPU reference and
`<PublicName>Qaic(..., QAICPredictor)` for real `qaic.Session` inference. CPU
code must not subclass or instantiate `QAICPredictor`. A public factory accepts
only `CPUExecutionProvider` and `QAicExecutionProvider`; reject other values.
Resolve generated paths to absolute paths.

For one model with several sessions, create named sessions and normalize
intermediate handoff in `_forward()`. For separate chained models, create one
predictor pair per stage and a pipeline helper; do not force a single default
factory where none exists. Return the base result shape:
`{"prediction": <json-safe>, "latency_ms": <float>}`.

Preprocessing returns contiguous arrays/tensors with validated names, shapes,
dtypes, layouts, and metadata. Postprocessing normalizes CPU and QAIC outputs,
validates counts/names/shapes/dtypes, applies only validated decode rules, and
preserves the documented result frame. Visualization handles empty results and
stays recipe-local.

`model.yaml` contains catalog/default metadata only: IDs, tasks, variants,
shapes, semantics, thresholds, artifact names, licenses, and compile notes. It
does not select implementations or execute commands.

## Naming and results

Names for package slugs, classes, variants, tensors, and artifacts come from
evidence or explicit direction. Use `<PublicName>OnnxCPU`, `<PublicName>Qaic`,
and `Model = <PublicName>` where appropriate. Define the prediction JSON schema
once in `postprocess.py`, reuse it in `demo.py` and the README, and test it.
Keep raw tensors out of default user-facing results. Treat labels, tokenizers,
class maps, precision specs, metadata, and golden fixtures as explicit inputs
or generated artifacts with documented locations and failure behavior.
