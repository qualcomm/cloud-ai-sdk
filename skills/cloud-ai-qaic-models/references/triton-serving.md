# Triton Serving for QAIC Model Recipes

Use this reference only after export, compile, CPU reference, QAIC runtime, and demo contracts are validated. Build the Triton repository from generated ONNX/QPC artifacts and the recipe's established I/O and postprocess contracts.

## Repository generation

- Use `qaic_models/base/triton.py` and recipe I/O metadata where possible.
- Add an `export.py` option such as `--triton-model-repo`. Write only beneath its caller-supplied path and only after required artifacts exist in `--work-dir`.
- Derive raw-tensor specifications from ONNX/QPC metadata. For user-facing predictions, use a Python-backend postprocess model with an explicit Triton schema.
- Test generated repository layout and config text without hardware. Document repository generation, container/device setup, package extras, the `tritonserver --model-repository=...` command, model names, a minimal client request, and raw versus postprocessed outputs in the recipe README.

## Model and Python-backend layout

Name models by role:

| Role | Name |
| --- | --- |
| End-to-end generated QPC | `<slug>_qpc` |
| Raw ONNX | `<slug>_onnx` |
| Python postprocess | `<slug>_postprocess` |
| ONNX plus postprocess ensemble | `<slug>_onnx_ensemble` |
| Stage-wise pipeline | `<slug>_<stage>`; add `_qpc` when QPC-served |

Keep backend implementations in the recipe: `triton_model.py` for one model, or `triton_<stage>_model.py` per stage. The generated version directory's `model.py` points to that module. Put client examples in `demo_triton.py`.

## QPC batching and variable outputs

A QPC-served model uses the throughput-build QPC. Configure dynamic-batching instances so `instances × compiled batch` fills one SoC. If a coalesced request underfills a compiled batch, cycle real request inputs into empty slots. Never zero-pad compiled QPC batches. Cross-batch interaction can let zero rows shift real outputs.

Choose and test an explicit representation for variable-size outputs. Default user-facing output: a `STRING [1]` JSON tensor containing the recipe postprocess schema plus an `INT32` count tensor. Use padded tensors plus a count or fixed top-k tensors only for clients that require raw tensors.
