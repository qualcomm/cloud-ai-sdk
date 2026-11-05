# Demo, README, and Testing

## Demo contract

Every runnable image, text, or audio recipe has a small CLI. It accepts the
user input and `--model` pointing to a generated artifact directory or direct
artifact, plus provider selection when CPU and QAIC paths both exist. It may
accept `--out` for a rendered/file result and supports `--json-out` when the
machine-readable prediction is separate from that output.

The CLI instantiates the public factory, calls `predict()`, prints stable
latency/prediction output, and reports written files. It does not duplicate
predictor logic. `--json-out` writes `result["prediction"]` exactly: the
postprocess payload without the latency or outer `prediction` wrapper.

## README contract

Write each recipe README as an end-user runbook. Cover:

- What runs, supported variants, input/output form, target accelerator, and limitations.
- Generated files and the path passed to `--model`.
- Python version, `env_<slug>`, repository extras, CPU/headless dependencies, SDK/device checks, and installation.
- Exact export/compile commands, flags, samples, generated layout, and work-directory regeneration.
- CPU and QAIC demo commands, API examples, result schema, thresholds, metadata, and empty-result behavior.
- A concrete smoke result and CPU-versus-QAIC comparison when validated.
- Troubleshooting for missing tools/devices/extras, headless imports, checkpoints, paths, QPCs, stale work directories, compiler errors, and Triton mounts/imports.
- Upstream license and distribution restrictions.

Recipes with Triton generation also document repository generation, server
container/device access, in-container package installation, the
`tritonserver --model-repository=...` command, model names, request shape, and
raw-versus-postprocessed output schemas.

## Testing contract

Default tests are hardware-free and cover structure/imports, absence of
`models.*` imports and generated binaries, deterministic preprocessing,
synthetic/golden postprocessing, predictor inheritance, artifact resolution,
and `demo.py --help` / `export.py --help`. Triton tests inspect generated
repository layout and config text without hardware.

Hardware tests live in `test_<model>_hardware.py`, auto-skip without the
`qaic` module, a Ready device, or `<MODEL>_WORK_DIR`, and run export/compile,
CPU-versus-QAIC comparison, and a real sample smoke check.
