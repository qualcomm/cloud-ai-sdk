# Environment and Artifacts

Load this reference before writing or running recipe code.

## Environment

Work from the recipe directory and use a recipe-scoped `env_<slug>` virtualenv:

```bash
python3.10 -m venv env_<slug>
source env_<slug>/bin/activate
python -m pip install --upgrade pip
```

Define two repository extras in `pyproject.toml`: `<slug>` for export/compile
and `<slug>-infer` for runtime-only use. Include recipe resources such as
`model.yaml` through package data. Use CPU-only PyTorch, headless image
packages, ONNX tooling, and the container-provided `qaic` wheel. Triton users
compose the shared `triton` extra with the runtime extra:

```bash
pip install -e '/path/to/cloud-ai-sdk-root[<slug>]'
pip install -e '/path/to/cloud-ai-sdk-root[<slug>-infer,triton]'
```

Before export, verify imports such as `onnx`, `onnxruntime`, `cv2`, and `PIL`,
then check device readiness with `/opt/qti-aic/tools/qaic-util -q`. Run all
subsequent commands in this virtualenv.

## Required facts

Record the source checkpoint/framework, export command, input and output names,
shapes, dtypes, layouts, batch policy, preprocessing, postprocessing, labels,
artifact names, compile flags, SDK assumptions, sample inputs, and golden
outputs. Unknown behavior is a blocker, not an invitation to guess.

## Generated artifacts

Every export/compile path accepts `--work-dir` and writes generated ONNX, QPC,
logs, and auxiliary files beneath that caller-owned directory. Resolve it to
an absolute path before passing it to subprocesses or runtime constructors.
Recipes never package, commit, or load model binaries from their source
package. Runtime paths must be caller-provided and documented as generated
outputs.
