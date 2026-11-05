# Tool Paths and Python Environment


## Tool Paths

All QAic tools are installed under `/opt/qti-aic/` and are **not** on `PATH` by default:

| Tool | Path |
|---|---|
| `qaic-compile` | `/opt/qti-aic/exec/qaic-compile` |
| `qaic-runner` | `/opt/qti-aic/exec/qaic-runner` |
| `qaic-util` | `/opt/qti-aic/tools/qaic-util` |

Use full paths in commands. For a command that requires `PATH` lookup:

```bash
docker exec "$CONTAINER" bash -lc \
  'export PATH=/opt/qti-aic/exec:/opt/qti-aic/tools:$PATH; <command>'
```

## Install the `qaic` Python Package

The `qaic.Session` Python API is provided as a `.whl` file inside the container:

```bash
docker exec "$CONTAINER" pip install \
  /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

Verify the install:

```bash
docker exec "$CONTAINER" python3 -c 'import qaic; print(qaic.__file__)'
```

## Python Dependencies for Model Work

**Always use the CPU-only PyTorch wheel.** Cloud AI 100 uses QAIC, not NVIDIA CUDA — `pip install torch` defaults to a 2+ GB CUDA build that is completely unused here. The CPU wheel is ~200 MB.

```bash
# torch/torchvision: CPU-only (required — never omit --index-url)
docker exec "$CONTAINER" pip install \
  --index-url https://download.pytorch.org/whl/cpu torch torchvision
# ONNX toolchain
docker exec "$CONTAINER" pip install onnx onnxruntime onnx-simplifier onnxscript
# HuggingFace models
docker exec "$CONTAINER" pip install transformers huggingface_hub
```

Run work through `docker exec "$CONTAINER"`; save files under `/workspace` to ensure persistence.
