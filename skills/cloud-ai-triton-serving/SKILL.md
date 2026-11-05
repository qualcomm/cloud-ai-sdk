---
name: cloud-ai-triton-serving
description: Use when serving Cloud AI 100 models with Triton Inference Server, configuring QAIC or ONNX model repositories, or troubleshooting model shapes, readiness, endpoint contracts, or serving configuration.
---

# Cloud AI 100 - Triton Serving

This skill covers Triton model repositories, Triton endpoints, and the
backends loaded by Triton.

## Quick Start

Use the SDK 1.23.2.0 Triton image below. For another SDK, use its matching
Triton image.

```bash
docker run --rm -it \
  --network host \
  --shm-size=4g \
  --ipc=host \
  --device /dev/accel/ \
  --mount type=bind,source=<path/to/workspace>,target=/workspace \
  --mount type=bind,source=<path/to/model-repository>,target=/models \
  ghcr.io/qualcomm/cloud_ai_triton_server:1.23.2.0 \
  /opt/tritonserver/bin/tritonserver --model-repository=/models
```

- Use `--shm-size=4g` for this image.
- Use `/opt/qti-aic/integrations/triton/release-artifacts/config-generation-script/triton_config_generator.py` to generate `config.pbtxt` from `programqpc.bin` or `model.onnx`.
- Sample repositories are under
  `/opt/qti-aic/aic-triton-model-repositories/` inside the container.
- Inspect the installed image for the available QAIC custom, ONNX Runtime,
  Python, and vLLM backends before using backend-specific options.

Each model directory in a repository contains `config.pbtxt` and a versioned
`1/` directory containing its backend artifact, such as `programqpc.bin`,
`model.onnx`, or `model.py`. Triton ensembles can compose these models.

### Validate the model contract

Treat generated configuration as a starting point. Before launch, obtain the
compiled model's exact input/output names, dtypes, and full shapes from its IO
descriptor or source graph, then make the Triton config match those values.

For a fixed-shape artifact whose shape already includes batch, use
`max_batch_size: 0` and keep the complete dimensions. For a Triton-batched
model, use `max_batch_size > 0` and omit the batch dimension from every
`dims` entry; use this only when the compiled artifact supports that batching
contract. Never combine a nonzero `max_batch_size` with dimensions that already
include batch unless the backend explicitly documents that behavior.

Launch with strict configuration checking when available, inspect the loaded
model config/metadata, and send one real inference request with the exact
declared shape. Readiness alone does not validate tensor compatibility.

## Backend selection

| Backend | Use |
|---|---|
| `qaic` | Run a precompiled QPC (`programqpc.bin`) on Cloud AI hardware. |
| `onnxruntime_onnx` | Run an ONNX graph through ONNX Runtime with the QAIC execution provider. |
| `python` | Run custom Python model logic, including LLM and embedding backends. |
| `ensemble` | Compose multiple Triton models into one request pipeline. |

### QAIC backend

Set `backend: "qaic"` in `config.pbtxt`. The backend parameters are:

| Parameter | Guidance |
|---|---|
| `qpc_path` | Optional path to `programqpc.bin`; if omitted, the backend searches the model directory. |
| `device_id` | Optional Cloud AI device ID; omitted means the server auto-selects an available device. |
| `set_size` | Triton QAIC runtime queue size; the documented default is 20. |
| `no_of_activations` | Number of model activations; the documented default is 1. |

Set Triton's `max_batch_size` and tensor shapes to match the compiled model
contract. Do not infer them from a generic Triton default.

Keep the QPC path portable. When `programqpc.bin` is in the model's versioned
directory, omit `qpc_path` if the backend searches that directory by default.
If `qpc_path` is required, resolve it at deployment time inside the serving
container; do not retain a path that exists only in the build environment.

### Serving boundary

Expose one client-facing API surface through Triton using its native tensor
contract. Clients send and receive the named, typed tensors declared by the
top-level model configuration. Clients own decoding, preprocessing, and any
required postprocessing; validate them against the same reference
implementation.

### ONNX Runtime backend

Use platform `onnxruntime_onnx` and set the `use_qaic` parameter to `true` to
execute an ONNX graph through the QAIC execution provider. The backend also
supports a compiler `config`, optional `device_id`, and `share_session` for
sharing an ONNX Runtime session across instances.

## Recipe-generated repositories

When a model recipe provides a `--triton-model-repo` option, generate the
repository after export or compilation, then mount the source checkout into
the Triton container. Install the recipe's runtime extra and the shared
`[triton]` extra inside that container when the repository's Python backend
imports recipe code:

```bash
python -m <recipe>.export \
  --compile \
  --triton-model-repo <work-dir>/triton-model-repo

docker run --rm -it \
  --network host --shm-size=4g --ipc=host --device /dev/accel/ \
  --mount type=bind,source=<path/to/sdk-root>,target=/cloud-ai-sdk \
  ghcr.io/qualcomm/cloud_ai_triton_server:1.23.2.0 \
  bash -lc 'pip3 install "/cloud-ai-sdk[<recipe-runtime>,triton]" && \
    /opt/tritonserver/bin/tritonserver \
      --model-repository=/cloud-ai-sdk/<work-dir>/triton-model-repo'
```

Do not assume the plain Triton image contains recipe-local Python packages.

## Triton vLLM backend

The Triton vLLM backend and standalone vLLM serving image are version-coupled.
Match the Triton image with the same SDK-tagged vLLM container and the same
variant, such as pairing an `_085` Triton backend with
`cloud_ai_inference_vllm_085` at the same image tag. Do not pair a Triton
image with the default vLLM container when the backend was built for another
vLLM variant or version.

Before using version-specific options, inspect the installed package in both
containers and confirm that the versions match:

```bash
source /opt/vllm-env/bin/activate
command -v vllm
pip show vllm
```

## Triton Python LLM backend

This backend supports decoder-only causal models with batch responses or
decoupled streaming responses. Activate the QEfficient environment and use
the repository generator shipped in the container:

```bash
source /opt/qeff-env/bin/activate
python /opt/qti-aic/integrations/triton/release-artifacts/llm-models/generate_llm_model_repo.py \
  --model_name <model> \
  --aic_binary_dir <path/to/qpc> \
  --python_backend_dir /opt/qti-aic/integrations/triton/backends/qaic/qaic_backend_python/
/opt/tritonserver/bin/tritonserver --model-repository=<path/to/llm_model_dir>
```

Batch mode uses HTTP. Decoupled or streaming mode requires gRPC.

The generator can create a repository from an existing QPC or compile from a
configuration that describes values such as `batch_size`, `full_batch_size`,
`prefill_seq_len`, `ctx_len`, `num_cores`, and precision options. Use the
generator's `--help` output for the version-specific option names.

## Health checks and benchmarking

The Triton server exposes readiness at `/v2/health/ready`. A loaded model can
be checked at `/v2/models/<model-name>/ready`. Use HTTP or gRPC clients that
match the model's configured inputs and outputs.

Use Triton's `perf_analyzer` for serving benchmarks. Keep the model's compiled
batch contract, instance count, request shape, concurrency, and endpoint
visible in the benchmark record.

## Related skill

Use `cloud-ai-vllm-serving` for a standalone vLLM server, its CLI flags, and
its OpenAI-compatible API.
