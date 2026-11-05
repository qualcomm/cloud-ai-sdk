---
name: cloud-ai-vllm-serving
description: Use when serving, deploying, or configuring a standalone vLLM server on Cloud AI 100, or when using vLLM features such as speculative decoding, LoRAX, disaggregated serving, prefix caching, multimodality, CCL, embeddings, on-device sampling, guided decoding, or tool calling.
---

# Cloud AI 100 - vLLM Serving Reference

## Quick Start - vLLM Docker

**SDK 1.23.2.0 vLLM image:** `ghcr.io/qualcomm/cloud_ai_inference_vllm:1.23.2.0`
Registry: <https://github.com/qualcomm/cloud-ai-containers/pkgs/container/cloud_ai_inference_vllm>

> Use `cloud_ai_inference_vllm` for serving. Use `cloud_ai_inference_ubuntu24` for compilation, benchmarking, and general dev work.

The vLLM image entrypoint is the vLLM server itself - pass model arguments directly (no `python -m` prefix).

```bash
docker pull ghcr.io/qualcomm/cloud_ai_inference_vllm:1.23.2.0

docker run --rm -it \
  --network host \
  --shm-size=2gb \
  --device /dev/accel/ \
  -e HF_TOKEN=<your_hf_token> \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/qeff_models:/root/.cache/qeff_models" \
  ghcr.io/qualcomm/cloud_ai_inference_vllm:1.23.2.0 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --additional-config \
  '{"override_qaic_config":{"num_devices":4}}' \
  --host 0.0.0.0 --port 8000
```

The QAIC override sets QEfficient's tensor-sliced compile device count to four.
With no explicit `device_group`, the QAIC runtime auto-picks available device
IDs. Add a `device_group` list in `additional-config` when explicit IDs are
required. Replace `<your_hf_token>` with a token for gated models; omit the
`-e HF_TOKEN` argument for public models if no token is needed.

## Test the endpoint

After the server reports ready, use the OpenAI Python client. The vLLM
serving image already includes the `openai` package in
`/opt/vllm-env`. Set `VLLM_BASE_URL` to the reachable server address; the
default below is the standard local port.

```python
import os

from openai import OpenAI

base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
client = OpenAI(
    base_url=f"{base_url.rstrip('/')}/v1",
    api_key="not-used-by-vllm",
)
model = client.models.list().data[0].id
expected = "QAIC endpoint is working."
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": f"Reply with exactly: {expected}"}
    ],
    temperature=0,
    max_tokens=128,
)
answer = response.choices[0].message.content.strip()
assert answer == expected, f"Unexpected response: {answer!r}"
print(answer)
```

For a forwarded or non-default listener, set `VLLM_BASE_URL` accordingly, for
example `http://127.0.0.1:8905`.

The standard image uses `/root` as its home directory. For the vLLM
QEFFAutoModel path, with neither `QEFF_HOME` nor `XDG_CACHE_HOME` set,
QEfficient stores compiled models under `/root/.cache/qeff_models`; Hugging
Face uses `/root/.cache/huggingface`.
Mount those paths unless you explicitly set the corresponding environment
variables to different in-container paths.

## Serving guidance

Read [serving-reference.md](serving-reference.md) for:

- SDK-specific CLI flags and QAIC settings;
- container inspection, cache paths, and environment variables;
- SDK 1.21 compatibility;
- advanced serving features, benchmarking, and tool calling.

Read [disaggregated-serving.md](disaggregated-serving.md) for PD/E-PD serving.
Read [multimodality.md](multimodality.md) for multimodal QPC and input guidance.
**REQUIRED REFERENCE:** Read
[speculative-decoding.md](speculative-decoding.md) before configuring
draft-model or N-gram speculative decoding.

Use `cloud-ai-triton-serving` for Triton model repositories and Triton's vLLM
backend.
