---
name: cloud-ai-llm-onboarding
description: Use when onboarding, compiling, or validating a QEfficient-supported LLM, VLM, MoE, diffusion, embedding, reranker, or ASR model for Cloud AI 100.
---

# Cloud AI LLM Onboarding

Use QEfficient for supported generative-model workflows. For current
architecture support, see the [upstream validation guide](https://github.com/qualcomm/efficient-transformers/blob/main/docs/source/validate.md).
Use this skill when QEfficient / efficient-transformers is the requested workflow.

## QEfficient Runtime

Cloud AI containers provide QEfficient at `/opt/qeff-env/bin/python`. Set the
artifact root before importing it, then use the bundled package:

```bash
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME
mkdir -p "$HF_HOME"

QEFF_HOME="${QEFF_HOME:-$HOME/.cache/qeff_models}"
export QEFF_HOME
mkdir -p "$QEFF_HOME"

QEFF_PYTHON="${QEFF_PYTHON:-/opt/qeff-env/bin/python}"
"$QEFF_PYTHON" -c "import QEfficient; print(QEfficient.__file__)"
```

Use a writable, persistent `QEFF_HOME` for exported models and QPC artifacts.
By default, Hugging Face weights use `HF_HOME`; QEfficient exports and QPC
artifacts use `QEFF_HOME`. Ensure the chosen storage can hold downloaded
weights, exported ONNX, and QPC artifacts at the same time.
If the task explicitly requires a newer release or upstream `main`, install that requested version and set
`QEFF_PYTHON` to its interpreter.

Clone the [upstream repository](https://github.com/qualcomm/efficient-transformers)
when its examples, reference implementations, or contributor skills are needed.

### Quick Start

First read `examples/README.md` in the upstream checkout and select the closest
example for the model family or workflow. The generic CLI and Python snippets are causal-LM examples; use
the closest family example instead of adapting them across model families.
Select examples for their architecture, conversion, and export patterns.

Pre-quantized checkpoints are loaded automatically from their
`quantization_config`; ordinary `from_pretrained` and `cloud.infer` usage
applies. For AWQ, repeated `AWQ unpacked values exceed 4-bit range` messages
during export are expected masking notices, not checkpoint corruption.

```bash
"$QEFF_PYTHON" -m QEfficient.cloud.infer \
  --model_name <model-id> --num_cores <nsp-cores> --batch_size <batch-size>
```

Before executing a QPC, including the final `cloud.infer` stage, confirm a target accelerator is ready:

```bash
/opt/qti-aic/tools/qaic-util -q
```

Continue only if all required QIDs report `Status:Ready`. If none do,
**REQUIRED SUB-SKILL:** use `cloud-ai-env-setup`.

`cloud.infer` loads or downloads, exports, compiles, and executes a single QPC
with the requested shapes and specializations. Set `--prompt-len`, `--ctx-len`,
`--generation-len`, and `--device-group` for the workload; use TTFT and decode
rate in its output as the smoke-test signal.

```python
from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained("<model-id>")
model.export()
model.compile(num_cores=<nsp_cores>, batch_size=<batch_size>)
model.generate(prompts=["Hello"])
```

| Workflow | Entry point | Example area |
|---|---|---|
| Causal LLMs and MoE | `QEFFAutoModelForCausalLM` | `examples/text_generation/` |
| Vision-language models | `QEFFAutoModelForImageTextToText` | `examples/image_text_to_text/` |
| Text embeddings | `QEFFAutoModel` | `examples/embeddings/` |
| Sequence classification and reranking | `QEFFAutoModelForSequenceClassification` | `examples/sequence_classification/` |
| Whisper ASR | `QEFFAutoModelForSpeechSeq2Seq` | `examples/audio/speech_to_text.py` |
| Wav2Vec2 ASR | `QEFFAutoModelForCTC` | `examples/audio/wav2vec2_inference.py` |
| FLUX image generation | `QEffFluxPipeline` | `examples/diffusers/flux/` |
| WAN text-to-video | `QEffWanPipeline` | `examples/diffusers/wan/` |
| WAN image-to-video | `QEffWanImageToVideoPipeline` | `examples/diffusers/wan_i2v/` |

Follow the selected example's processor, compile arguments, and inference call.

## Required Output

Create an end-to-end `<model_name>_runner.py`. Use a filesystem-safe model stem,
for example `gpt2_runner.py`. Start from the closest upstream example as a reference, if available.

The runner must:

- accept model-appropriate inputs and the selected compile/runtime settings;
- load, export, compile, and execute with QEfficient, allowing it to reuse
  matching cached artifacts;
- support `--compile-only` for a host that builds a QPC without a QAIC device;
- execute a real model-specific input and emit the decoded or task-appropriate result with
  measured performance including TTFT time to first token and decode throughput.  Also print the QPC path location.  Emit both the prompt or input used for testing and the generated output.

If QEfficient source changes are required for the model, preserve the commit
SHA(s) or a `qefficient.patch` with a pinned SHA alongside the runner. Do not rely on
uncommitted QEfficient changes.



## Causal LLM Baseline

For causal LLMs and MoE, start with these settings unless the closest model
example explicitly differs. They do not apply unchanged to VLM, diffusion,
embedding, reranking, or ASR workflows.

| Setting | Initial choice | Change when |
|---|---|---|
| `prefill_seq_len` | `128` prompt tokens per prefill chunk | The closest model example requires a different prompt shape or chunking policy. |
| `mxfp6_matmul` | `True` | Quality validation requires a precision change. |
| `mxint8_kv_cache` | `True` | Quality validation requires a KV-cache precision change. |
| Context | min(8192, max supported by model) | If larger context is requested or use case requires it then measure TTFT and decode throughput before locking in higher value, especially above 32k.  32k and above need smaller CCL buckets added.
| SoCs | Choose the smallest of 1, 2, 4, 8, or 16 SoCs that fits the model, KV cache, and activations; add SoCs only for measured decode throughput. | See `references/multi-device-sizing.md`. |
| Cores per SoC | Use the highest available core count. | Reduce it only to place multiple models on the same SoC. |

## Workflow Decisions

- TTS, standalone ViT, and standalone CNN models are outside QEfficient's
  supported workflows. Use the model-conversion workflow for those models.
- To measure perplexity of a QPC (wikitext), or if a QEff artifact regresses
  in output quality or perplexity, use
  `references/qefficient-accuracy.md`.
- For causal LLMs, benchmark context lengths and prompt/generation mix before
  selecting artifacts for deployment. Use `references/perf_sweep.py`, a
  parameterized causal-LM sweep that writes
  one JSON record per measured QPC and workload.
- Long-context QPCs require Compute Context Length (CCL). Add smaller
  prefill/decode buckets so short requests avoid the largest bucket's TTFT and
  decode throughput. See
  `references/compute-context-lengths.md` for compatibility checks, selection,
  and compile inputs.
- For tensor slicing, first fit model weights, KV cache, and activation
  headroom, then add SoCs only when measured decode throughput requires it.
  `mdp_ts_num_devices` counts SoCs (QIDs), not cards. See
  `references/multi-device-sizing.md` for sizing and concurrent-pipeline
  allocation.

**REQUIRED SUB-SKILL:** For vLLM or Triton serving, use `cloud-ai-vllm-serving`
after producing QPC artifacts.

For an unsupported architecture, use `qeff-model-onboarding` to add the model
wrapper or `qeff-transform-authoring` to add a required transform in an
upstream fork. If neither is available, use fixed-context QPCs where the
existing model path supports them.

For QEfficient internals, transforms, custom operations, and export modes, see
`references/qefficient_transformers_details.md`.
