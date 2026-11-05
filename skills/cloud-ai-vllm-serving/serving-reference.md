## Inspect the installed container

Treat the installed image as the source of truth for version-specific flags.
Run the commands in the selected serving container, not on the host.
The standard SDK 1.22 image initializes its serving environment automatically.
Check the executable, Python environment, package version, and supported flags
that are actually being used:

```bash
command -v vllm
vllm --version
python -c 'import sys, vllm; print(sys.executable); print(sys.prefix); print(vllm.__version__)'
vllm serve --help
```

In SDK 1.22.8.1, `vllm` resolves to `/opt/vllm-env/bin/vllm` and reports
`0.15.0+qaic1.22.aot1.22`. Use `command -v` and `sys.executable` rather than
assuming that a host virtual environment or `$VIRTUAL_ENV` is the serving
environment.

The installed QAIC integration is under `/opt/qti-aic/integrations/vllm/`:

```bash
ls /opt/qti-aic/integrations/vllm/
ls /opt/qti-aic/integrations/vllm/qaic_vllm_*.patch
```

For SDK 1.22 source builds, the matching patch is
`/opt/qti-aic/integrations/vllm/qaic_vllm_0.15.0.patch`. The directory also
contains benchmark scripts, `qaic_disagg`, and `vllm-qaic` wheels and install
scripts. The pre-built image already contains the QAIC integration, so do not
apply the patch during normal serving. Apply a version-matched patch only
when building vLLM from source, after verifying the installed vLLM version.

### Container source map

This reference is a starting point, not an exhaustive list of image-specific
flags. When a setting or feature is not covered here, inspect the selected
image before inventing a command:

```bash
vllm serve --help=all
VLLM_ROOT="$(python -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent)')"
grep -RInE 'additional_config|override_qaic_config|device_group|num_devices|num_cores' \
  "$VLLM_ROOT" /opt/qti-aic/integrations/vllm \
  | sed -n '1,160p'
```

Use these locations first:

| Need | Inspect in the container |
|---|---|
| QAIC integration setup and version coupling | `/opt/qti-aic/integrations/vllm/README.md` and the matching `qaic_vllm_*.patch` |
| Known-good QAIC examples | `/opt/qti-aic/integrations/vllm/examples/offline_inference/qaic_*.py` and `examples/online_serving/` |
| Benchmark commands and input modes | `/opt/qti-aic/integrations/vllm/benchmarks/README.md` and `benchmarks/benchmark_*.py` |
| Installed parser and QAIC behavior | `$VLLM_ROOT/platforms/qaic_base.py` and `$VLLM_ROOT/model_executor/model_loader/qaic.py` |
| Feature-specific vLLM behavior | `$VLLM_ROOT` and the matching example under `/opt/qti-aic/integrations/vllm/examples/` |

If the installed `--help`, source, README, and an older example disagree,
follow the installed parser and source for the reported vLLM/SDK version.
Do not copy a flag from another SDK image without verifying it in that image.

## QAIC Serving Settings

For Cloud AI accelerators, use these workload starting points instead of generic vLLM
defaults. Set them explicitly and measure the target model and workload.

| Setting | QAIC starting guidance |
|---|---|
| `--model` | Set the HuggingFace model name or local model path. |
| `--additional-config` | For SDK 1.22, pass `device_group` when selecting IDs and nested `override_qaic_config` values as JSON. |
| `--kv-cache-dtype mxint8` | Use this as the standard QAIC starting point. |
| `--quantization mxfp6` | Start with this as the standard QAIC quantization setting. |
| `--max-num-seqs` | Use `--max-num-seqs 1` for latency-oriented LLM runs. For short-context throughput tests on smaller models, choose `--max-num-seqs 8`, `--max-num-seqs 16`, `--max-num-seqs 32`, or `--max-num-seqs 64` according to the workload. Choose the value that matches the workload. This is the maximum concurrent sequence count, not a universal batch size. |
| `--max-model-len` | Set the required context length for the workload. Use `--max-model-len 8192` or `--max-model-len 32768` when that matches the workload. |

For SDK 1.22, pass QAIC compiler/runtime overrides through
`--additional-config` (see below). Do not pass `--override-qaic-config` to
`vllm serve`; that standalone flag belongs to the older SDK 1.21 wrapper.

## `additional-config` QAIC Overrides (SDK 1.22)

```bash
--additional-config '{
  "device_group": [0, 1, 2, 3],
  "override_qaic_config": {
    "num_cores": 16,
    "mxfp6_matmul": true,
    "mos": 1
  }
}'
```

| Attribute | Default | Description |
|---|---|---|
| `num_cores` | Device-dependent | Set the available NSP cores per SoC for the target SKU. AI080 has 8 NSP cores per SoC. Use a lower value only for deliberate multi-model packing on shared SoCs, then validate the topology and performance. |
| `num_devices` | 1 unless `device_group` or an override sets it | QEfficient tensor-sliced device count. In SDK 1.22, pass it as nested `override_qaic_config` JSON under `--additional-config`, as shown in the quick start. |
| `draft_override_qaic_config` | Absent | QAIC integration key for a different draft-QPC configuration during draft-model SpD. It is not a standalone vLLM flag. |
| `mxfp6` / `mxfp6_matmul` | False | MXFP6 weight quantization |
| `mxint8` / `mxint8_kv_cache` | False | MXINT8 KV cache compression |
| `mos` | Omitted | Set explicitly for a tested compiler configuration; do not treat omission as `1`. |
| `dfs` / `aic_enable_depth_first` | True | Depth-first scheduling |
| `device_group` | None | Top-level `additional-config` list of device IDs, e.g. `[0, 1, 2, 3]`. QEfficient derives `num_devices` from this list. |
| `ccl_enabled` | False | Enable CCL context specializations |
| `aic_include_sampler` | False | On-device sampling |
| `kv_offload` | False | KV cache offload |
| `comp_ctx_lengths_prefill` | None | CCL prefill context lengths (manual mode) |
| `comp_ctx_lengths_decode` | None | CCL decode context lengths (manual mode) |
| `embed_seq_len` | None | Multi-length support for embeddings |
| `aic_include_guided_decoding` | False | Enable guided decoding |
| `skip_lang` / `skip_vision` | False | Skip language/vision binary in dual-QPC |
| `pooling_method` / `normalize` | None/False | Embedding pooling settings |

When `draft_override_qaic_config` is absent, the top-level
`override_qaic_config` is used for both target and draft QPCs. If the draft
override is supplied, it replaces the draft configuration; repeat every
required setting, including `num_devices`.

## Environment Variables

| Variable | Description |
|---|---|
| `QAIC_VISIBLE_DEVICES` | Comma-separated QIDs visible to eager-mode vLLM. AOT serving uses `device_group` in `additional-config` directly. |
| `QEFF_HOME` | QEfficient compiled-model/QPC cache. When both `QEFF_HOME` and `XDG_CACHE_HOME` are unset, the current image uses `/root/.cache/qeff_models`. |
| `HF_HOME` | Hugging Face model cache. When `HF_HOME` is unset, the current image uses `/root/.cache/huggingface` unless another cache setting changes it. |
| `OUTLINES_CACHE_DIR` | Cache directory for guided-decoding artifacts. |
| `VLLM_QAIC_QPC_PATH` | Path to pre-compiled QPC (skip compile) |
| `VLLM_QAIC_NUM_CORES` | Override `num_cores` for the selected SKU; AI080 has 8 NSP cores per SoC. |
| `VLLM_QAIC_COMPILER_ARGS` | Additional compiler arguments |
| `VLLM_QAIC_MAX_CPU_THREADS` | Limit CPU threads used by Python sampling, useful under container CPU limits or with multiple instances. |
| `VLLM_QAIC_DFS_EN` | Enable depth-first scheduling |
| `VLLM_QAIC_MOS` | MOS value |
| `VLLM_TORCH_QAIC_PROFILER_DIR` | Output directory for QAIC profiler data. |
| `HF_TOKEN` | HuggingFace token for gated models |

## Bare Metal (no Docker)

```bash
ulimit -n 1048576           # required - increase file descriptor limit
export OMP_NUM_THREADS=8    # prevent CPU oversubscription in multi-instance
export VLLM_QAIC_MAX_CPU_THREADS=8

vllm serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --device qaic \
  --additional-config '{"device_group":[0,1,2,3]}' \
  --host 0.0.0.0 --port 8000
```

Use the installed entry point for the selected image. Do not replace
`vllm serve` with a Python module path copied from another vLLM version unless
that image's `--help` or README requires it.

## Image selection

Use an image tag confirmed in the package registry and matched to the SDK and
integration version. Do not infer that a suffix such as `_py312` or `_disagg`
exists, or that it contains the same vLLM integration. Verify the selected
image with `vllm --version`, `vllm serve --help`, and the container source map
above. For SDK 1.23.2.0, use:
`ghcr.io/qualcomm/cloud_ai_inference_vllm:1.23.2.0`.

---

## SDK 1.21 compatibility

SDK 1.21 uses vLLM `0.10.1.1+qaic1.21`. Its wrapper exposes direct
`--device-group` and `--override-qaic-config` flags. It also exposes
`--max-seq-len-to-capture`; the SDK 1.21 QAIC integration reuses that setting
for the prefill compile length. It is not part of the SDK 1.22 vLLM 0.15
interface used by the SDK 1.23.2.0 quick-start command.

These flags describe the historical SDK 1.21 interface. Do not use them with
the SDK 1.23.2.0 image referenced by the current examples.

Use the legacy non-JSON syntax in that image:

```bash
--device-group 0,1,2,3 \
--override-qaic-config "num_cores=16 mxfp6_matmul=True mos=1"
```

Do not wrap the legacy override string in SDK 1.22's
`{"override_qaic_config":{...}}` JSON object. Before reusing a command across
SDK versions, run `vllm serve --help` in the installed image and inspect the
version-matched QAIC integration.

Use only flags supported by that image.

## Advanced Features

### On-Device Sampling

```bash
--additional-config '{"override_qaic_config":{"aic_include_sampler":true,"max_top_k_ids":512}}'
```

Supported strategies: Repetition Penalty, Presence Penalty, Temperature Scaling, Top-K, Top-P, Min-P, Greedy, Random.
**Incompatible with:** Prefix Caching, Speculative Decoding, Multimodality, LoRAX. Frequency penalties not supported.
For image-specific options and limits, inspect
`/opt/qti-aic/integrations/vllm/examples/offline_inference/qaic_on_device_sampling.py`
and the installed QAIC model-loader source.

### Guided Decoding

Constrain output to JSON schema, regex, EBNF grammar, or choices. Only **xgrammar** backend supported.

Model support is version- and integration-dependent. Check the installed
vLLM/QEfficient compatibility information before enabling guided decoding.

### Embedding Networks

```bash
--task embed \
--max-model-len 512 \
--additional-config '{"override_qaic_config":{"embed_seq_len":[128,256,512],"pooling_device":"qaic"}}'
```

Model and pooling support is image- and integration-dependent. Inspect the
installed model-loader source and
`/opt/qti-aic/integrations/vllm/examples/offline_inference/basic/qaic_embed.py`
before selecting a model or adding `trust_remote_code=True`.
**Incompatible with** Prefix Caching unless the installed integration documents
support for the selected embedding path.

### Compute Context Length (CCL)

CCL is a QEfficient compile feature. It consumes automatic or explicit
context-length lists, adds the corresponding network specializations, and
writes them to the generated `specializations.json` used for compilation.
The resulting specializations are packaged into a single QPC:

```bash
--additional-config '{"override_qaic_config":{"ccl_enabled":true}}'
# Manual (advanced): use workload context buckets, such as 1K, 8K, and 32K
--max-model-len 32768 \
--additional-config '{"override_qaic_config":{"ccl_enabled":true,"comp_ctx_lengths_prefill":[1024,8192,32768],"comp_ctx_lengths_decode":[1024,8192,32768]}}'
```

Set `--max-model-len` equal to the largest intended CCL bucket:
`32768` in this example. A larger value adds an upper-bound bucket; a smaller
value clamps requested values.

Inspect the generated artifact in the selected QEfficient cache or workspace:

```bash
QEFF_ROOT="${QEFF_HOME:-/root/.cache/qeff_models}"
if [ -d "$QEFF_ROOT" ]; then
  find "$QEFF_ROOT" -type f -name specializations.json -print
fi
```

If the file is elsewhere, use the compile log's output path. The installed
integration example for context-length behavior is
`/opt/qti-aic/integrations/vllm/examples/offline_inference/context_extension.py`.

Biggest gains are typically at 8K-token and longer contexts. CCL is supported
with standalone vLLM and speculative decoding, and with disaggregated serving
on the decode side.

### Prefix Caching

```bash
--enable-prefix-caching
```

**Incompatible with:** multimodality, LoRAX, embeddings, disaggregated serving, on-device sampling.
For the QAIC-specific implementation and example, inspect
`/opt/qti-aic/integrations/vllm/examples/offline_inference/qaic_prefix_caching.py`.

**REQUIRED REFERENCE:** Read
[speculative-decoding.md](speculative-decoding.md) before configuring
draft-model or N-gram speculative decoding.

### LoRAX - Multi-Adapter Serving

```bash
--enable-lora --max-loras <N>
```

Up to 128 adapters (`VLLM_QAIC_LORA_MAX_ID_SUPPORTED`). Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` only.
**Incompatible with** prefix caching.
Check `/opt/qti-aic/integrations/vllm/examples/offline_inference/qaic_lora.py`
and the installed model-loader source for version-specific adapter limits.

### Disaggregated Serving

Read [disaggregated-serving.md](disaggregated-serving.md) for PD/E-PD topology,
launch examples, and proxy validation. Confirm its flags with
`python3 -m qaic_disagg --help` in the installed image.

### Multimodality

Read [multimodality.md](multimodality.md) for single/dual QPC and
multi-resolution input guidance.

### Benchmarking vLLM serving performance

Use the benchmark tooling shipped with the installed integration:

- `vllm bench serve` or `benchmark_serving.py` for online serving;
- `benchmark_throughput.py` for offline throughput;
- the installed benchmark README for dataset and request modes.

The shipped files are under
`/opt/qti-aic/integrations/vllm/benchmarks/`. Read that directory's README and
the selected script's `--help` before copying a benchmark command; benchmark
arguments vary with the installed vLLM version.

Select a dataset and endpoint compatible with the model under test. For
multimodal serving, use the OpenAI-compatible chat backend and inputs matching
the model's supported modalities.

Record the model, endpoint, dataset, prompt/output lengths, request rate or
concurrency, sampling parameters, runtime/compiler configuration, and result
metrics. Keep benchmark inputs and result files outside this skill.

### Tool Calling

Use `vllm serve --help=all` to choose parser options supported by the installed
model and integration:

```bash
--tool-call-parser <parser> --enable-auto-tool-choice
```

Add a reasoning parser or chat template only when the installed integration
documents it for the selected model.
Known image-specific chat templates are under
`/opt/qti-aic/integrations/vllm/examples/tool_chat_template_*.jinja`.

## Triton integration

Triton is a separate serving layer. For Triton model repositories, QAIC,
ONNX Runtime, Python, or Triton's vLLM backend, use the
`cloud-ai-triton-serving` skill. Triton's vLLM backend is configured through a
Triton model repository and uses Triton endpoints; it is not the standalone
vLLM CLI or API documented here.

---

## Handoffs

- **REQUIRED SUB-SKILL:** To compile the model QPC through QEfficient, use
  `cloud-ai-llm-onboarding` before serving.
- **REQUIRED SUB-SKILL:** For container or device setup, use
  `cloud-ai-env-setup`.
- **REQUIRED SUB-SKILL:** To tune throughput or latency, use
  `cloud-ai-performance-optimization`.
- **REQUIRED SUB-SKILL:** If a serving change affects output accuracy, use
  `cloud-ai-accuracy-tuning`.
