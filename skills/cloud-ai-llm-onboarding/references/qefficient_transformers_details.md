# QEfficient Technical Details

Cloud AI containers provide the default QEfficient runtime at
`/opt/qeff-env/bin/python`. Select the package and artifact root before source
inspection or direct API use:

```bash
QEFF_HOME="${QEFF_HOME:-$HOME/.cache/qeff_models}"
export QEFF_HOME
mkdir -p "$QEFF_HOME"

QEFF_PYTHON="${QEFF_PYTHON:-/opt/qeff-env/bin/python}"
"$QEFF_PYTHON" -c "import QEfficient; print(QEfficient.__file__)"
```

The parent skill owns workflow selection and model entry points. Consult this
reference when an existing QEfficient implementation needs source-level
investigation. The top-level `QEFFAutoModelForCTC` class is the Wav2Vec2 CTC
entry point.

## Package Structure

```text
QEfficient/
├── base/               # Base models and ONNX transforms
├── transformers/       # Optimized Transformers layer replacements
├── exporter/           # ONNX export helpers
├── compile/            # QPC compilation helpers
├── cloud/              # CLI entry points
├── generation/         # Inference execution
├── diffusers/          # Image and video pipelines
├── finetune/           # LoRA and QLoRA infrastructure
├── peft/               # Adapter model support
├── customop/           # Custom NSP operator kernels
├── quantizers/         # Quantization implementations
├── spd/                # Speculative decoding
├── sampler/            # Sampling strategies
└── utils/              # Model registry and helpers
```

## ONNX Graph Transformation Pipeline

Models go through three stages before reaching `qaic-compile`.

### Stage 1: PyTorch Transforms

Implemented under `QEfficient/transformers/models/` and
`QEfficient/base/pytorch_transforms.py`.

| Transform | Purpose | Graph impact |
|---|---|---|
| `KVCacheTransform` | Replaces attention, decoder, and model classes with QEff variants | Uses flat KV inputs and outputs |
| RoPE optimization | Precomputes sin/cos caches and indexes them with `position_ids` | Removes dynamic sin/cos nodes |
| `CustomOpsTransform` | Replaces RMSNorm with `CustomRMSNormAIC` | Inserts the compiler-recognized RMSNorm op |
| `SplitGateUpWeightsTransform` | Splits fused gate/up projections and biases | Removes a device-side split and exposes concurrent MatMuls |
| `BlockingAttentionTransform` | Applies compile-time blocked-attention modes | Avoids full attention-score materialization at long context |
| `BlockedKVAttentionTransform` | Processes KV cache in blocks | Changes the attention gather/scatter structure |
| `SpDTransform` | Exposes multiple logits | Adds speculative-decoding outputs |
| `SamplerTransform` | Appends sampling operations | Adds output-side sampling nodes |
| `KVCacheExternalModuleMapperTransform` | Maps non-standard external modules | Makes custom forwards traceable |

### Stage 2: ONNX Transforms

Implemented in `QEfficient/base/onnx_transforms.py`.

| Transform | Purpose |
|---|---|
| `CustomOpTransform` | Registers `CustomRMSNorm` and `CtxScatter`/`CtxGather` operations |
| `RenameFunctionOutputsTransform` | Names KV retained-state outputs for compiler recognition |
| `FP16ClipTransform` | Clips FP32 initializers to the FP16 range |
| `SplitTensorsTransform` | Stores large tensors as external ONNX data |
| `AdapterWeightsToInputsTransform` | Converts LoRA initializers into dynamic graph inputs |

### Stage 3: Export-Time Graph Surgery

Implemented in `QEfficient/exporter/export_hf_to_cloud_ai_100.py`.

- Flattens nested `past_key_values` into graph I/O tensors.
- Sets dynamic axes for tokens and KV tensors.
- Generates custom-I/O configuration for KV-cache precision.

## Custom Operations

`QEfficient/customop/ctx_scatter_gather.py` provides position-indexed KV-cache
reads and writes. `rms_norm.py` provides the compiler-recognized RMSNorm
operation.

| Operation | Direction | Variant |
|---|---|---|
| `CtxScatter` / `CtxScatter3D` | Write KV at `position_ids` | Standard or 3D `(batch, seq, dim)` |
| `CtxGather` / `CtxGather3D` | Read KV with invalid-index masking | Standard or 3D |
| `CtxGatherBlockedKV` | Read KV in blocks | Large batch |
| `CtxScatterCB` / `CtxScatterCB3D` | Write KV | Continuous batching |
| `CtxGatherCB` / `CtxGatherCB3D` | Read KV | Continuous batching |
| `CtxGatherBlockedKVCB` | Read KV in blocks | Continuous batching and large batch |

## Export Modes

| Mode | KV retention | Use |
|---|---|---|
| Bertstyle | None | Encoder-style or short-context models |
| KV-style | `_RetainedState` outputs | Autoregressive decode and separate prefill/decode QPCs |

For KV-style export, custom-I/O configuration controls KV precision independently
from activations.
