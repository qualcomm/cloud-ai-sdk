# AIMET Reference — Cloud AI 100

[AIMET (AI Model Efficiency Toolkit)](https://github.com/qualcomm/aimet) — Qualcomm's open-source PTQ/QAT toolkit.
Docs: https://qualcomm.github.io/aimet-pages/releases/latest/

---

## Installation

```bash
pip install aimet-onnx    # ONNX ModelProto — no PyTorch needed
pip install aimet-torch   # PyTorch nn.Module — more techniques available
```

Both install to the same version. **Requirements:** Python 3.10, Ubuntu 22.04, x86-64. GPU optional (Nvidia compute 5.2+).

Verify:
```python
import aimet_onnx; print(aimet_onnx.__version__)
import aimet_torch; print(aimet_torch.__version__)
```

---

## Config Files

`config_file` controls per-op quantization settings: symmetry, per-channel, which ops get FP16, supergroup fusions.

**Named aliases** (pass as string to `config_file=`):

| Alias | File | Use when |
|---|---|---|
| `"default"` | `default_config_per_channel.json` | Generic per-channel INT8 |
| `"htp_v68"` … `"htp_v81"` | `htp_quantsim_config_v*.json` | Qualcomm HTP (phone DSP) |

**Cloud AI 100 configs ship in the package but have no alias** — reference by path:

```python
import os, aimet_onnx
_cfg = os.path.join(os.path.dirname(aimet_onnx.__file__), "common/quantsim_config/{}")

AIC100_CONFIG         = _cfg.format("aic100_per_channel_config.json")
AIC100_BACKEND_CONFIG = _cfg.format("backend_aware_aic100_quantsim_config.json")
```

**`aic100_per_channel_config.json`** — recommended for CNN vision models:
- Per-channel quantization on Conv weights
- Asymmetric activations, symmetric params, bias not quantized
- LayerNorm + Gelu forced to FP16
- Supergroups: `Conv+ReLU`, `Add+ReLU`, `Conv+Clip`, `Sigmoid+Mul` (fused layers share one quantizer)

**`backend_aware_aic100_quantsim_config.json`** — more aggressive:
- All op types default to FP16 `supported_kernels`; AMP will select INT8 only where accuracy allows

---

## CPU-Only Constraints

All Cloud AI 100 AIMET work runs on CPU (no CUDA). Techniques vary enormously in cost:

| Technique | CPU cost | CPU guidance |
|---|---|---|
| CLE | Instant | Always run — no trade-off |
| BN Fold | Instant | Always run |
| Basic calibration | Seconds–minutes | 50–200 samples sufficient for range estimation; use 100–1000 real samples for production accuracy |
| SeqMSE | Minutes | Fine; reduce `num_batches` if slow |
| QuantAnalyzer | Minutes | Use ≤20 samples in eval callback |
| AMP Phase 1 | Minutes | Use ≤20 samples in phase1 callback |
| AdaRound | **Minutes–hours** | **Reduce `default_num_iterations` to 500–2000** (Adam gradient descent × iterations × layers) |
| AutoQuant | Hours | **Avoid on CPU** — run techniques individually |

**AdaRound iteration budget on CPU:**

| Model size | Layers | 10000 iter (GPU default) | 1000 iter (CPU recommended) |
|---|---|---|---|
| ResNet-50 | ~50 | ~30 min | ~3 min |
| YOLOv11m | ~200 | ~2 hr | ~12 min |
| ViT-large | ~300 | ~3 hr | ~18 min |

AdaRound with 500–2000 iterations on CPU still recovers most of the accuracy gain — the diminishing returns above ~2000 iterations on well-behaved models make the 5–10× reduction in time worthwhile.

```python
# CPU-appropriate AdaRound parameters
params = AdaroundParameters(
    data_set=calibration_inputs,
    num_batches=1,
    default_num_iterations=1000,   # not 10000
)
```

AMP and QuantAnalyzer are fast if the eval callback uses a small dataset:
```python
eval_callback_for_phase1 = lambda sess: accuracy_on_n_samples(sess, n=10)   # 10–20 samples
eval_callback_for_phase2 = lambda sess: accuracy_on_n_samples(sess, n=200)
```

**torch_qaic acceleration — transformers only:**
`torch_qaic` registers the `qaic` device and accelerates `mm`/`bmm`/`scaled_dot_product_attention` and most elementwise ops. For **transformer / ViT models** (`aimet-torch` + AdaRound), moving tensors to `.qaic()` could speed up the Adam optimizer steps 5–10×. For **CNN vision models** (ResNet, YOLO, MobileNet), it provides no benefit: without `torch.compile` (not supported in v0.1.0), PyTorch dispatches convolutions as `aten::conv2d` which has no QAIC kernel and falls back to CPU — the benchmark shows ResNet50 runs at 0.2 img/sec on QAIC vs normal CPU speed. Stick with `aimet-onnx` for CNN models.

---

## ONNX vs PyTorch — Which to Use

**Default: use `aimet-onnx`.** It works directly on the ONNX file you already have.

**Escalate to `aimet-torch`** only when the ONNX workflow doesn't reach your accuracy target — AutoQuant applies a more aggressive pipeline (CLE → BN fold → AdaRound → AMP automatically) but requires access to the original PyTorch checkpoint. If the checkpoint is unavailable, stay on the ONNX path and tune AdaRound iterations or add more calibration data instead.

## PTQ Techniques: ONNX vs PyTorch

| Technique | `aimet-onnx` | `aimet-torch` | Impact | Notes |
|---|:---:|:---:|---|---|
| CLE (Cross-Layer Equalization) | ✅ | ✅ | Medium | Apply before creating sim; best for MobileNet/EfficientNet |
| AdaRound | ✅ | ✅ | **High** | Adaptive weight rounding; highest single-technique gain |
| SeqMSE | ✅ | ✅ | Medium | Layer-wise MSE optimization; no labels needed |
| BN Fold | ✅ | ✅ | Low–Med | Run onnxsim first (already folds BN in most ONNX exports) |
| BN Re-estimation | ❌ | ✅ | Medium | Recalibrate BN stats; only if BN still present |
| AutoQuant | ❌ | ✅ | **High** | Automated pipeline: CLE→BN→AdaRound→AMP |
| AMP (Mixed Precision) | ✅ | ✅ | High | Per-layer INT8 vs FP16 sensitivity search |
| QuantAnalyzer | ✅ | ✅ | — | Sensitivity analysis; identifies FP16 candidates |
| SpinQuant / OmniQuant | ✅ (exp) | ✅ | LLMs only | Transformer weight outlier rotation — not for CNNs |

---

## Recommended ONNX Workflow (Cloud AI 100)

```python
import os
import onnx
import numpy as np
import aimet_onnx
from aimet_onnx.cross_layer_equalization import equalize_model
from aimet_onnx.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_onnx.quantsim import QuantizationSimModel

AIC100_CFG = os.path.join(os.path.dirname(aimet_onnx.__file__),
                          "common/quantsim_config/aic100_per_channel_config.json")

# 1. Load and simplify ONNX (run onnxsim first)
model = onnx.load("model_fp16ready.onnx")

# 2. CLE — equalizes Conv weight ranges (esp. beneficial for ResNet/MobileNet)
equalize_model(model)   # modifies in-place

# 3. AdaRound — adaptive weight rounding (highest-impact PTQ technique)
calibration_inputs = [{"pixel_values": np.fromfile(f, dtype=np.float32).reshape(1,3,224,224)}  # adjust shape for your model
                      for f in input_files]
params = AdaroundParameters(data_set=calibration_inputs, num_batches=1,
                            default_num_iterations=1000)   # 10000 is GPU default; 1000 for CPU
model = Adaround.apply_adaround(model, params, path="./output",
                                filename_prefix="model_adaround")

# 4. Final calibration sim
sim = QuantizationSimModel(
    model=model,
    param_type="int8",
    activation_type="int8",
    quant_scheme="tf_enhanced",
    config_file=AIC100_CFG,
)

def forward_pass(session):
    for inp in calibration_inputs:
        session.run(None, inp)

sim.compute_encodings(forward_pass)
sim.export(path="./output", filename_prefix="model")
```

**Before investing in AdaRound**, run a quick sensitivity scan (see Sensitivity Analysis) — if all per-layer drops are small, full INT8 may already be viable and AdaRound can be skipped. If accuracy is still insufficient after AdaRound, use the sensitivity results as FP16 candidates for AMP (see AMP section).

---

## Recommended PyTorch Workflow (Cloud AI 100)

PyTorch unlocks AutoQuant — automated CLE → BN folding → AdaRound → AMP pipeline. AutoQuant runs sensitivity analysis internally as part of AMP, so no separate QuantAnalyzer step is needed.

```python
import torch
from aimet_torch import QuantizationSimModel
from aimet_torch.auto_quant import AutoQuantWithAutoMixedPrecision
from aimet_torch.common.defs import QuantScheme, QuantizationDataType

dummy = torch.from_numpy(calibration_input)

# AutoQuant applies CLE → BN fold → AdaRound → AMP automatically
auto_quant = AutoQuantWithAutoMixedPrecision(
    model=model.eval(),
    dummy_input=dummy,
    data_loader=calibration_loader,   # unlabeled DataLoader
    eval_callback=eval_fn,            # fn(model) -> float accuracy score
    param_bw=8,
    output_bw=8,
)
# Optional: configure AMP candidates for INT8 vs FP16 search
auto_quant.set_mixed_precision_params(
    candidates=[
        ((8,  QuantizationDataType.int),   (8,  QuantizationDataType.int)),   # INT8
        ((16, QuantizationDataType.float), (16, QuantizationDataType.float)), # FP16
    ],
    allowed_accuracy_drop=0.01,
)
sim, accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=0.01)
sim.export(path="./output", filename_prefix="model")
```

---

## Sensitivity Analysis (QuantAnalyzer)

Available in **both ONNX and PyTorch**. Identifies which layers are most sensitive to INT8 quantization — use the output to build the FP16 candidate list for `qaic-compile -node-precision-info`.

### Which method to use

| Method | When | Cost |
|---|---|---|
| `perform_per_layer_analysis_by_enabling_quantizers(sim)` | Quick scan, no labeled data | Fast — ~5 layers/sec; just toggles quantizers on existing sim |
| `analyzer.analyze(...)` | Full report (MSE, histograms, min/max range) | Slow — creates its own sim + runs all sub-analyses |

**Avoid `check_model_sensitivity_to_quantization(sim)`** — it creates a raw ORT session that fails with `QcQuantizeOp is not a registered function/op` unless the AIMET custom op library is loaded. Skip it.

### Lightweight per-layer scan (recommended starting point)

Use cosine similarity between FP32 and INT8 outputs when no labeled validation set is available.

```python
import onnx, onnxruntime as ort, numpy as np, os, aimet_onnx
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.quant_analyzer import QuantAnalyzer

AIC100_CFG = os.path.join(os.path.dirname(aimet_onnx.__file__),
                          "common/quantsim_config/aic100_per_channel_config.json")

inp = np.fromfile("input.raw", dtype=np.float32).reshape(1, 3, 224, 224)
calibration_inputs = [{"pixel_values": inp}]

# FP32 reference for cosine-similarity eval
fp32_sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
fp32_out   = fp32_sess.run(None, {"pixel_values": inp})[0].flatten().astype(np.float64)
del fp32_sess

def cosine_eval(sess):
    out = sess.run(None, {"pixel_values": inp})[0].flatten().astype(np.float64)
    denom = np.linalg.norm(fp32_out) * np.linalg.norm(out)
    return float(np.dot(fp32_out, out) / denom) if denom > 0 else 0.0

def forward_pass(sess):
    for x in calibration_inputs:
        sess.run(None, x)

model = onnx.load("model.onnx")
sim = QuantizationSimModel(model=model, param_type="int8", activation_type="int8",
                           quant_scheme="tf_enhanced", config_file=AIC100_CFG)
sim.compute_encodings(forward_pass)

analyzer = QuantAnalyzer(
    model=model,
    dummy_input={"pixel_values": inp},
    forward_pass_callback=forward_pass,
    eval_callback=cosine_eval,
)

layer_scores = analyzer.perform_per_layer_analysis_by_enabling_quantizers(
    sim=sim,
    results_dir="./quant_analysis",
    filename="per_layer_sensitivity",
)
# Returns dict[layer_name] = eval_score (cosine similarity)
# Layers with the largest drop from 1.0 are FP16 candidates
ranked = sorted(layer_scores.items(), key=lambda kv: kv[1])
```

### Full analysis (when labeled data is available)

`analyze()` creates its own `QuantizationSimModel` internally — pass quant params, not a pre-built sim:

```python
analyzer.analyze(
    quant_scheme="tf_enhanced",
    default_param_bw=8,
    default_activation_bw=8,
    config_file=AIC100_CFG,
    results_dir="./quant_analysis",
)
# Writes:
#   per_layer_quant_enabled.json   — score when ONLY this layer is quantized
#   per_layer_quant_disabled.json  — score when this layer is EXCLUDED from INT8
#   per_layer_encoding_min_max_range.json
#   per_layer_mse_loss.json
```

### Interpreting results

For CNNs, the stem and first residual block typically show 4–10× larger drops than the rest of the network. If drops are uniformly small, try full INT8 before adding mixed-precision overhead.

---

## Automatic Mixed Precision (AMP)

AMP sensitivity-tests each layer group at candidate precisions and builds an accuracy vs bit-ops Pareto curve. **Candidates can include FP16**, making this directly useful for Cloud AI 100 mixed INT8+FP16.

```python
from aimet_onnx.mixed_precision import choose_mixed_precision
from aimet_onnx.common.defs import QuantizationDataType

candidates = [
    ((8,  QuantizationDataType.int),   (8,  QuantizationDataType.int)),   # INT8 — baseline
    ((16, QuantizationDataType.float), (16, QuantizationDataType.float)), # FP16 — fallback
]

pareto = choose_mixed_precision(
    sim=sim,
    candidates=candidates,
    eval_callback_for_phase1=lambda sess: fast_eval(sess),   # smaller dataset
    eval_callback_for_phase2=lambda sess: full_eval(sess),   # full dataset
    allowed_accuracy_drop=0.01,
    results_dir="./amp_results",
    clean_start=True,
    forward_pass_callback=lambda sess: calibrate(sess),
)
# pareto: list of (bitops, eval_score, QuantizerGroup, candidate) — use to build node_precision.yaml
```

`CANDIDATE_WITH_DTYPE = Tuple[Tuple[act_bw, act_dtype], Tuple[param_bw, param_dtype]]`

AMP respects `supported_kernels` in the config file. With `backend_aware_aic100_quantsim_config.json`, ops not listed in `supported_kernels` will be constrained to their declared supported precisions.

**Extracting FP16 layer names from the pareto result** for `node_precision.yaml`:

```python
# pareto entries: (QuantizerGroup, candidate, eval_score, bitops)
fp16_layers = set()
for qg, candidate, _, _ in pareto:
    act_bw, act_dtype = candidate[0]
    if act_dtype == QuantizationDataType.float and act_bw == 16:
        fp16_layers.update(qg.output_quantizer_map.keys())

import yaml
with open("node_precision.yaml", "w") as f:
    yaml.dump({"FP16NodeInstanceNames": sorted(fp16_layers)}, f)
```

---

## Version Compatibility & Format Conversion

`qaic-compile -external-quantization` requires **v0.5.0 YAML format**. AIMET 2.x supports multiple encoding formats; **use `encoding_version="0.6.1"`** when exporting — it is the most structurally similar to v0.5.0 and easiest to convert:

```python
sim.export(path="./output", filename_prefix="model", encoding_version="0.6.1")
```

### Converting v0.6.1 JSON → v0.5.0 YAML

v0.6.1 uses the same list-of-dicts structure as v0.5.0 but in JSON, with an extra `dtype` field and `offset` as a float. Strip `dtype` and cast `offset` to int:

```python
import json, yaml

with open("model.encodings") as f:
    v061 = json.load(f)

def to_v05_entries(entries_list):
    return [
        {
            "bitwidth":     e["bitwidth"],
            "is_symmetric": e["is_symmetric"],   # already "True"/"False" string
            "min":          e["min"],
            "max":          e["max"],
            "scale":        e["scale"],
            "offset":       int(e["offset"]),    # float -128.0 → int -128
        }
        for e in entries_list
    ]

v05 = {
    "activation_encodings": {k: to_v05_entries(v) for k, v in v061.get("activation_encodings", {}).items()},
    "param_encodings":      {k: to_v05_entries(v) for k, v in v061.get("param_encodings", {}).items()},
}
with open("model_v05.yaml", "w") as f:
    yaml.dump(v05, f)
```

> **Tensor name mismatch:** AIMET 2.x keys activations by ONNX node output path (`/layer1/conv/Conv_output_0`), which should match qaic-compile's expected names. Param keys use PyTorch module paths (`layer1.conv.weight`) — verify these match ONNX initializer names with `onnx.load("model.onnx").graph.initializer`.

## Format Comparison: v0.5.0 vs v0.6.1 vs v1.0.0/v2.0

| Aspect | v0.5.0 (**qaic-compile**) | v0.6.1 (AIMET 2.x recommended) | v1.0.0 / v2.0 |
|---|---|---|---|
| File type | YAML | JSON | JSON |
| Top-level keys | `activation_encodings`, `param_encodings` | `activation_encodings`, `param_encodings` | `version`, `quantizer_args`, `activation_encodings`, `param_encodings` |
| Tensor key | ONNX tensor / node name | ONNX tensor / node name | Module path (`model.layer1.conv.weight`) |
| Entry type | List of dicts | List of dicts (+ `dtype` field) | Dict of dicts (`enc_type`, `scale`, `offset`, …) |
| `scale` / `offset` | scalar float / int | scalar float / **float** (-128.0) | list (per-channel has multiple elements) |
| `is_symmetric` | `"True"` / `"False"` (string) | `"True"` / `"False"` (string) | `true` / `false` (bool) |

### v0.6.1 JSON Example

```json
{
  "activation_encodings": {
    "/layer1/conv/Conv_output_0": [
      {
        "bitwidth": 8, "dtype": "int",
        "is_symmetric": "True",
        "min": -17.225, "max": 17.090,
        "scale": 0.13457, "offset": -128.0
      }
    ]
  },
  "param_encodings": {
    "layer1.conv.weight": [
      {
        "bitwidth": 8, "dtype": "int",
        "is_symmetric": "True",
        "min": -21.938, "max": 21.767,
        "scale": 0.17139, "offset": -128.0
      }
    ]
  }
}
```

### AIMET v1.0.0 / v2.0 JSON Example

```json
{
  "version": "1.0.0",
  "quantizer_args": { "activation_bitwidth": 8, "param_bitwidth": 8 },
  "activation_encodings": {
    "/layer1/conv/Conv_output_0": {
      "enc_type": "PER_TENSOR_ASYMMETRIC",
      "dtype": "int", "bw": 8,
      "scale": [0.003922], "offset": [-128], "is_symmetric": false
    }
  },
  "param_encodings": {
    "layer1.conv.weight": {
      "enc_type": "PER_CHANNEL_SYMMETRIC",
      "dtype": "int", "bw": 8,
      "scale": [0.00152, 0.00189], "offset": [0, 0], "is_symmetric": true
    }
  }
}
```

---

## Encodings File Format (v0.5.0 — required by qaic-compile)

```yaml
activation_encodings:
  <tensor_name>:            # ONNX output tensor name or numeric ID string
  - bitwidth: 8
    is_symmetric: 'True'    # string, not bool
    min: -17.225431710481644
    max: 17.090858025243506
    scale: 0.13457368523813784
    offset: -128

param_encodings:
  <weight_or_bias_name>:    # ONNX initializer name
  - bitwidth: 8
    is_symmetric: 'True'
    min: -21.93815803527832
    max: 21.76676617562771
    scale: 0.17139185965061188
    offset: -128
```

| Field | Type | Notes |
|---|---|---|
| `bitwidth` | int | `8` for INT8; `32` for biases |
| `is_symmetric` | string | `"True"` or `"False"` — string, not bool |
| `min` / `max` | float | Calibration range |
| `scale` | float | Quantization scale |
| `offset` | int | `-128` signed INT8; `0` unsigned; `-2147483648` INT32 |

---

## Common Mistakes

| Mistake | Fix |
|---|---|
| Using `"htp_v81"` config on Cloud AI 100 | Use `aic100_per_channel_config.json` — HTP is Snapdragon DSP, not AIC100 |
| Skipping onnxsim before AIMET | Run `onnxsim model.onnx model_sim.onnx` first; AIMET can fail on unsimplified graphs |
| CLE after creating `QuantizationSimModel` | CLE must run on the plain ONNX model *before* sim creation — it modifies weights in-place |
| AdaRound with `num_iterations=10000` on CPU | Reduce to 1000–2000; still recovers most accuracy, 5–10× faster |
| Running AutoQuant on CPU | Too slow (hours) — run CLE, AdaRound, and AMP individually instead |
| Using PyTorch module paths as ONNX tensor names | Verify param names match ONNX initializer names before passing to qaic-compile |
| Calibrating on too few or synthetic inputs | Encoding ranges collapse; use ≥50–100 real images representative of the deployment distribution |

---

## SDK Example

Full YOLOv4 profile (4425 lines):
```
/opt/qti-aic/tools/qaic-pytools/configs/accuracy_evaluator/CV/yolo/yolov4/external_quant_profile_yolov4.yaml
```

`qaic-generate-histograms.py` in `/opt/qti-aic/tools/qaic-pytools/` generates compatible profiles without AIMET.
