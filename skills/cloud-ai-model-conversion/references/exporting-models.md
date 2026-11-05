# Exporting Models for Cloud AI 100

## Contents

- Supported ONNX and TensorFlow sources
- Source repository and licensing rules
- Autoregressive decoder and KV-cache export
- Layer-slice export for deep transformers
- Output verification and trimming
- Export pipeline, dependencies, and artifact generation

## 1. Supported Model Formats

### 1.1 ONNX

- Use a recent opset supported by your QAic SDK (for example opset 13–16; check SDK docs).
- Model requirements:
  - All non-batch tensor dimensions are fully defined (no -1 / None); the
    batch dimension may remain a named symbol and be resolved at compile time.
  - All weights are stored in `graph.initializer`.
  - Control flow is static or unrolled where practical.

Example export from PyTorch:

```python
import torch

# model: nn.Module; example_inputs: example tensor or tuple

torch.onnx.export(
    model,
    example_inputs,
    "model.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
```

### 1.2 TensorFlow

`qaic-compile` accepts these TensorFlow source formats directly:

- TensorFlow frozen GraphDef: `.pb` or `.pbtxt`.
- TensorFlow SavedModel: a directory containing `saved_model.pb`.

A Python TensorFlow model must be exported to one of these formats before
`qaic-compile` can consume it.
Try the direct path for these source formats. TensorFlow compilation requires
concrete input shapes and explicit output node names. Read [Compiling to QPC](compiling-to-qpc.md)
for the command and compile-log checks.

If the direct path rejects an operator, shape, signature, or TensorFlow export
version, convert the TensorFlow source to ONNX and use the ONNX export pipeline:
Install `tf2onnx` in the same Python environment as TensorFlow before running
the fallback (`python -m pip install tf2onnx`).

Example SavedModel → ONNX:

```bash
python -m tf2onnx.convert \
  --saved-model ./saved_model_dir \
  --output model.onnx \
  --opset 16
```

Once you have ONNX, follow the ONNX path below.

---

### Step 1: Export to ONNX

**Every Python file you write must begin with this copyright header (after the shebang line):**
```python
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
```

If no ONNX is present in the workspace, obtain the model using one of the
sources in the table below.

A source may be an ONNX file, a repository, a model card, or a model name.
Common vision model sources:

| Source | Models | Export method |
|---|---|---|
| **Torchvision** | ResNet, EfficientNet, ViT, DETR, Mask R-CNN | `torch.onnx.export` |
| **Huggingface** | ViT, DETR, Segformer, CLIP, SAM | `optimum` (preferred) or `transformers.onnx` |
| **Ultralytics** | YOLO v5/v8/v11 | `yolo export format=onnx` |
| **Detectron2** | Mask R-CNN, Faster R-CNN, Panoptic | `torch.onnx.export` via `torch.jit.trace` — **avoid Caffe2 export path**, it requires an old Python/Torch environment and Caffe2 op support may be deprecated |
| **MMDetection / MMSeg** | Large detection/segmentation zoo | `mmdeploy` (handles graph surgery) |
| **TF Model Garden** | EfficientDet, MobileNet, SSD | Export GraphDef/SavedModel, then try `qaic-compile`; `tf2onnx` fallback |
| **ONNX Model Zoo** | Pre-exported models | Ready to compile directly |
| **Community GitHub repo** | Models not in the sources above — e.g. LPRNet, CRAFT, DB-Net | `git clone` the repo inside the Dockerfile (add a `RUN git clone ...` layer), then import from it in the export script (`sys.path.insert(0, '/path/to/cloned/repo')`). **Never copy-paste or inline architecture code into the export script** — inline copies receive no upstream fixes, can silently diverge from the original weights, and inflate the repo with copied source. **Never reimplement the architecture from scratch** — reimplementation introduces silent bugs (wrong padding, wrong channel counts) and produces random weights that are not representative of real accuracy or performance. Training knowledge of the architecture may be used to understand the model and identify needed fixes (e.g. unsupported ops), but must never substitute for the cloned source — always import from the repo. Check the repo license before use: prefer BSD/MIT/Apache-2.0; flag GPL/AGPL/commercial-only licenses to the user before proceeding. |

For standard PyTorch/Torchvision models, use `torch.onnx.export` with opset 16, a symbolic batch axis (`dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}`), and a static spatial input shape — see **1.1 ONNX** above for the export example. For Huggingface models prefer `optimum` over `transformers.onnx` for newer architectures.

When using Ultralytics export, always pass `simplify=False` — Ultralytics uses `onnxslim` as its built-in simplifier, which has undocumented graph transformation passes that can produce graphs incompatible with `qaic-compile`. Run `onnxsim` separately in Step 2 instead. Do not substitute `onnxslim` for `onnxsim` in any step of this workflow for the same reason.

### Autoregressive decoder export (KV-cache models)

`torch.onnx.export` uses TorchScript tracing: it executes the `forward()` method once with the supplied inputs and records only the operations that actually run. Any Python `if/else` branch conditioned on a tensor value is resolved at trace time — only the taken branch is recorded; the other branch is silently dropped from the graph.

**Consequence for KV-cache decoders:** a wrapper that does `if step_idx == 0: past_kv = None` traces with `step_idx=0`, baking `past_kv=None` permanently into every compiled QPC. The KV input tensors appear in the graph signature but are never read by the attention ops. The QPC produces wrong output at every step > 0 with no error.

**Rule:** autoregressive decoder wrappers must not branch on `step_idx` or any tensor constant during export. Build the cache (or pass KV buffers) unconditionally. If step-0 behaviour must differ, export two separate ONNX models — one for prefill (step 0, no past KV) and one for decode (steps 1+, fixed-shape KV buffers) — and dispatch from the host.

**Verify KV wiring before compilation:**
```python
import numpy as np

# Run decoder QPC twice: once with zero KV, once with large random KV.
# If logits are identical, the KV inputs are dangling — the branch was traced away.
out_zero = decoder_session.run({"input_ids": ids, "encoder_hidden_states": enc_h,
                                 **{f"past_key_self_{i}": np.zeros((1,H,S,D), np.float32)
                                    for i in range(L)},
                                 **{f"past_val_self_{i}": np.zeros((1,H,S,D), np.float32)
                                    for i in range(L)}})
rng = np.random.default_rng(0)
out_rand = decoder_session.run({"input_ids": ids, "encoder_hidden_states": enc_h,
                                 **{f"past_key_self_{i}": rng.standard_normal((1,H,S,D)).astype(np.float32)*10
                                    for i in range(L)},
                                 **{f"past_val_self_{i}": rng.standard_normal((1,H,S,D)).astype(np.float32)*10
                                    for i in range(L)}})
diff = np.abs(out_zero["logits"] - out_rand["logits"]).max()
assert diff > 0.01, f"KV inputs are dangling (max diff {diff:.6f}) — branch was traced away; fix the export wrapper"
print(f"KV wiring OK (max diff {diff:.3f})")
```

Run this check on the ONNX Runtime CPU session before compiling. A diff of 0.000000 means the export is broken regardless of what cosine similarity says at step 0.

### Slice mode: 4-layer export for deep transformers

Models with 12+ homogeneous layers (causal LM decoders, VLM backbones, ViT/DiT, encoder-decoder pairs) take 10-15 min per compile. Export a 4-layer copy of the same model first and compile that — iterations drop to 2-3 min. Unsupported ops, graph-structure problems, and per-layer numerical bugs (attention, RoPE, RMSNorm, KV-cache scatter) all reproduce on layer 0, so slice mode surfaces both compile failures and accuracy failures.

Slice results say nothing about layers 4..N-1 or about scale-dependent behaviour (VTCM spill, MDP partitioning, FP16 accumulation across depth). **Compiling and validating the full model is mandatory regardless of slice results** — slice mode shortens the debug loop, it does not replace full-model validation.

Slice mode requires repeated homogeneous layers. For a heterogeneous graph, or a failure already localized to one operation, use `cloud-ai-accuracy-tuning` [cost-aware semantic bisection](../../cloud-ai-accuracy-tuning/references/output-comparison.md#cost-aware-semantic-bisection) instead — never cut an arbitrary layer count or node range that has no semantic boundary. If its [narrow CPU offload evidence gate](../../cloud-ai-accuracy-tuning/references/output-comparison.md#narrow-cpu-offload-evidence-gate) passes, export the remaining AI 100 suffix from a named semantic wrapper and validate the captured-boundary handoff before integrating it into the production pipeline.

Eligibility:

```python
cfg = AutoConfig.from_pretrained(model_id)
n = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
has_layers = hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList)
eligible = n is not None and n >= 12 and has_layers
```

Slice via config, not `.layers[:N]` — list slicing breaks tied weights and leaves `config.num_hidden_layers` inconsistent with the model:

```python
cfg.num_hidden_layers = 4
sliced_model = ModelClass(cfg)
full_sd = full_model.state_dict()

def keep(k):
    parts = k.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1]) < cfg.num_hidden_layers
    return True

sliced_model.load_state_dict({k: v for k, v in full_sd.items() if keep(k)}, strict=False)
```

Export the sliced model to its own ONNX filename so the slice and full-model artifacts never overwrite each other.

### Verify and trim ONNX outputs (required for all models)

After export and before onnxsim, print and verify the ONNX output list:

```python
import onnx
model = onnx.load("<model>.onnx")
for i, o in enumerate(model.graph.output):
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"  output[{i}] {o.name}: {dims}")
```

**How to identify which outputs to keep:** Match the ONNX output count to the original
framework model's output count. Run the framework model directly and count its outputs
before exporting — the ONNX should have exactly the same number. Any outputs beyond that
count were added by the exporter and should be trimmed.

```python
# PyTorch — count framework outputs before export
import torch
model.eval()
with torch.no_grad():
    out = model(dummy_input)
n_expected = len(out) if isinstance(out, (list, tuple)) else 1
print(f"Framework produces {n_expected} output(s)")
```

**⚠ Custom wrapper `nn.Module`:** The `n_expected` check above applies when you export
the original model's `forward()` directly. When you write a custom wrapper that bypasses
the original forward (e.g. to split at a pre-NMS or pre-decode boundary), the wrapper's
`return` statement owns the output list — the exporter cannot detect silently dropped heads.
Before writing `forward()`, list every prediction head in the original model and confirm
each is assigned in `__init__` and included in `return`. A dropped head produces a
silently incomplete ONNX that `predict.py` cannot recover in postprocess.

As a secondary check: if an output tensor name contains an intermediate layer name
(`silu_`, `cat_`, `add_`, backbone stage names, etc.) rather than a semantic name
(`output`, `logits`, `boxes`, `scores`), it is almost certainly an intermediate
activation accidentally exposed and should be trimmed.

**Trim extra outputs in the export script (not as a manual command):**

```python
KEEP = n_expected  # outputs your framework model actually produces
if len(model.graph.output) > KEEP:
    keep = list(model.graph.output[:KEEP])
    del model.graph.output[:]
    model.graph.output.extend(keep)
    onnx.save(model, "<model>.onnx")
    print(f"Trimmed to {KEEP} output(s): "
          + ", ".join(o.name for o in model.graph.output))
```

After trimming, regenerate `cpu_output.raw` from the trimmed ONNX — it must be generated
from the exact file that will be passed to `qaic-compile`, or the accuracy comparison will
have a byte-size mismatch.

QAic requires all shapes to be concrete at compile time — either baked as static dims in the graph, or left as named symbolic dims and resolved via `-onnx-define-symbol=name,value` (e.g. `batch_size`, `seq_len`). No `-1` or unresolved symbols may remain. The batch axis must always be a symbolic dim — never baked static (see `cloud-ai-model-conversion/references/qaic-model-requirements.md`). All weights must be in `graph.initializer`. The model will run in FP16 — always include `FP16ClipTransform` in the `run_pipeline()` call at the bottom of the export script. This clips initializers to the FP16 representable range and zeros denormals, preventing silent underflow warnings at `-convert-to-fp16` compile time. It runs after all model-specific transforms — see **Export Pipeline**, **“Step 5 — run the export pipeline”** below.

### Export Pipeline

**Step 1 — record SDK version (for MODEL_CARD):**
```bash
/opt/qti-aic/tools/qaic-version-util --platform
/opt/qti-aic/tools/qaic-version-util --apps
```

**Step 2 — create a venv and install required pip packages:**

> **Timeout warning:** `pip install` with large packages (torch, torchvision, onnxruntime,
> transformers) can take 10-30 minutes. Use `task_create`, not `bash`.

```bash
python3 -m venv /workspace/<model_name>/venv
. /workspace/<model_name>/venv/bin/activate
pip install <pkg1>==<version> <pkg2>==<version> ...
```

**HuggingFace models** (resnet, vit, detr, segformer, etc.) always need this base set — install it upfront, then add model-specific extras.

**Always use the CPU-only PyTorch wheel** — Cloud AI 100 uses QAIC, not NVIDIA CUDA. The default `pip install torch` pulls 2+ GB of unused CUDA packages; the CPU build is ~200 MB.

```bash
# torch/torchvision: CPU-only (required — never omit --index-url)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install transformers huggingface_hub onnx onnxruntime onnx-simplifier \
    onnxscript Pillow numpy
```

List every package the export script needs explicitly — do not rely on container
pre-installs. This makes the MODEL_CARD pip list a complete, self-contained recipe.

Prefer `*-headless` pip variants for packages with optional display dependencies
(e.g. `opencv-python-headless` not `opencv-python`).

**Step 3 - apt packages (last resort):**

Only when no pip alternative exists (e.g. some system shared libraries).
Must be installed inside a container.

To confirm you are inside a qaic container:**
```bash
env | grep QAIC_DOCKER_BUILD_TYPE 2>/dev/null && echo "qaic container confirmed" || \
ls /opt/venv_py310/lib/python3.10/site-packages/qaic 2>/dev/null && echo "qaic container confirmed" || \
echo "ERROR: not in a qaic container — start one before proceeding"
```
If not inside a container, see the `cloud-ai-env-setup` skill to start the appropriate
container with the workspace directory mounted to `/workspace/<model_name>`.

Prefer headless/minimal pip variants first — `opencv-python-headless`
eliminates the need for `libGL1` entirely. Record every apt package in the MODEL_CARD so a
fresh container can be reproduced identically.

**Step 4 — copy transform libraries into the workspace:**
```bash
cp "$(find -L /skills "$HOME/.agents" "$HOME/.claude" .agents .claude -path "*/cloud-ai-model-onboarding/scripts/base_onnx_transform.py" 2>/dev/null | head -1)" \
   /workspace/<model_name>/base_onnx_transform.py
cp "$(find -L /skills "$HOME/.agents" "$HOME/.claude" .agents .claude -path "*/cloud-ai-model-onboarding/scripts/base_pytorch_transform.py" 2>/dev/null | head -1)" \
   /workspace/<model_name>/base_pytorch_transform.py
```

`base_pytorch_transform.py` is used for pre-export PyTorch-level patches (GridSample, NonZero, custom forward methods). `base_onnx_transform.py` is imported by the export script — its `run_pipeline()` function drives the onnxsim + model-specific transforms + FP16Clip pipeline that produces `model_fp16ready.onnx`. Both should be copied; apply pre-export transforms first, then export, then call `run_pipeline()` at the end of the export script. Run `python base_pytorch_transform.py --list` to see available concrete transforms before writing inline patches.

**Standalone/rescue path:** For a pre-downloaded ONNX or an ONNX produced outside
the export script, run the copied `base_onnx_transform.py` directly instead of
writing a clamp script:

```bash
python /workspace/<model_name>/base_onnx_transform.py \
  /workspace/<model_name>/<model>.onnx \
  /workspace/<model_name>/<model>_fp16ready.onnx
```

The CLI applies `ClipUndefinedInputsTransform`, `OnnxSimTransform`,
`GlobalAveragePoolFusion`, and `FP16ClipTransform` in that order. Recompile from
the `_fp16ready.onnx` output and regenerate its matching CPU baseline.

**Step 5 — run the export pipeline:**
```bash
. /workspace/<model_name>/venv/bin/activate
cd /workspace/<model_name>

python3 export_<model_name>_onnx.py
```

The export script owns the full pipeline and must end with a `run_pipeline()` call and a `cpu_output.raw` generation block. **Every export script must follow this tail pattern:**

```python
from base_onnx_transform import (BaseOnnxTransform, ClipUndefinedInputsTransform,
                                  OnnxSimTransform, FP16ClipTransform,
                                  assert_symbolic_batch_axis, run_pipeline)
# Import specific PyTorch-level transform classes if pre-export patches are needed:
# from base_pytorch_transform import MethodPatchTransform, GridSampleTransform

# Define model-specific BaseOnnxTransform subclasses here (see
# cloud-ai-model-conversion/references/onnx-graph-surgery.md, "Replacing unsupported ops").
# If none are needed, omit this section entirely.
#
# class FixMyOp(BaseOnnxTransform):
#     @classmethod
#     def apply(cls, model):
#         ...
#         return model, changed

# ... rest of export code: framework goldens, torch.onnx.export → "<model_name>.onnx" ...

# Run the standard pipeline in one shot.
# ClipUndefinedInputsTransform removes opset 17+ Clip undefined inputs before onnxsim.
# OnnxSimTransform sorts, constant-folds, and cleans the graph.
# Add model-specific transforms between OnnxSimTransform and FP16ClipTransform.
# If no model-specific transforms are needed: omit them, keep the four standard ones.
run_pipeline("<model_name>.onnx", "<model_name>_fp16ready.onnx",
             [ClipUndefinedInputsTransform,
              OnnxSimTransform,
              # FixMyOp,   # ← insert model-specific transforms here, in dependency order
              FP16ClipTransform])
assert_symbolic_batch_axis("<model_name>_fp16ready.onnx")

# Generate cpu_output.raw from the fp16ready ONNX — the exact file qaic-compile receives.
# Resolve symbols only for this representative ORT feed shape. Do not mutate the ONNX.
import onnxruntime as ort
m = onnx.load("<model_name>_fp16ready.onnx")
symbol_values = {"batch_size": 1}  # add seq_len/height/width if they are symbolic

def concrete_dim_for_feed(dim):
    if dim.HasField("dim_value") and dim.dim_value > 0:
        return dim.dim_value
    if dim.HasField("dim_param") and dim.dim_param in symbol_values:
        return symbol_values[dim.dim_param]
    raise ValueError(f"No representative value for ONNX dim: {dim}")

input_shape = [concrete_dim_for_feed(d) for d in m.graph.input[0].type.tensor_type.shape.dim]
sess = ort.InferenceSession("<model_name>_fp16ready.onnx", providers=["CPUExecutionProvider"])
inp = np.fromfile("input.raw", dtype=np.float32).reshape(input_shape)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
for i, o in enumerate(out):
    o.tofile("cpu_output.raw" if len(out) == 1 else f"cpu_output_{i}.raw")

# Fidelity gate. The export script is where a low-fidelity ONNX is rejected, so it
# can never reach qaic-compile. Threshold and troubleshooting order: cloud-ai-accuracy-tuning
# references/onnx-baseline-validation.md, "Verify shapes and compare numerically"
# followed by "Troubleshoot ONNX export."
bad = []
for i, o in enumerate(out):
    fw = np.fromfile(f"framework_output_{i}.raw", dtype=np.float32)
    ort_out = np.ascontiguousarray(o, dtype=np.float32).ravel()
    assert fw.shape == ort_out.shape, f"output[{i}] shape {ort_out.shape} != framework {fw.shape}"
    cos = float(fw @ ort_out / (np.linalg.norm(fw) * np.linalg.norm(ort_out) + 1e-12))
    print(f"fidelity output[{i}]: cosine={cos:.8f}")
    if cos < 0.9999:
        bad.append((i, cos))
if bad:
    raise SystemExit(f"ONNX fidelity gate FAILED {bad}; repair the export before compiling")
```

> **All graph surgery belongs inside `run_pipeline()`.** Never apply ad-hoc transforms to the raw ONNX between export and `run_pipeline()` — express every fix as a `BaseOnnxTransform` subclass in the pipeline. If a raw-export issue prevents `ClipUndefinedInputsTransform` or `OnnxSimTransform` from running (e.g., a malformed node type not covered by the base), add a new `BaseOnnxTransform` subclass first in the pipeline rather than patching outside it.

---
