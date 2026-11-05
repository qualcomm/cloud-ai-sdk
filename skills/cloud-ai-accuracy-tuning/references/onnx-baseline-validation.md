# ONNX Baseline Validation Reference

Run this workflow **before compiling or running any QPC.** A CPU ONNX baseline that diverges from the original framework will corrupt all downstream QAIC comparisons — passing QAIC accuracy numbers against a wrong baseline is meaningless.

Two parts:
- **Numerical fidelity**: does the ONNX produce the same numbers as the original framework?
- **Input pipeline sanity**: is preprocessing wired up correctly?

Both must pass before validating an FP16 QPC.

> **If original framework weights are unavailable**, use a second ONNX Runtime execution as reference — run the pre-simplification `model.onnx` against `model_fp16ready.onnx`. You lose the ability to catch export bugs, but can still catch simplification and FP16 transform regressions.

---

## Generate framework golden outputs

Run the original model on the reference input inside the **export container** and save raw outputs. `model.eval()` is required both here (for correct golden values) and at the time `torch.onnx.export` is called — forgetting it at export time causes BatchNorm and Dropout to behave differently from the golden.

> **If pre-export PyTorch transforms were applied during export** (e.g. `GridSampleTransform`, `MethodPatchTransform`): re-apply those same transforms before running the model here. The ONNX reflects the patched graph; golden outputs generated from the unpatched model will diverge at every patched op and make the fidelity check meaningless. The correct order is: `transform.apply()` → run model → save goldens → (same script context) `torch.onnx.export` → `transform.undo()`.

**PyTorch:**
```python
import torch, numpy as np

# Adjust model loading to match the repo's pattern
model = MyModel()
model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
model.eval()

inp = torch.from_numpy(np.fromfile("input.raw", dtype=np.float32).reshape(INPUT_SHAPE))
with torch.no_grad():
    out = model(inp)

# Normalise to a flat list — handles tensor, tuple, list, and dict outputs
if isinstance(out, dict):
    out = list(out.values())
if not isinstance(out, (tuple, list)):
    out = [out]
for i, o in enumerate(out):
    arr = o.numpy()
    arr.tofile(f"framework_output_{i}.raw")
    print(f"framework_output[{i}] shape={arr.shape} dtype={arr.dtype}")
```

**TensorFlow / Keras:**
```python
import tensorflow as tf, numpy as np
model = tf.keras.models.load_model("saved_model/")
inp = np.fromfile("input.raw", dtype=np.float32).reshape(INPUT_SHAPE)
out = model(inp, training=False)
if not isinstance(out, (list, tuple)):
    out = [out]
for i, o in enumerate(out):
    arr = o.numpy()
    arr.tofile(f"framework_output_{i}.raw")
    print(f"framework_output[{i}] shape={arr.shape} dtype={arr.dtype}")
```

Record the printed shapes and compare them against the ONNX output shapes before
performing any numerical comparison.

---

## Run ONNX Runtime on CPU

Use the **final pre-compilation ONNX** (after `onnxsim` and `FP16ClipTransform` — the same file passed to `qaic-compile`):

```python
import onnxruntime as ort, numpy as np

sess = ort.InferenceSession("model_fp16ready.onnx",
                            providers=["CPUExecutionProvider"])
inp = np.fromfile("input.raw", dtype=np.float32).reshape(INPUT_SHAPE)
feed = {sess.get_inputs()[0].name: inp}  # extend for multi-input models

outputs = sess.run(None, feed)
for i, o in enumerate(outputs):
    o.tofile(f"cpu_onnx_output_{i}.raw")
    print(f"cpu_onnx_output[{i}] shape={o.shape} dtype={o.dtype}")
```

---

## Verify shapes and compare numerically

**Shape check first.** Compare the framework shapes against the ONNX Runtime
shapes. A mismatch means the ONNX graph has silently transformed the output
tensor, so numerical comparison is meaningless until it is resolved. Check the
non-standard-op, tracing-versus-scripting, and opset guidance under
**Troubleshoot ONNX export** before proceeding.

```python
import numpy as np, glob

def compare(fw_path, onnx_path, label="output"):
    fw   = np.fromfile(fw_path,   dtype=np.float32)
    onnx = np.fromfile(onnx_path, dtype=np.float32)
    if fw.shape != onnx.shape:
        print(f"{label}: SHAPE MISMATCH fw={fw.shape} onnx={onnx.shape}")
        return
    cosine = np.dot(fw, onnx) / (np.linalg.norm(fw) * np.linalg.norm(onnx) + 1e-12)
    mse    = np.mean((fw - onnx) ** 2)
    maxabs = np.abs(fw - onnx).max()
    print(f"{label}: cosine={cosine:.6f}  MSE={mse:.2e}  max_abs={maxabs:.2e}")

for i, fw_path in enumerate(sorted(glob.glob("framework_output_*.raw"))):
    compare(fw_path, f"cpu_onnx_output_{i}.raw", label=f"output[{i}]")
```

**Pass criteria:**

| Metric | Threshold | Action if failing |
|--------|-----------|-------------------|
| Output shapes | must match | Resolve before numerical comparison using **Troubleshoot ONNX export** |
| Cosine similarity | ≥ 0.9999 | Follow **Troubleshoot ONNX export** |
| Max absolute error | < 1e-3 | If cosine also fails, use per-layer bisection. If cosine passes (≥ 0.9999), treat it as a single outlier and apply `FP16ClipTransform` in the export pipeline (see `cloud-ai-model-conversion/references/exporting-models.md`, "Export Pipeline") |

---

## Troubleshoot ONNX export

Work through these in order — stop at the first fix that achieves matching shapes and cosine ≥ 0.9999:

1. **`model.eval()` at export time** — confirm `model.eval()` was called before `torch.onnx.export`, not only before golden generation. BatchNorm running stats and Dropout are baked into the graph at export; training mode produces a different graph.

2. **`export_params=True` and `do_constant_folding=True`** — if either is missing or explicitly `False` in `torch.onnx.export`, weights are absent or constant subgraphs remain unevaluated. Both cause divergence that `onnxsim` cannot repair.

3. **Patch non-standard op usage before export** — list op types and identify anything outside the standard ONNX opset:
   ```python
   import onnx
   m = onnx.load("model.onnx")
   print(sorted({n.op_type for n in m.graph.node}))
   ```
   More importantly: **inspect the PyTorch model for ops that map ambiguously to ONNX**. A common example is `MaxPool3d` applied to a 4D input (N, C, H, W) — PyTorch accepts it but ONNX `MaxPool` with 3D kernel parameters expects a 5D input; the tracer emits syntactically valid ONNX that produces wrong output shapes. Patch these in the model *before* export (replace `MaxPool3d` with `MaxPool2d` where the input is 4D) rather than relying on opset-level behavior to preserve the intent.

4. **Tracing vs scripting** — if exported with `torch.onnx.export` in trace mode and the model contains data-dependent control flow, switch to `torch.jit.script` first, then export.

5. **Opset version** — try opset 17 if you used 16, or vice versa. `Resize`, `RoiAlign`, and several other ops changed semantics between opsets.

6. **Dynamic vs static shapes** — re-export with fully static input shapes and verify `onnxsim` constant-folds any remaining shape computations.

7. **Input dtype mismatch** — check the ONNX graph input dtype:
   ```python
   import onnx
   m = onnx.load("model.onnx")
   print(m.graph.input[0].type.tensor_type.elem_type)  # 1=float32, 10=float16
   ```
   Cast `input.raw` to the correct dtype if it doesn't match.

8. **Per-layer bisection** — use `qaic-acc-analyzer.py -snooping layerwise` (see `qaic-pytools.md`) or the `add_all_nodes_as_outputs()` helper below to expose all intermediate tensors, run ONNX Runtime on the debug model, and compare layer-by-layer to find the first diverging node. Focus the fix at that layer.

```python
import onnx
from onnx import helper

def add_all_nodes_as_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    existing = {o.name for o in model.graph.output}
    value_info_map = {vi.name: vi for vi in model.graph.value_info}
    for node in model.graph.node:
        if node.op_type == "Constant":
            continue
        for output_name in node.output:
            if output_name and output_name not in existing:
                vi = value_info_map.get(output_name)
                if vi is None:
                    vi = helper.make_tensor_value_info(output_name, 1, None)
                model.graph.output.append(vi)
                existing.add(output_name)
    return model
```

After each fix, rerun ONNX Runtime and the shape/numerical comparison until shapes
match and cosine is at least 0.9999. Record any fix in the model card under
"Model Modifications."

---

## Smoke-test the input pipeline

After the shape and numerical gate passes, verify preprocessing end to end. Wrong
preprocessing silently produces plausible-looking but wrong outputs: high
confidence on wrong classes or geometrically consistent but semantically wrong
detections.

**Common preprocessing bugs — in order of frequency:**

1. **Normalization mismatch** — does the ONNX graph apply `/255` and mean/std internally, or does it expect the caller to normalise?
   - Symptom: all output scores near 1.0 (raw `[0,255]` fed to a `[0,1]` model) or near 0.0 (opposite).
   - Search the graph for normalisation ops connected to the input node:
     ```python
     import onnx
     model = onnx.load("model.onnx")
     input_names = {i.name for i in model.graph.input}
     norm_ops = {"Div", "Mul", "Sub", "Add", "Normalize",
                 "BatchNormalization", "InstanceNormalization"}
     for node in model.graph.node:
         if node.op_type in norm_ops:
             if any(i in input_names for i in node.input):
                 print(f"BAKED IN: {node.op_type} inputs={list(node.input)}")
     ```

2. **Resize strategy** — square resize vs aspect-preserving letterbox with padding?
   - Detection and segmentation models are typically trained with letterbox (scale to fit, pad remainder). Squash-resizing distorts aspect ratio and shifts anchor positions.
   - Check the original training/export code for the resize mode and fill value.

3. **Channel order** — RGB vs BGR?
   - PIL gives RGB; OpenCV gives BGR. A channel swap causes a systematic score shift without crashing.

4. **Double-applied activation** — sigmoid, softmax, or normalisation applied twice?
   - Many exporters (TorchVision, Ultralytics, HuggingFace) bake the final activation into the ONNX graph.
   - Symptom: scores cluster near 0.5 (double sigmoid) or all sum to 1.0 per row (double softmax).
   - Check output-side nodes:
     ```python
     import onnx
     model = onnx.load("model.onnx")
     activation_ops = {"Sigmoid", "Softmax", "LogSoftmax",
                       "HardSigmoid", "HardSwish", "Relu", "Clip"}
     output_names = {o.name for o in model.graph.output}
     for node in model.graph.node:
         if node.op_type in activation_ops:
             if any(o in output_names for o in node.output):
                 print(f"BAKED IN: {node.op_type} -> {node.output}")
     ```

5. **Wrong output container type** — `ort.InferenceSession.run()` returns a **list**; `qaic.Session.run()` returns a **dict** keyed by output name. Extract tensors explicitly before postprocessing.
