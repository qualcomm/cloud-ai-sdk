---
model_name: [e.g. trocr-large-printed]
model_source: [HF slug, GitHub URL, torchvision:name, or torchhub:name — exact value of $MODEL_SOURCE]
task: [classification | detection | segmentation | ocr | pose | depth | face | tracking | embedding]
architecture: [e.g. ViT-Large encoder + TrOCR decoder]
precision: [fp16 | int8 | mixed]
tags: [comma-separated: task, arch family, ai100, and any notable properties e.g. transformer, two-stage, moe]
last_updated: [YYYY-MM-DD]
---

# Model Card: [MODEL_NAME] on Cloud AI 100

> **Generated:** [DATE]  
> **Onboarding Engineer:** [NAME]

---

## Model Summary

[2–3 sentences: what the model does, what it was trained on, what this card covers.
Example: "YOLOv8m is a real-time object detector trained on COCO 2017 (80 classes). This card
covers its onboarding to Cloud AI 100: ONNX export, FP16 compilation, accuracy validation,
and throughput benchmarking. The production configuration achieves X ms latency at N cores."]

---

## Model Information

| Field | Value |
|-------|-------|
| **Model Name** | [e.g., YOLOv8m, ResNet-50, ViT-B/16] |
| **Architecture** | [e.g., YOLOv8, ResNet-50, ViT-B] |
| **Parameters** | [e.g., 25.9M] |
| **Task** | [Classification / Detection / Segmentation / OCR / Pose / Depth] |
| **Training Dataset** | [e.g., COCO 2017, ImageNet-1K] |
| **Original Framework** | [PyTorch / TensorFlow / HuggingFace / Torchvision] |
| **Source / Repository** | [GitHub URL / HuggingFace card] |
| **License** | [e.g., MIT, Apache 2.0, GPL-3.0] |
| **Input Shape** | [e.g., 1×3×640×640 NCHW, float32] |
| **Inference Unit** | [e.g., image, crop, tile, 100-frame clip - one unit counted by inf/sec] |
| **Input Normalization** | [The value scheme the QPC expects, derived from input.raw — NOT the model name. One of: raw pixels [0,255] / scaled [0,1] (divide by 255 only) / ImageNet mean-std (divide by 255, then (x-mean)/std). State whether normalization is baked into the ONNX graph: if so, predict.py must NOT re-apply it. e.g. "scaled [0,1] — ImageNet mean/std baked into graph; predict.py divides by 255 only"] |
| **Output** | [e.g., boxes + scores (N×4 + N×80), class logits (1×1000)] |
| **ONNX Opset** | [e.g., 16] |

---

## Intended Use

**Designed for:**
- [e.g., Real-time object detection in images and video frames]
- [e.g., Edge or server batch inference on Cloud AI 100]

**Out of scope:**
- [e.g., Training or fine-tuning]
- [e.g., Classes outside the COCO 80-category set]
- [e.g., Video input with temporal reasoning]

---

## Known Limitations

- [e.g., Input must be exactly 640×640; aspect ratio padding may reduce accuracy on strongly non-square images]
- [e.g., INT8 quantization not validated on this card — FP16 only]
- [e.g., Batch size > 1 not tested]
- [If the Input Normalization scheme has a non-obvious constraint for predict.py authors (e.g. normalization is baked into the ONNX graph so predict.py must NOT re-apply it, or the model requires an unusual scaling), add a bullet here calling it out. Omit this bullet if the normalization is straightforward (e.g. standard caller-applied ImageNet mean/std or plain /255).]

---

## Performance

FP16, Cloud AI 100 unless otherwise noted. Build a benchmark ledger before
filling these tables. Configurator `results.csv` throughput is completed
inference units/sec and is copied directly. Direct `qaic-runner` `Inf/Sec` is
QPC executions/sec; record it separately and multiply by compiled batch for
completed inference units/sec. Do not derive compiled batch from runner
`BatchSize`; query the QPC input dimensions. `-a` controls independent
pipelines and is already aggregated in direct-runner `Inf/Sec`; `-S` controls
ExecObjs per activation. Use and record `-S 1` for standard onboarding
measurements.

For throughput selection, use only full-SoC rows where `cores × instances ==
NSP_TOTAL`. Rank those rows by aggregate completed `Throughput (inf/sec)` and
never select a partial-SoC row, even if its reported value is higher. Record the
winning core count, instance count, batch, and `NSP_TOTAL` together.

### Core / Activation / Batch / Compiler-Flag Sweep

| Source | Batch | Cores | Instances | MOS (`-mos`) | OLS (`-ols`) | Activations (`-a`) | QPC executions/sec | Throughput (inf/sec) | Execution latency P50 (ms) | Execution latency P99 (ms) |
|--------|-------|-------|-----------|--------------|--------------|----------------------|--------------------|----------------------|----------------------------|----------------------------|
| [configurator results.csv / qaic-runner] | [B] | [N] | [I] | [unset (compiler heuristic) / 1 / 2 / 4] | [unset (compiler heuristic) / 1 / 2 / 4] | [N] | [n/a for configurator / XX.X] | [XX.X] | [X.XX] | [X.XX] |
| [configurator results.csv / qaic-runner] | [B] | [N] | [I] | [unset (compiler heuristic) / 1 / 2 / 4] | [unset (compiler heuristic) / 1 / 2 / 4] | [N] | [n/a for configurator / XX.X] | [XX.X] | [X.XX] | [X.XX] |
| [configurator results.csv / qaic-runner] | [B] | [N] | [I] | [unset (compiler heuristic) / 1 / 2 / 4] | [unset (compiler heuristic) / 1 / 2 / 4] | [N] | [n/a for configurator / XX.X] | [XX.X] | [X.XX] | [X.XX] |

**Selected configuration:** [N] cores × [I] instances, batch [B] - [build mode: latency or throughput] - [reason]. For the throughput selection, `[N] × [I]` must equal `NSP_TOTAL=[N]`; partial-SoC rows are not eligible. Record omitted MOS/OLS as `unset (compiler heuristic)`, never `1`. Standard onboarding measurements use explicit `-S 1`. Document any different set size as a separately requested concurrency experiment.

### Production Benchmark

| Source | Precision | Cores | Instances | Batch | MOS (`-mos`) | OLS (`-ols`) | Activations (`-a`) | QPC executions/sec | Throughput (inf/sec) | Execution latency mean (ms) | Execution latency P50 (ms) | Execution latency P99 (ms) |
|--------|-----------|-------|-----------|-------|--------------|--------------|----------------------|--------------------|----------------------|-----------------------------|----------------------------|----------------------------|
| [configurator results.csv / qaic-runner] | FP16 | [N] | [I] | [B] | [unset (compiler heuristic) / 1 / 2 / 4] | [unset (compiler heuristic) / 1 / 2 / 4] | [N] | [n/a for configurator / XX.X] | [XX.X] | [X.XX] | [X.XX] | [X.XX] |
| [configurator results.csv / qaic-runner] | INT8 | [N] | [I] | [B] | [unset (compiler heuristic) / 1 / 2 / 4] | [unset (compiler heuristic) / 1 / 2 / 4] | [N] | [n/a for configurator / XX.X] | [XX.X] | [X.XX] | [X.XX] | [X.XX] *(if applicable)* |

### Stability

| Test | Result |
|------|--------|
| 60-second sustained run | [✅ Passed / ❌ Failed] |
| Device health after test (`qaic-util -q`) | [✅ All healthy / ❌ Issues: ___] |

---

## Accuracy

### FP16 vs CPU Baseline

N = [number] validation samples. CPU baseline: ONNX Runtime FP32 on `<model>_fp16ready.onnx`.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Cosine Similarity | [0.999X] | ≥0.99 | [✅ Pass / ❌ Fail] |
| MSE | [X.XXXX] | <1.0 | [✅ Pass / ❌ Fail] |

### INT8 Quantization (if applied)

| Metric | FP16 | INT8 | Degradation |
|--------|------|------|-------------|
| Cosine Similarity | [X.XXXX] | [X.XXXX] | [±X.XXXX] |
| Top-1 Accuracy | [XX.X%] | [XX.X%] | [±X.X%] |
| Top-5 Accuracy | [XX.X%] | [XX.X%] | [±X.X%] |

---

## Generating This Model

QPC binaries must be built from source. Follow the steps below.

### Step 1 — Export to ONNX

Environment used for this card:

| Field | Value |
|-------|-------|
| **Platform SDK** | `[/opt/qti-aic/tools/qaic-version-util --platform]` |
| **Apps SDK** | `[/opt/qti-aic/tools/qaic-version-util --apps]` |
| **pip packages** | `[e.g. torch==2.1.0 transformers==4.40.0 onnx==1.16.0 onnxruntime==1.18.0 onnxsim==0.4.36 optimum==1.21.0]` |
| **apt packages** | `[e.g. libgl1]` — or: none *(prefer headless pip variants first)* |

> **If `apt packages` is non-empty**, run the export inside a qaic container — apt requires it:
> ```bash
> docker run --rm -it --shm-size=2gb --network host \
>   -v /your/workspace:/workspace --device /dev/accel/ \
>   ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:1.23.2.0 bash
> ```
> Full image list: https://github.com/qualcomm/cloud-ai-containers

#### Framework Source Patches (if required)

If the export required patching framework source (e.g., replacing dynamic ops generated inside
the framework's own export path), record the patch metadata here:

| Field | Value |
|-------|-------|
| **Framework** | `[e.g., detectron2]` |
| **Pinned SHA** | `[e.g., 3566c6b]` |
| **Patch file** | `patches/<framework>_onnx_export.patch` |
| **Reason** | `[e.g., NonZero in level-assignment path, dynamic topk in proposals]` |

See `cloud-ai-model-conversion/references/onnx-graph-surgery.md`, "Source-Level Framework Patches", for patch generation and maintenance guidance.

```bash
# Install apt packages (if any — requires a qaic container):
apt-get install -y [apt-packages]

# Create venv and install all required packages:
python3 -m venv /workspace/<model_name>/venv
. /workspace/<model_name>/venv/bin/activate
pip install [pip-packages]

# Apply framework source patches (if any):
git clone <framework-repo> /workspace/<model_name>/src/<framework>
cd /workspace/<model_name>/src/<framework> && git checkout <sha>
git apply /workspace/<model_name>/patches/<framework>_onnx_export.patch
pip install -e /workspace/<model_name>/src/<framework>

# Run export pipeline:
cd /workspace/<model_name>
python3 export_<model_name>_onnx.py
python3 -m onnxsim <model_name>.onnx <model_name>_sim.onnx
python3 base_onnx_transform.py <model_name>_sim.onnx <model_name>_fp16ready.onnx

# Regenerate cpu_output.raw from the fp16ready ONNX:
python3 -c "
import onnx, onnxruntime as ort, numpy as np
m = onnx.load('<model_name>_fp16ready.onnx')
shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
sess = ort.InferenceSession('<model_name>_fp16ready.onnx', providers=['CPUExecutionProvider'])
inp = np.fromfile('input.raw', dtype=np.float32).reshape(shape)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
out[0].tofile('cpu_output.raw')
print('cpu_output.raw shape=' + str(out[0].shape))
"
```

**Artifacts produced:**

| File | Description |
|------|-------------|
| `<model>.onnx` | Raw ONNX with model-specific graph surgery applied |
| `<model>_fp16ready.onnx` | After run_pipeline() (ClipUndefinedInputs + OnnxSim + GlobalAveragePoolFusion + FP16Clip); input to `qaic-compile` |
| `input.raw` | Reference input (float32) |
| `framework_output_0.raw` | PyTorch/TF golden output |
| `cpu_output.raw` | ORT CPU baseline on `<model>_fp16ready.onnx` |

### Step 2 — Compile to QPC

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=<model_name>_fp16ready.onnx \
  -convert-to-fp16 \
  -aic-hw -aic-hw-version=ai100 \
  -aic-num-cores=[N] \
  -onnx-define-symbol=batch_size,[B] \
  [-onnx-define-symbol=<symbol>,<value> ...] \
  [-mos=N] \
  [-ols=N] \
  [-multicast-weights] \
  -aic-binary-dir=./qpc_[N]c
```

| Flag | Value | Rationale |
|------|-------|-----------|
| `-aic-num-cores` | [N] | [Full card / partial for pipeline] |
| `-onnx-define-symbol=batch_size` | [B] | [Batch size compiled into QPC; e.g. 1 for latency, 4/8 for throughput] |
| `-onnx-define-symbol=<symbol>` | [value] | [Repeat for each additional ONNX symbol: seq_len, height, width, etc.] |
| `-mos` | [unset (compiler heuristic) / 1 / 2 / ...] | [Measured sweep result] |
| `-ols` | [unset (compiler heuristic) / 1 / 2 / 4] | [Measured sweep result; use unset when the heuristic wins, depth-first is active, spatial ops are irrelevant, or explicit candidates regress/fail] |
| `-multicast-weights` | [enabled / omitted] | [Reduced DDR traffic] |

### Step 3 — Run Inference

```bash
python3 /workspace/<model_name>/predict.py \
  --input /workspace/<model_name>/demo_input.jpg \
  --output /workspace/<model_name>/demo_output.jpg
# stdout: {"prediction": ..., "latency_ms": X.X}
```

**Output format:**
- [fill in after predict.py smoke test — e.g.:
  detection: `{"boxes": [[x1,y1,x2,y2], ...], "labels": [...], "scores": [...]}`
  classification: `{"class": "tabby cat", "index": 281, "confidence": 0.92}`
  OCR/ALPR: `"ABC1234"`]

---

## Technical Details

### Graph Surgery Applied

| Modification | Tool / Method | Reason |
|--------------|---------------|--------|
| onnxsim simplification | `OnnxSimTransform` via `run_pipeline()` | Topological sort, constant-fold, dead-node removal |
| FP16 weight underflow clamp | `FP16ClipTransform` via `run_pipeline()` | Prevent zero conversion for weights < 5.96e-8 |
| [Replaced dynamic NonZero] | [`BaseOnnxTransform` subclass in export script] | [Replaced with static mask for on-device execution] |
| [Eliminated redundant Transposes] | [`BaseOnnxTransform` subclass in optimize_graph.py] | [NCHW layout optimization] |

### Unsupported Ops (if any)

| Op | Domain | Resolution |
|----|--------|------------|
| [Op Name] | [custom_domain] | [Custom HVX kernel / CPU fallback / Graph rewrite] |
| [None — all ops supported] | — | — |

### Suggested Base Transforms

Model-specific transforms written for this onboarding that appear broadly applicable and are candidates for promotion to `base_onnx_transform.py`:

| Transform class | Op / pattern targeted | Why it generalises |
|----------------|----------------------|-------------------|
| [None] | — | — |

> Fill this in if you wrote a `BaseOnnxTransform` subclass that fixes a pattern likely to appear in other models (e.g. constant-folding a parameter input that is always static in practice, an opset-compatibility shim, a cleanup that onnxsim misses). Leave as "None" if all transforms are model-specific. This section is reviewed by the skills maintainer — do not modify `base_onnx_transform.py` directly from within a job.

### Custom I/O (if any)

| I/O | Layout | Precision |
|-----|--------|-----------|
| [input] | [NCHW / NHWC] | [float32 / float16 / int8] |
| [output] | [NCHW / NHWC] | [float32 / float16 / int8] |

### Compiler Metrics

From first compile with `-aic-perf-metrics`. Lead with actionable signals.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AIC HMX MAC estimate** | [X.X]B | [Matrix compute on HMX] |
| **AIC non-HMX MAC estimate** | [0] | [All matrix ops mapped to HMX ✅] |
| **barriers per core** | [0] | [No inter-core sync stalls] |
| **DDR write / spill / DMAIssue** | [none / details] | [Actionable off-chip traffic signal] |
| **Not VTCM multicast** | [0] | [Weight multicast destinations stayed on-chip] |
| **VTCMMaxSize** | [X.X] MB | [Scratchpad high-water mark; not pass/fail] |

### PGQ Profile (if INT8)

| Field | Value |
|-------|-------|
| **Profile file** | `pgq_profile.yaml` |
| **Quantization schema** | [symmetric_with_uint8] |
| **Calibration method** | [KLMinimization / SQNR / MinMax] |
| **Calibration samples** | [100–500] |
| **Real data used** | [Yes / No (synthetic)] |

### Mixed Precision (if applicable)

| Field | Value |
|-------|-------|
| **FP16 nodes** | [conv0_out, layernorm1_out, …] |
| **Rationale** | [These layers showed cosine_sim < 0.99 in INT8 testing] |
| **Profile regenerated** | [Yes — required after precision changes] |

---

## Graph Optimization Roadmap

### Applied

| Transform | Nodes before → after | Bottleneck addressed | Status |
|-----------|----------------------|----------------------|--------|
| [FP16ClipTransform] | [N initializers clipped] | [FP16 underflow warnings] | ✅ Applied |
| [LayerNormFusion] | [9N → N nodes] | [barrier HVX XX%] | ✅ Applied |
| [None] | — | — | — |

Graph changes validated by accuracy re-check (cosine similarity: [X.XXXXX]).

### Future Work

| Transform | Bottleneck | Estimated impact | Complexity |
|-----------|-----------|-----------------|------------|
| [Channels-last re-export] | [aictranspose XX%] | [Eliminate N transpose nodes] | High — requires re-export |
| [INT8 quantization] | [sync HMX XX%] | [2–3× throughput] | High — requires calibration data |

---

## Artifacts

| Artifact | Path | Notes |
|----------|------|-------|
| **Export script** | `export_<model_name>_onnx.py` | Authoritative record of all graph surgery |
| **ONNX (fp16ready)** | `<model_name>_fp16ready.onnx` | Input to `qaic-compile` |
| **Reference input** | `input.raw` | Float32 reference input for accuracy comparison |
| **CPU baseline** | `cpu_output.raw` | ORT FP32 output on `<model_name>_fp16ready.onnx` |
| **predict.py** | `predict.py` | Demo inference script |
| **PGQ profile** | `pgq_profile.yaml` | If INT8 |

---

## Version History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| [2026-XX-XX] | 1.0 | Initial onboarding, FP16 | [Name] |
| [TBD] | 1.1 | INT8 quantization with PGQ | [Name] |
