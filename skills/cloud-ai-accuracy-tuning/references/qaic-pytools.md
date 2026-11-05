# qaic-pytools Accuracy Evaluation Reference

Tools live at `/opt/qti-aic/tools/qaic-pytools/`.

---

## Tools Overview

| Tool | Purpose | When to use |
|---|---|---|
| `qaic-acc-evaluator.py` | End-to-end accuracy pipeline (preproc → compile → infer → metric) | Measure top-1/top-5, mAP, mIoU against a dataset |
| `qaic-acc-analyzer.py` | Layer-wise accuracy debug; snoops per-layer outputs vs reference | Diagnose where INT8/FP16 diverges from FP32 |
| `qaic-compare.py` | Compare two finished evaluation run directories | Diff FP16 vs INT8 output distributions without re-running |
| `qaic-generate-histograms.py` | Activation/weight distribution visualization | Inspect quantization ranges; generate external quant profiles (see `aimet.md`) |

---

## qaic-acc-evaluator.py

### Evaluator invocation

```bash
python3 /opt/qti-aic/tools/qaic-pytools/qaic-acc-evaluator.py \
  -config /opt/qti-aic/tools/qaic-pytools/configs/accuracy_evaluator/CV/resnet/resnet50/resnet50_config.yaml
```

### Evaluator flags

| Flag | Notes |
|---|---|
| `-config <yaml>` | **Required.** Path to model config |
| `-platform-tag <tag>` | Run only platforms tagged with this (e.g. `aic_fp16`, `aic_int8`, `pgq_sweep`) |
| `-platform <name>` | Run only this platform (e.g. `onnxrt`, `aic`) |
| `-pipeline-start {preproc,compiled,infer}` | Resume from intermediate stage |
| `-pipeline-end {preproc,compiled,infer,postproc,metric}` | Stop early (useful for staged debugging) |
| `-device-id <id>` | Override device in config (e.g. `0`) |
| `-batchsize <n>` | Override batchsize in config |
| `-work-dir <path>` | Output directory (default `./qacc_temp`) |
| `-cleanup {end,intermediate}` | Delete intermediate files to save disk |
| `-enable-perf` | Capture throughput/latency alongside accuracy |
| `-set-global key:value ...` | Override any `globals:` variable in config. Multiple overrides in one flag: `-set-global count:20 calib:20`. |

### Common workflows

**Run FP16 accuracy only:**
```bash
python3 qaic-acc-evaluator.py -config model_config.yaml -platform-tag aic_fp16
```

**Run PGQ sweep (INT8 with multiple calibration schemes):**
```bash
python3 qaic-acc-evaluator.py -config model_config.yaml -platform-tag pgq_sweep
```

**Preprocess only (generate inputlist, no compile/infer):**
```bash
python3 qaic-acc-evaluator.py -config model_config.yaml -pipeline-end preproc
```

**Resume from existing compiled QPC:**
```bash
python3 qaic-acc-evaluator.py -config model_config.yaml -pipeline-start compiled -platform-tag aic_fp16
```

---

## qaic-acc-analyzer.py

Layer-wise snooping: adds intermediate outputs to the graph, runs both reference (onnxrt/interpreter) and AIC, compares per-layer.

### Analyzer invocation

```bash
python3 /opt/qti-aic/tools/qaic-pytools/qaic-acc-analyzer.py \
  -model model.onnx \
  -input-list input_list.txt \
  -precision fp16 \
  -snooping layerwise
```

### Analyzer flags

| Flag | Notes |
|---|---|
| `-model <onnx>` | **Required.** ONNX model path |
| `-input-list <txt>` | **Required.** Text file listing input `.raw` paths |
| `-input-info <name>,<shape>,<type>` | Override input tensor info |
| `-precision {int8,fp16,default}` | Precision for AIC compile |
| `-snooping {layerwise,oneshot,fp16sweep,amp}` | Snooping strategy |
| `-comparator <metric>` | Layer comparison metric (default `avg`) |
| `-tol-thresh <float>` | Global tolerance (default `1e-2`) |
| `-strategy {binary,sequential,pgq-sweep,none}` | Search strategy for finding sensitive layers |
| `-num-sensitive-nodes <n>` | How many top-sensitive nodes to report |
| `-reference {interpreter,onnxrt,tensorflow,caffe,none}` | Reference backend |
| `-aic-compile-args "<flags>"` | Extra flags forwarded to qaic-compile |
| `-work-dir <path>` | Output directory |

### Snooping modes

| Mode | Use |
|---|---|
| `layerwise` | Compare every layer; reports per-layer cosine similarity / avg error |
| `oneshot` | Single pass; compare final outputs only |
| `fp16sweep` | Iteratively promote layers to FP16; find the minimum set needed for the `SKILL.md` mixed-precision gate |
| `amp` | Automatic mixed precision search |

### fp16sweep workflow

`fp16sweep` is the recommended first step when INT8 accuracy is degraded. It systematically tries promoting each layer to FP16 (keeping the rest INT8) and ranks layers by sensitivity. Use the output to build the `-node-precision-info` list for mixed precision compilation.

**Step 1 — Build an input list file** (plain text, one `.raw` path per line — not IO-files JSON):

```bash
ls /workspace/<model>/calibration_data/imagenet/images/*.raw | head -20 > input_list_20.txt
```

20 images is sufficient; more improves ranking stability but increases runtime.

**Step 2 — Run fp16sweep:**

```bash
python3 /opt/qti-aic/tools/qaic-pytools/qaic-acc-analyzer.py \
  -model model_fp16ready_ir9.onnx \
  -input-list input_list_20.txt \
  -precision int8 \
  -snooping fp16sweep \
  -strategy sequential \
  -reference onnxrt \
  -aic-compile-args "aic-num-cores=8;quantization-schema-activations=asymmetric;quantization-calibration=MSE" \
  -silent \
  -work-dir /tmp/fp16sweep_out
```

**`-reference` is required.** Without it the tool runs in standalone oneshot mode (no comparison, no sensitivity ranking). Always pass `-reference onnxrt`.

**`-aic-compile-args` format:** semicolon-separated key=value pairs, **no leading dashes** (e.g. `"aic-num-cores=8;mos=1;quantization-schema-activations=asymmetric"`). Passing flags space-separated with dashes (e.g. `"-aic-num-cores=8 -mos=1"`) is the wrong format and will cause failures.

**Use `model_fp16ready_ir9.onnx`** (IR version downgraded to 9) — the analyzer's `check_model` rejects IR 10 with `Your model ir_version 10 is higher than the checker's (9)`. See **ONNX IR version** under **Prerequisites & Gotchas** below.

| Flag | Recommended value | Notes |
|---|---|---|
| `-snooping` | `fp16sweep` | Required |
| `-reference` | `onnxrt` | **Required** — without this the tool runs standalone (no ranking produced) |
| `-strategy` | `sequential` | 10 coarse iterations then layerwise on sensitive sub-graph; ~20–40 min with 20 images |
| `-strategy` | `binary` | Bisection search; faster but less interpretable |
| `-strategy` | `none` | Per-layer sensitivity report only — no automatic selection |
| `-aic-compile-args` | `"aic-num-cores=8;quantization-schema-activations=asymmetric;quantization-calibration=MSE"` | Semicolon-separated, no leading dashes |
| `-fp16sweep-layers` | e.g. `Conv,MatMul` | Restrict to specific op types to speed up the sweep |
| `-num-sensitive-nodes` | e.g. `10` | With `binary`: number of top-sensitive nodes to identify |
| `-tol-thresh` | `0.01` | Per-layer comparison tolerance (default `1e-2`) |

**Step 3 — Interpret output and build node_precision.yaml:**

The tool prints a per-layer sensitivity table ranked by accuracy impact. Layers at the top degrade INT8 accuracy most. Take the top-N layer output names and promote them to FP16 (start with N=10; reduce progressively until accuracy drops — fewer FP16 nodes preserves INT8 throughput):

```yaml
# node_precision.yaml
FP16NodeInstanceNames:
  - <layer_output_name_1>
  - <layer_output_name_2>
```

Recompile with `-node-precision-info=node_precision.yaml` using the
[mixed-precision override workflow](profile-guided-quantization.md#mixed-precision-override-workflow).

---

## qaic-compare.py

Post-hoc comparison of two completed evaluator runs.

> **Finding platform names:** After running `qaic-acc-evaluator.py`, the `work-dir` contains subdirectories named `plat<N>_<tag>` (e.g. `plat0_aic_fp16`, `plat1_onnxrt`). Use those directory names as the `-plat` and `-ref-plat` arguments.

```bash
python3 /opt/qti-aic/tools/qaic-pytools/qaic-compare.py \
  -work-dir ./qacc_temp_fp16 \
  -ref-work-dir ./qacc_temp_fp32 \
  -plat plat0_aic \
  -ref-plat plat1_onnxrt \
  -comparator rme \
  -tol-thresh 0.01
```

Comparators: `abs` (max absolute), `avg` (mean absolute), `rme` (relative mean error), `l1norm`, `l2norm`.

---

## Config YAML Structure

Configs are at `configs/accuracy_evaluator/CV/<model>/` and `configs/accuracy_evaluator/NLP/<model>/`.

### Required sections

```yaml
defaults:
    memory_pipeline: True          # file-I/O pipeline (recommended)

model:
    info:
        batchsize: 1

    globals:                       # variables; override with -set-global
        count: -1                  # -1 = all samples
        calib: -1

    dataset:
        name: ILSVRC2012           # or COCO2017, SQUAD, etc.
        path: /data/imagenet/      # local dataset root (must exist)
        inputlist_file: inputlist.txt
        annotation_file: ground_truth.txt
        calibration:
            type: dataset
            file: calibration.txt

    processing:
        preprocessing:
            transformations:
                - plugin: {name: resize, params: {dims: "256,256"}}
                - plugin: {name: crop,   params: {dims: "224,224"}}
                - plugin: {name: normalize, params: {norm: 1, channel_order: RGB,
                               means: {R: 123.68, G: 116.78, B: 103.94}}}
                - plugin: {name: convert_nchw}
                - plugin: {name: create_batch}

    inference-engine:
        model_path: path/to/model.onnx
        aic_device_ids: [0]
        platforms:
            - platform:
                  name: onnxrt
                  tag: ci
            - platform:
                  name: aic
                  precision: fp16
                  tag: aic_fp16
                  params: {aic-num-cores: 8}
            - platform:
                  name: aic
                  precision: int8
                  reuse_pgq: True
                  pgq_group: group1
                  tag: aic_int8
                  params:
                      aic-num-cores: 8
                      quantization-calibration: KLMinimization
                      quantization-schema-activations: symmetric_with_uint8
                      enable-channelwise: True

        inputs_info:
            - input_tensor_0:
                  type: float32
                  shape: ["*", 3, 224, 224]   # * = batchsize placeholder
        outputs_info:
            - output_0:
                  type: float32
                  shape: ["*", 1000]

    evaluator:
        metrics:
            - plugin:
                  name: topk           # topk | map_coco | miou
                  params: {kval: "1,5"}
```

### `pgq_group` / `reuse_pgq`

Set `pgq_group` to the same string on multiple platforms to share one compiled PGQ profile across them. `reuse_pgq: True` (default) skips recompile when the group's profile already exists.

---

## Supported Metrics

| Plugin name | Measures | Dataset |
|---|---|---|
| `topk` | Top-1 / Top-5 accuracy | ImageNet (ILSVRC2012) |
| `map_coco` | COCO mAP (bbox / mask) | COCO 2017 |
| `miou` | Mean IoU | Segmentation datasets |
| Custom | Any | Implement `qacc_memory_metric` base class |

---

## qaic-generate-histograms.py

Runs CPU inference on calibration inputs and collects per-layer activation and weight histograms. The output is a quantization profile YAML (`encodings.yaml`) that can be passed to `qaic-compile -external-quantization` as an alternative to `-load-profile` — see [External quantization profiles](profile-guided-quantization.md#external-quantization-profiles) and `aimet.md`.

Use this when you want to inspect activation distributions before choosing a quantization scheme, or to generate an external quantization profile without running AIMET.

### Histogram invocation

```bash
# Build input list (plain text, one .raw path per line — use calibration set only)
ls /workspace/<model>/calibration_data/imagenet/images/*.raw | sort | head -500 > calib_list.txt

python3 /opt/qti-aic/tools/qaic-pytools/qaic-generate-histograms.py \
  -model model_fp16ready.onnx \
  -input-list calib_list.txt \
  -reference onnxrt \
  -work-dir ./histograms_out
```

### Histogram flags

| Flag | Notes |
|---|---|
| `-model <path>` | **Required.** ONNX model path |
| `-input-list <txt>` | **Required.** Plain-text file, one `.raw` input path per line |
| `-reference {onnxrt,tensorflow,caffe}` | **Required.** Reference backend for CPU inference |
| `-work-dir <path>` | Output directory (default: current dir) |
| `-histogram-method {double_pass,rescale}` | `double_pass` (default): two-pass, more accurate range estimation. `rescale`: single-pass, faster, less precise for long-tail distributions |
| `-num-bins <n>` | Histogram bin count (default: 512); higher = finer range estimates, slower |
| `-enable-plotting` | Save PNG histogram plots for every activation and weight tensor to `<work-dir>/plots/` |
| `-onnx-symbol <sym:val>` | Bind symbolic ONNX dims (e.g. `-onnx-symbol batch_size:1`) |

### Input list format

Plain text file, one absolute path per line — **not** the IO-files JSON used by `qaic-compile` and `qaic-runner`. Use only the calibration split (images 0–499 or 0–749 depending on model type — never the test split):

```
/workspace/resnet-50/calibration_data/imagenet/images/00000.raw
/workspace/resnet-50/calibration_data/imagenet/images/00001.raw
...
```

### Output

`<work-dir>/` contains:

| File | Contents |
|---|---|
| `encodings.yaml` | Per-layer activation and weight min/max ranges (external quantization profile) |
| `plots/` | PNG histograms per tensor (only with `-enable-plotting`) |

`encodings.yaml` structure (AIMET-compatible format):

```yaml
activation_encodings:
  <tensor_name>:
    - bw: 8
      max: 6.2341
      min: -0.0001
      offset: 0
      scale: 0.024625
param_encodings:
  <weight_name>:
    - bw: 8
      max: 0.4821
      min: -0.4821
      ...
```

To compile INT8 using this file, follow [Compile INT8 with the
profile](profile-guided-quantization.md#compile-int8-with-the-profile), replacing
`-load-profile=pgq_profile.yaml` with
`-external-quantization=./histograms_out/encodings.yaml`. Use the same compiler
flags (cores, MOS, etc.) determined during the FP16 performance sweep.

### When to use vs PGQ

| Approach | Use when |
|---|---|
| PGQ (`-dump-profile`) | Standard path; compiler derives ranges from calibration data automatically |
| `qaic-generate-histograms.py` + `-external-quantization` | Need to inspect distributions first, or calibration data is in a format the compiler can't consume directly; also the output of AIMET CLE workflows |

---

## Prerequisites & Gotchas

- **Dataset must exist locally** — configs specify absolute paths; the tool does not download data.
- **Device required for AIC platforms** — physical Cloud AI 100 device or simulator must be running for `aic` platforms; `onnxrt` platform works without hardware.
- **`inputlist.txt` format** — one path per line, relative to `dataset.path`. Example: if `dataset.path` is `/data/imagenet/` and the file contains `val/00000.raw`, the full path is `/data/imagenet/val/00000.raw`. This file is generated automatically when running `-pipeline-end preproc`.
- **Dynamic shapes** — use `-onnx-symbol` (evaluator) or `-onnx-symbol` (analyzer) to bind symbolic dims: `-onnx-symbol batch_size=1`.
- **INT8 calibration file** — `calibration.txt` lists a subset of `inputlist.txt` used for PGQ profile generation; typically 100–500 entries.
- **`-cleanup intermediate`** saves significant disk but prevents using `qaic-compare.py` afterwards (inference outputs are deleted).
- **Non-interactive runs** — use `-silent` to suppress all confirmation prompts and run non-interactively.
- **ONNX IR version** — the evaluator bundles ONNX 1.14.1 which supports IR up to version 9. Models exported at opset 17+ use IR 10 and are rejected at `check_model` with `Your model ir_version 10 is higher than the supported IR version`. Downgrade to IR 9 before running the evaluator:
  ```python
  import onnx
  m = onnx.load("model_fp16ready.onnx")
  m.ir_version = 9
  onnx.save(m, "model_fp16ready_ir9.onnx")
  ```
  Use `model_fp16ready_ir9.onnx` as `model_path` in the config. The IR version field is metadata only — it does not affect compiled QPC behaviour.

---

## Image counts for pgq_sweep vs production

The evaluator's `count` and `calib` globals control how many test and calibration images are used. Use `-set-global` to override without changing the config file.

| Purpose | count | calib | Notes |
|---|---|---|---|
| pgq_sweep scheme comparison | 20 | 20 | Enough to rank schemas stably; does not give reliable absolute accuracy |
| Production profile + accuracy validation | full test split | full calib split | Use after sweep identifies the best schema |

```bash
# Fast scheme comparison sweep (20 images each)
python3 qaic-acc-evaluator.py -config model_config.yaml \
  -platform-tag pgq_sweep -silent -work-dir /tmp/sweep_out \
  -set-global count:20 calib:20

# Production run (full splits, defined in config)
python3 qaic-acc-evaluator.py -config model_config.yaml \
  -platform-tag pgq_sweep -silent -work-dir /tmp/prod_out
```

20 calibration images produce coarser quantization ranges than the full set, but the relative ranking between schemas is stable. Do not use a 20-image profile for the final shipped QPC — regenerate with the full calibration split after the sweep identifies the winner.

---

## Using CPU FP32 Reference Outputs for pgq_sweep

When there is no dataset ground truth, use pre-computed ONNX Runtime FP32 outputs as the accuracy reference for the cosine_sim plugin.

> **Exception:** to measure FP16→INT8 drift in isolation (FP16 QPC accuracy already validated), FP16 QPC outputs may be used as the reference — note this in the model card.

**Pre-compute CPU FP32 reference outputs:**

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

for i, inp_path in enumerate(test_input_paths):
    inp = np.fromfile(inp_path, dtype=np.float32).reshape(input_shape)
    outputs = sess.run(None, {input_name: inp})
    for j, out in enumerate(outputs):
        out.astype(np.float32).tofile(f"/path/to/cpu_ref_outputs/ref_{i:04d}_out{j}.bin")
```

**Config: `create_batch` preprocessing for pre-processed `.raw` inputs**

When inputs are already preprocessed `.raw` float32 files, `create_batch` is the only preprocessing step. The `dataset.name` field is a label only — no built-in handler is triggered when no `dataset.transformations` are specified:

```yaml
defaults:
    memory_pipeline: True
    custom_plugin_path: /path/to/qacc_plugins   # isolated dir with plugin .py files only

model:
    globals:
        count: 20    # override with -set-global count:N for fast sweeps
        calib: 20

    dataset:
        name: MyModel
        path: /path/to/inputs/
        inputlist_file: test_inputs.txt          # one absolute .raw path per line
        annotation_file: test_inputs.txt
        calibration:
            type: dataset
            file: calib_inputs.txt

    processing:
        preprocessing:
            transformations:
                - plugin: {name: create_batch}   # reads .raw files using inputs_info dtype/shape
        postprocessing:
            transformations: []

    inference-engine:
        inputs_info:
            - pixel_values:
                  type: float32
                  shape: ["*", 3, 512, 672]
        outputs_info:
            - logits:
                  type: float32
                  shape: ["*", 100, 92]
            - pred_boxes:
                  type: float32
                  shape: ["*", 100, 4]

    evaluator:
        metrics:
            - plugin:
                  name: cosine_sim
                  params:
                      ref_dir: /path/to/cpu_ref_outputs
                      num_outputs: 2
                      output_names: logits,pred_boxes
```

**`inputlist_file` resolution:** resolved relative to `dataset.path`. Use absolute paths inside the list file to avoid ambiguity.

---

## Custom Metric Plugins

Use `qacc_memory_metric` when no built-in metric matches the model.

**Plugin registration:** set `custom_plugin_path` in `defaults` to an isolated directory containing only plugin `.py` files. The evaluator imports every `.py` in that directory — mixed directories with export scripts or preprocessing code will cause import errors.

**Plugin interface:**
- `__init__(self, **kwargs)` — receives YAML `params` as keyword args
- `calculate(self, pipeline_item, **kwargs) → dict | None` — called per inference; `pipeline_item['data']` is a list of numpy arrays; `pipeline_item['input_idx']` is the sample index
- `finalize(self) → dict` — returns aggregate metrics after all inferences

**Cosine similarity plugin:**

```python
# qacc_plugins/cosine_sim_metric.py
import numpy as np, os
from qaic_pytools.qacc.plugin import qacc_memory_metric

class cosine_sim(qacc_memory_metric):
    def __init__(self, dataset=None, ref_dir="", num_outputs=1, output_names="output", **kwargs):
        super().__init__(**kwargs)
        self.ref_dir = ref_dir
        self.num_outputs = int(num_outputs)
        self.output_names = [s.strip() for s in str(output_names).split(",")]
        self.cos_sums = [0.0] * self.num_outputs
        self.counts   = [0]   * self.num_outputs

    def calculate(self, pipeline_item, **kwargs):
        idx, data = pipeline_item.get("input_idx", 0), pipeline_item["data"]
        results = {}
        for i in range(min(self.num_outputs, len(data))):
            ref_path = os.path.join(self.ref_dir, f"ref_{idx:04d}_out{i}.bin")
            if not os.path.exists(ref_path): continue
            q = data[i].flatten().astype(np.float64)
            r = np.fromfile(ref_path, dtype=np.float32).flatten().astype(np.float64)
            if q.shape != r.shape: continue
            denom = np.linalg.norm(q) * np.linalg.norm(r)
            cos = float(np.dot(q, r) / (denom + 1e-12)) if denom > 0 else 0.0
            self.cos_sums[i] += cos; self.counts[i] += 1
            name = self.output_names[i] if i < len(self.output_names) else f"out{i}"
            results[name] = round(cos, 6)
        return results or None

    def finalize(self):
        out, total, n = {}, 0.0, 0
        for i in range(self.num_outputs):
            if self.counts[i]:
                name = self.output_names[i] if i < len(self.output_names) else f"out{i}"
                mean = self.cos_sums[i] / self.counts[i]
                out[f"cosine_{name}"] = round(mean, 6); total += mean; n += 1
        if n: out["cosine_mean"] = round(total / n, 6)
        return out
```

Reference files must be named `ref_{idx:04d}_out{output_index}.bin` (raw float32).

---

## Known Issues

**NFS workspace: `fallocate` fails during PGQ profile generation**

The evaluator sets `TMPDIR` to a subdirectory of the work-dir internally, overriding the environment variable. On NFS mounts this causes the compiler's `fallocate` call to fail:

```
Error: Failed fallocate of /workspaces/.../tmpXXXXX/constdata-...
```

**Fix:** Set `-work-dir` to a local filesystem path (e.g. `/tmp/pgq_sweep_out`). The evaluator's TMPDIR is created inside the work-dir, so moving the work-dir off NFS resolves it.

```bash
python3 qaic-acc-evaluator.py -config model_config.yaml \
  -platform-tag pgq_sweep -silent \
  -work-dir /tmp/pgq_sweep_out    # local filesystem, not NFS
```

**`qaic-acc-analyzer.py` crashes with `UnboundLocalError: local variable 'cfg'`**

Misleading error — the real cause is an exception during config initialization. Check the full traceback. Common causes:

- `-input-list` file path does not exist inside the container (container and host paths differ)
- `-reference` flag omitted — `-reference none` is required even when no reference run is needed
- Graph topology unsupported by the snooping engine (e.g. SHA-decomposed attention graphs fail with `Node with id not found in map`)

**`fp16sweep` fails on non-standard graph topologies**

The `PatternMatcher` inside the snooping engine cannot traverse certain graph structures. Fall back to [manual node selection](profile-guided-quantization.md#select-sensitive-nodes).

**`qaic-acc-analyzer.py` rejects IR version > 9**

Same as the evaluator: pass `model_fp16ready_ir9.onnx`. See **ONNX IR version**
under **Prerequisites & Gotchas** above.
