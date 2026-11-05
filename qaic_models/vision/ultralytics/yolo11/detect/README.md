# YOLO11 Detect

Ultralytics YOLO11 object detection on Qualcomm Cloud AI 100.

Use this guide to export YOLO11 to ONNX, compile Cloud AI 100 QPC binaries, and run either
ONNX Runtime CPU reference inference or QAIC/QPC inference.

## What this recipe runs

| Field | Value |
|---|---|
| Model | Ultralytics YOLO11 |
| Task | Object detection |
| Supported variants | `yolo11n`, `yolo11s`, `yolo11m` (default), `yolo11l`, `yolo11x` |
| Input | RGB image, letterboxed to `640x640`, NCHW float32 |
| Output | COCO-80 boxes, labels, and scores |
| Accelerator target | Qualcomm Cloud AI 100, FP16 QPC |

Select the size with `--variant` (weights default to `<variant>.pt` and download on first use). QPC output directories (`qpc_latency/`, `qpc_throughput/`) are variant-blind — export each variant to its own `--work-dir`.

## Generated files

The export command writes ONNX, QPC, sample input/output dumps, and optional annotated demo
images into the directory you pass as `--work-dir`.

Runtime APIs and the demo CLI accept either:

- the generated export work directory, such as `build/yolo11`; or
- a direct generated artifact path, such as `build/yolo11/yolo11m_fp16ready.onnx` or
  `build/yolo11/qpc_latency`.

Run export first, then pass that generated path to `--model`.

## Environment setup

Run commands from the Cloud AI SDK repository root.

Create a Python environment in a location outside the recipe source tree:

```bash
python3.12 -m venv env_yolo11
source env_yolo11/bin/activate
python -m pip install --upgrade pip
```

Install export dependencies when you need to generate ONNX/QPC artifacts:

```bash
python -m pip install "/path/to/cloud-ai-sdk-root[yolo11]"
```

Install runtime-only dependencies when artifacts already exist and you only need inference:

```bash
python -m pip install "/path/to/cloud-ai-sdk-root[yolo11-infer]"
```

For QAIC export, compile, or runtime, use an environment where the Cloud AI 100 SDK tools and
QAIC Python wheel are available. Quick checks:

```bash
python -c "import qaic_models; print('qaic_models import OK')"
python -c "import qaic; print('qaic import OK')"
/opt/qti-aic/tools/qaic-util -q
```

If you use the Cloud AI SDK container, run the setup, export, compile, and demo commands inside
that container so the SDK tools and devices are visible.

## Export and compile

Generate ONNX artifacts and both latency/throughput QPCs into `build/yolo11`:

```bash
python -m qaic_models.vision.ultralytics.yolo11.detect.export \
  --variant yolo11m \
  --work-dir build/yolo11 \
  --compile-latency \
  --compile-throughput
```

If `--image` is omitted, the exporter downloads Ultralytics' bundled `zidane.jpg` sample into
`--work-dir` and uses it to create `input.raw` and `cpu_output.raw`.

Use your own sample image for golden-output generation with:

```bash
python -m qaic_models.vision.ultralytics.yolo11.detect.export \
  --variant yolo11m \
  --work-dir build/yolo11 \
  --image /path/to/sample.jpg \
  --compile-latency
```

Use a local checkpoint instead of letting Ultralytics resolve `yolo11m.pt`:

```bash
python -m qaic_models.vision.ultralytics.yolo11.detect.export \
  --variant yolo11m \
  --weights /path/to/yolo11m.pt \
  --work-dir build/yolo11 \
  --compile-latency
```

Generated layout:

```text
build/yolo11/
  zidane.jpg                         # default sample image, if --image is omitted
  input.raw                          # preprocessed sample input
  cpu_output.raw                     # CPU golden output from final ONNX
  yolo11m.onnx                       # raw ONNX export
  yolo11m_fp16ready.onnx             # OnnxSim + FP16-clipped deployable ONNX
  qpc_latency/
    programqpc.bin                   # 16-core, batch 1, -mos=1 latency QPC
  qpc_throughput/
    programqpc.bin                   # 4-core, batch 1 throughput QPC
  triton-model-repo/                 # optional, if --triton-model-repo is used
    yolo11_detect_onnx/              # raw ONNX Runtime tensor serving
    yolo11_detect_qpc/               # raw QAIC/QPC tensor serving
```

Compile modes:

- `--compile-latency`: writes `qpc_latency` for low-latency runs.
- `--compile-throughput`: writes `qpc_throughput` for throughput experiments.

To regenerate from scratch:

```bash
rm -rf build/yolo11
python -m qaic_models.vision.ultralytics.yolo11.detect.export \
  --variant yolo11m \
  --work-dir build/yolo11 \
  --compile-latency
```

To also generate a Triton model repository, pass `--triton-model-repo`; the
full runbook is in the "Triton serving" section below.

## Triton serving

Generate a Triton model repository from the generated ONNX and latency QPC:

```bash
python -m qaic_models.vision.ultralytics.yolo11.detect.export \
  --variant yolo11m \
  --work-dir build/yolo11 \
  --compile-latency \
  --triton-model-repo build/yolo11/triton-model-repo
```

This writes two raw tensor-serving models:

- `yolo11_detect_onnx`: ONNX Runtime backend, input `images`, output `output0` with shape `[1, 84, 8400]`.
- `yolo11_detect_qpc`: QAIC backend, input `images`, output `output0` with shape `[1, 84, 8400]`.

The Triton models serve raw tensors. Run this recipe's Python demo or API when you need decoded boxes,
labels, scores, and annotated images.

The canonical raw tensor contract is declared in `model.yaml` under the output's `record_layout`.
It is shared by the ONNX artifact, direct QAIC execution, and the Triton QPC model:
`[cx, cy, width, height, 80 class scores]`, with records on axis 2 and class scores on
the half-open field slice `[4, 84]`. The direct demo applies recipe postprocessing; Triton
responses are not postprocessed detections. A functional run records the top 20 raw records
ranked by the declared `record_layout.ranking` rule in `logs/triton_client.log`. That diagnostic
view does not map class IDs to labels, apply a threshold or sigmoid, perform NMS, undo letterboxing,
or produce an annotated image.

Start Triton with the generated repository mounted. Replace `</path/to/cloud-ai-sdk root>` with this repository checkout path. The container command installs the recipe runtime and starts `tritonserver --model-repository=...`.

```bash
docker run -it --rm \
  --shm-size=4g \
  --network host \
  --mount type=bind,source=</path/to/cloud-ai-sdk root>,target=/cloud-ai-sdk \
  --device /dev/accel/ \
  ghcr.io/qualcomm/cloud_ai_triton_server:1.23.2.0 \
  bash -lc 'pip3 install "/cloud-ai-sdk[yolo11-infer,triton]" && \
    /opt/tritonserver/bin/tritonserver \
      --model-repository=/cloud-ai-sdk/build/yolo11/triton-model-repo'
```

Request the QAIC raw tensor model `yolo11_detect_qpc`:

```python
import numpy as np
import tritonclient.http as triton_http
from tritonclient.http import InferInput, InferRequestedOutput

from qaic_models.vision.ultralytics.common.preprocess import load_ultralytics_letterbox_image

image, _ = load_ultralytics_letterbox_image("build/yolo11/zidane.jpg")
image = np.ascontiguousarray(image.astype(np.float32))
client = triton_http.InferenceServerClient(url="localhost:8000")
infer_input = InferInput("images", image.shape, "FP32")
infer_input.set_data_from_numpy(image)
result = client.infer(
    model_name="yolo11_detect_qpc",
    inputs=[infer_input],
    outputs=[InferRequestedOutput("output0")],
)
output0 = result.as_numpy("output0")
print(output0.shape)  # [1, 84, 8400]
```

## Demo CLI

Run QAIC inference from the generated work directory:

```bash
python -m qaic_models.vision.ultralytics.yolo11.detect.demo \
  --image build/yolo11/zidane.jpg \
  --model build/yolo11 \
  --provider QAicExecutionProvider \
  --out build/yolo11/zidane_qaic.jpg
```

Run CPU reference inference from the same work directory:

```bash
python -m qaic_models.vision.ultralytics.yolo11.detect.demo \
  --image build/yolo11/zidane.jpg \
  --model build/yolo11 \
  --provider CPUExecutionProvider \
  --out build/yolo11/zidane_cpu.jpg
```

`--model` may also point directly to a generated artifact:

```bash
# CPU direct ONNX
python -m qaic_models.vision.ultralytics.yolo11.detect.demo \
  --image /path/to/image.jpg \
  --model build/yolo11/yolo11m_fp16ready.onnx \
  --provider CPUExecutionProvider

# QAIC direct QPC directory
python -m qaic_models.vision.ultralytics.yolo11.detect.demo \
  --image /path/to/image.jpg \
  --model build/yolo11/qpc_latency \
  --provider QAicExecutionProvider
```

The demo prints latency, detection count, and one row per detection:

```text
latency_ms: 8.700
num detections: 5
idx    score   label   x1    y1    x2    y2
0      0.9123  person  120.4 42.0  260.8 390.1
```

If `--out` is provided, detections are drawn on the original image and written to that path. If `--json-out` is provided, the prediction dict (`boxes`, `labels`, `scores`) is written there as JSON.

## Python API

CPU reference using the generated work directory:

```python
from qaic_models.vision.ultralytics.yolo11.detect import Model

model = Model("build/yolo11", provider="CPUExecutionProvider")
result = model.predict("build/yolo11/zidane.jpg", "build/yolo11/zidane_cpu.jpg")

print(result["latency_ms"])
print(result["prediction"])
```

QAIC/QPC runtime using the generated work directory:

```python
from qaic_models.vision.ultralytics.yolo11.detect import Model

model = Model.from_compiled("build/yolo11", provider="QAicExecutionProvider")
result = model.predict("build/yolo11/zidane.jpg", "build/yolo11/zidane_qaic.jpg")

print(result["latency_ms"])
print(result["prediction"])
```

Repeated inference should reuse the same model instance so the predictor warms up once:

```python
from qaic_models.vision.ultralytics.yolo11.detect import Model

model = Model.from_compiled("build/yolo11", provider="QAicExecutionProvider")

for image_path in ["image1.jpg", "image2.jpg"]:
    result = model.predict(image_path)
    print(image_path, len(result["prediction"]["boxes"]))
```

## Input and output details

Input images are converted to RGB, letterboxed to `640x640` with gray padding, scaled to
`0.0-1.0`, and sent as NCHW float32.

The result contains boxes in original image coordinates:

```python
{
    "prediction": {
        "boxes": [[x1, y1, x2, y2], ...],
        "labels": ["person", "bus", ...],
        "scores": [0.93, 0.88, ...],
    },
    "latency_ms": 8.7,
}
```

Default postprocessing uses confidence threshold `0.25` and NMS IoU threshold `0.45`.

Triton raw tensor-serving inputs and outputs use the preprocessed tensor contract directly:

- Input `images`: NCHW float32 tensor with shape `[1, 3, 640, 640]`, RGB letterboxed and scaled to `0.0-1.0`.
- Output `output0`: float32 tensor with shape `[1, 84, 8400]`. Use recipe postprocessing to convert this tensor into boxes, labels, and scores.

## Expected smoke result

For a successful export and compile, the default `zidane.jpg` demo should complete, print a
nonzero detection count, and write an annotated image when `--out` is supplied.

For the validated bus image used during recipe development, the expected result is 5 detections:
4 `person` and 1 `bus`. CPU and QAIC predictions should be close enough for the same labels and
similar boxes.

## Troubleshooting

### `qaic` cannot be imported

Activate a Cloud AI SDK environment that includes the QAIC Python wheel, then verify:

```bash
python -c "import qaic; print('qaic import OK')"
```

For this repository, the QAIC wheel is normally installed from:

```text
/opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

### No Cloud AI 100 device is visible

Check device status inside the same host/container where you run the demo:

```bash
/opt/qti-aic/tools/qaic-util -q
```

### `No generated ONNX artifact found`

The CPU provider needs `yolo11m_fp16ready.onnx` or a direct `.onnx` path. Run export first:

```bash
python -m qaic_models.vision.ultralytics.yolo11.detect.export \
  --variant yolo11m \
  --work-dir build/yolo11
```

### `No generated QPC found` or missing `programqpc.bin`

The QAIC provider needs `qpc_latency/programqpc.bin` by default. Compile it:

```bash
python -m qaic_models.vision.ultralytics.yolo11.detect.export \
  --variant yolo11m \
  --work-dir build/yolo11 \
  --compile-latency
```

### Unsupported variant or provider

Supported variants:

```text
yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
```

Supported demo providers:

```text
CPUExecutionProvider
QAicExecutionProvider
```

### Stale work directory or compile failure

Remove the generated directory and rerun export/compile:

```bash
rm -rf build/yolo11
python -m qaic_models.vision.ultralytics.yolo11.detect.export \
  --variant yolo11m \
  --work-dir build/yolo11 \
  --compile-latency
```

If compile still fails, verify the final ONNX exists, the QAIC SDK tools are in the container,
and `/opt/qti-aic/tools/qaic-util -q` can see a usable device.

## License and distribution

Ultralytics YOLO11 source and pretrained weights are AGPL-3.0. Review the upstream license
before distributing weights, exported ONNX files, QPC binaries, or applications that include
or depend on those artifacts.

Generated ONNX and QPC files may be subject to the upstream model/weights license.
