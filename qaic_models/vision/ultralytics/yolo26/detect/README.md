# YOLO26 Detect

Ultralytics YOLO26 object detection on Qualcomm Cloud AI 100.

## What this runs

| Field | Value |
|---|---|
| Model | Ultralytics YOLO26 (n/s/m/l/x) |
| Task | Object detection |
| Input | RGB image letterboxed to `640x640` |
| Output | COCO-80 boxes, labels, and scores |

Supported variants: `yolo26n`, `yolo26s`, `yolo26m` (default), `yolo26l`, `yolo26x`. Select with `--variant` (weights default to `<variant>.pt` and download on first use). QPC output directories (`qpc_latency/`, `qpc_throughput/`) are variant-blind — export each variant to its own `--work-dir`.

## Generated files

Export writes files under `--work-dir`, including `yolo26m_fp16ready.onnx`, `qpc_latency/`,
`qpc_throughput/`, and optional Triton model repository entries when `--triton-model-repo`
is provided.

## Environment setup

Run commands from the Cloud AI SDK repository root.

Create a Python environment in a location outside the recipe source tree:

```bash
python3.12 -m venv env_yolo26
source env_yolo26/bin/activate
python -m pip install --upgrade pip
```

Install export dependencies when you need to generate ONNX/QPC files:

```bash
python -m pip install "/path/to/cloud-ai-sdk-root[yolo26]"
```

Install runtime-only dependencies when files already exist and you only need inference:

```bash
python -m pip install "/path/to/cloud-ai-sdk-root[yolo26-infer]"
```

For QAIC export, compile, or runtime, use an environment where the Cloud AI 100 SDK tools and
QAIC Python wheel are available. Quick checks:

```bash
python -c "import qaic_models; print('qaic_models import OK')"
python -c "import qaic; print('qaic import OK')"
/opt/qti-aic/tools/qaic-util -q
```

If you use the Cloud AI SDK container, run setup, export, compile, and demo commands inside that
container so the SDK tools and devices are visible.

## Export and compile

```bash
python -m qaic_models.vision.ultralytics.yolo26.detect.export \
  --variant yolo26m \
  --work-dir build/yolo26m \
  --compile-latency
```

To also generate a Triton model repository, pass `--triton-model-repo`; the
full runbook is in the "Triton serving" section below.

## Triton serving

Generate a Triton model repository from the generated ONNX and latency QPC:

```bash
python -m qaic_models.vision.ultralytics.yolo26.detect.export \
  --variant yolo26m \
  --work-dir build/yolo26m \
  --compile-latency \
  --triton-model-repo build/yolo26m/triton-model-repo
```

This writes two raw tensor-serving models:

- `yolo26_detect_onnx`: ONNX Runtime backend, input `images`, output `output0` with shape `[1, 300, 6]`.
- `yolo26_detect_qpc`: QAIC backend, input `images`, output `output0` with shape `[1, 300, 6]`.

The Triton models serve raw final-detection tensors. Each output row is `[x1, y1, x2, y2, confidence, class_id]`
in the model's letterboxed image frame.

The canonical raw tensor contract is declared in `model.yaml` under the output's `record_layout`.
It is shared by the ONNX artifact, direct QAIC execution, and the Triton QPC model:
`[x1, y1, x2, y2, confidence, class_id]`, with records on axis 1 and confidence as the
declared ranking field. The direct demo applies recipe postprocessing; Triton responses are not postprocessed detections. A functional run records the top 20 raw records ranked by the declared
`record_layout.ranking` rule in `logs/triton_client.log`. That diagnostic view does not map class
IDs to labels, apply a threshold, perform NMS, undo letterboxing, or produce an annotated image.

Start Triton with the generated repository mounted. Replace `</path/to/cloud-ai-sdk root>` with this repository checkout path. The container command installs the recipe runtime and starts `tritonserver --model-repository=...`.

```bash
docker run -it --rm \
  --shm-size=4g \
  --network host \
  --mount type=bind,source=</path/to/cloud-ai-sdk root>,target=/cloud-ai-sdk \
  --device /dev/accel/ \
  ghcr.io/qualcomm/cloud_ai_triton_server:1.23.2.0 \
  bash -lc 'pip3 install "/cloud-ai-sdk[yolo26-infer,triton]" && \
    /opt/tritonserver/bin/tritonserver \
      --model-repository=/cloud-ai-sdk/build/yolo26m/triton-model-repo'
```

Request the QAIC raw tensor model `yolo26_detect_qpc`:

```python
import numpy as np
import tritonclient.http as triton_http
from tritonclient.http import InferInput, InferRequestedOutput

from qaic_models.vision.ultralytics.common.preprocess import load_ultralytics_letterbox_image

image, _ = load_ultralytics_letterbox_image("build/yolo26m/bus.jpg")
image = np.ascontiguousarray(image.astype(np.float32))
client = triton_http.InferenceServerClient(url="localhost:8000")
infer_input = InferInput("images", image.shape, "FP32")
infer_input.set_data_from_numpy(image)
result = client.infer(
    model_name="yolo26_detect_qpc",
    inputs=[infer_input],
    outputs=[InferRequestedOutput("output0")],
)
output0 = result.as_numpy("output0")
print(output0.shape)  # [1, 300, 6]
```

## Demo CLI

```bash
python -m qaic_models.vision.ultralytics.yolo26.detect.demo \
  --image build/yolo26m/bus.jpg \
  --model build/yolo26m \
  --provider QAicExecutionProvider \
  --out build/yolo26m/out.jpg
```

`--out` draws the detections onto an annotated image; `--json-out <path>` writes the prediction dict (`boxes`, `labels`, `scores`) as JSON.

## Python API

```python
from qaic_models.vision.ultralytics.yolo26.detect import Model

model = Model.from_compiled("build/yolo26m", provider="QAicExecutionProvider")
result = model.predict("build/yolo26m/bus.jpg")
print(result)
```

## Input and output details

Images are letterboxed to `640x640` with gray padding and scaled to `0.0-1.0`. Results contain
`boxes`, `labels`, and `scores`; boxes are in original image coordinates.

Triton raw tensor-serving inputs and outputs use the preprocessed tensor contract directly:

- Input `images`: NCHW float32 tensor with shape `[1, 3, 640, 640]`, RGB letterboxed and scaled to `0.0-1.0`.
- Output `output0`: float32 tensor with shape `[1, 300, 6]` containing `[x1, y1, x2, y2, confidence, class_id]` rows.

## Expected smoke result

The default bus sample should produce person/bus style detections and a nonzero detection count.

## Troubleshooting

If `programqpc.bin` is missing, rerun export with `--compile-latency`. If `qaic` cannot be
imported or no devices are visible, run inside a Cloud AI SDK environment and check
`/opt/qti-aic/tools/qaic-util -q`.

## License and distribution

Ultralytics YOLO26 source and pretrained weights are AGPL-3.0. Review the upstream license
before distributing weights, exported ONNX files, or QPC binaries.
