# YOLO26 Classify

Ultralytics YOLO26 image classification on Qualcomm Cloud AI 100.

## What this runs

| Field | Value |
|---|---|
| Model | Ultralytics YOLO26 classification (n/s/m/l/x) |
| Task | Image classification |
| Input | RGB image resized and center-cropped to `224x224` |
| Output | ImageNet-1K top-k labels and scores |

Supported variants: `yolo26n-cls`, `yolo26s-cls` (default), `yolo26m-cls`, `yolo26l-cls`, `yolo26x-cls`. Select with `--variant` (weights default to `<variant>.pt` and download on first use). QPC output directories (`qpc_latency/`, `qpc_throughput/`) are variant-blind — export each variant to its own `--work-dir`.

## Generated files

Export writes files under `--work-dir`, including `yolo26s-cls_fp16ready.onnx`,
`qpc_latency/`, `qpc_throughput/`, and optional Triton model repository
entries when `--triton-model-repo` is provided.

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
python -m qaic_models.vision.ultralytics.yolo26.classify.export \
  --variant yolo26s-cls \
  --work-dir build/yolo26s-cls \
  --compile-latency
```

To also generate a Triton model repository, pass `--triton-model-repo`; the
full runbook is in the "Triton serving" section below.

## Triton serving

Generate a Triton model repository from the generated ONNX and latency QPC:

```bash
python -m qaic_models.vision.ultralytics.yolo26.classify.export \
  --variant yolo26s-cls \
  --work-dir build/yolo26s-cls \
  --compile-latency \
  --triton-model-repo build/yolo26s-cls/triton-model-repo
```

This writes raw tensor-serving models and top-k ensembles:

- `yolo26_classify_onnx`: ONNX Runtime backend, input `images`, output `probs` with shape `[1, 1000]`.
- `yolo26_classify_qpc`: QAIC backend, input `images`, output `probs` with shape `[1, 1000]`.
- `yolo26_classify_topk`: Python backend that converts `probs` to `topk_classes`, `topk_scores`, and `topk_labels`.
- `yolo26_classify_onnx_ensemble` and `yolo26_classify_qpc_ensemble`: raw model plus top-k postprocess.

Start Triton with the generated repository mounted. Replace `</path/to/cloud-ai-sdk root>` with this repository checkout path. The in-container install is required because the Python backend imports recipe-local postprocess code. The container command installs the recipe runtime and starts `tritonserver --model-repository=...`.

```bash
docker run -it --rm \
  --shm-size=4g \
  --network host \
  --mount type=bind,source=</path/to/cloud-ai-sdk root>,target=/cloud-ai-sdk \
  --device /dev/accel/ \
  ghcr.io/qualcomm/cloud_ai_triton_server:1.23.2.0 \
  bash -lc 'pip3 install "/cloud-ai-sdk[yolo26-infer,triton]" && \
    /opt/tritonserver/bin/tritonserver \
      --model-repository=/cloud-ai-sdk/build/yolo26s-cls/triton-model-repo'
```

Request the QAIC top-k ensemble model `yolo26_classify_qpc_ensemble`:

```python
import numpy as np
import tritonclient.http as triton_http
from tritonclient.http import InferInput, InferRequestedOutput

from qaic_models.vision.ultralytics.common.preprocess import load_ultralytics_classify_image

image, _ = load_ultralytics_classify_image("build/yolo26s-cls/bus.jpg")
image = np.ascontiguousarray(image.astype(np.float32))
client = triton_http.InferenceServerClient(url="localhost:8000")
infer_input = InferInput("images", image.shape, "FP32")
infer_input.set_data_from_numpy(image)
result = client.infer(
    model_name="yolo26_classify_qpc_ensemble",
    inputs=[infer_input],
    outputs=[
        InferRequestedOutput("topk_classes"),
        InferRequestedOutput("topk_scores"),
        InferRequestedOutput("topk_labels"),
    ],
)
print(result.as_numpy("topk_classes"))  # int32 [1, 5]
print(result.as_numpy("topk_scores"))   # float32 [1, 5]
print(result.as_numpy("topk_labels"))   # string [1, 5]
```

## Demo CLI

```bash
python -m qaic_models.vision.ultralytics.yolo26.classify.demo \
  --image build/yolo26s-cls/bus.jpg \
  --model build/yolo26s-cls \
  --provider QAicExecutionProvider
```

## Python API

```python
from qaic_models.vision.ultralytics.yolo26.classify import Model

model = Model.from_compiled("build/yolo26s-cls", provider="QAicExecutionProvider")
result = model.predict("build/yolo26s-cls/bus.jpg")
print(result["prediction"]["top1"])
```

## Input and output details

Images are resized, center-cropped to `224x224`, scaled to `0.0-1.0`, and sent as NCHW
float32. The model returns softmax probabilities, so the recipe ranks them directly.

Triton raw tensor-serving inputs and outputs use the preprocessed tensor contract directly:

- Input `images`: NCHW float32 tensor with shape `[1, 3, 224, 224]`, RGB resized, center-cropped, and scaled to `0.0-1.0`.
- Raw output `probs`: float32 tensor with shape `[1, 1000]` containing ImageNet probabilities.
- Top-k outputs: `topk_classes` int32 `[1, 5]`, `topk_scores` float32 `[1, 5]`, and `topk_labels` string `[1, 5]`.

## Expected smoke result

The validated sample classified as a vehicle class such as `minibus`.

## Troubleshooting

If `programqpc.bin` is missing, rerun export with `--compile-latency`. If ImageNet labels cannot
be downloaded, predictions still include numeric class IDs.

## License and distribution

Ultralytics YOLO26 source and pretrained weights are AGPL-3.0. Review the upstream license before
distributing weights, exported ONNX files, or QPC binaries.
