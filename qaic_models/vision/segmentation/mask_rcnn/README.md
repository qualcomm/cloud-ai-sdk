# Detectron2 Mask R-CNN

[Detectron2](https://github.com/facebookresearch/detectron2)'s Mask R-CNN (ResNet-50 FPN, COCO 3x) is an instance segmentation model: give it an RGB image and it returns per-object class labels, confidence scores, bounding boxes, and full-image instance masks.

This recipe exports the model to ONNX, compiles the heavy stages for Qualcomm Cloud AI 100, and runs the pipeline with backbone+FPN and ROI heads on the accelerator while the RPN and decode/NMS postprocess run on CPU.

Limitations: batch size 1; fixed input resolution 800x1216 (images are resized); fixed RPN proposal count 200.

## Environment setup

Python 3.12. System packages needed to build Detectron2 during export:

```bash
sudo apt update && sudo apt install -y build-essential cmake protobuf-compiler libprotobuf-dev
```

Create a virtualenv in this recipe directory and install the export/compile extra:

```bash
python3.12 -m venv env_maskrcnn
source env_maskrcnn/bin/activate
pip3 install pip -U
pip3 install "/path/to/cloud-ai-sdk-root[mask-rcnn]"
```

For an inference-only environment use `[mask-rcnn-infer]`; add `[triton]` for the Triton client. Export and demo commands below run inside this venv. Check the accelerator is visible:

```bash
/opt/qti-aic/tools/qaic-util -q    # expect at least one QID with Nsp Free >= 12
```

## Export and compile

Everything generated goes under `--work-dir` (default `build/mask_rcnn`). ONNX export alone (no device needed for this part):

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.export
```

Add `--compile` to also compile the Cloud AI 100 QPC binaries (requires the SDK and a device):

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.export --compile
```

Options: `--quantize` compiles a mixed-precision Int8 backbone instead of pure FP16; `--max-cores N` caps compilation (backbone uses up to 12 cores, ROI heads up to 4); `--weights`, `--detectron2-repo`, `--detectron2-commit`, and `--config-file` select the source model; `--image PATH` uses an existing image instead of downloading the sample; `--detectron2-src PATH` uses an already-installed Detectron2 checkout instead of cloning and installing one.

The exporter uses a pinned Detectron2 checkout (cloned into `build/mask_rcnn/detectron2`) without modifying Detectron2 source. Cloud AI-specific behavior lives in this recipe as config overlays, PyTorch export wrappers, and ONNX cleanup transforms: fixed 800x1216 resolution, fixed Top-K 200, static ROI pooling (no dynamic `NonZero`), and split-ONNX export.

### Generated files

In `build/mask_rcnn` (pass this directory to `--model` everywhere):

| File | Produced by |
|---|---|
| `000000000139.jpg`, `input.raw` | sample input (downloaded) and its preprocessed tensor |
| `model_backbone_fpn.onnx` / `_fp16ready.onnx` | backbone+FPN export, post-transform deployable |
| `model_rpn.onnx` / `_fp16ready.onnx` | RPN (runs on CPU via ONNX Runtime) |
| `model_roi_heads.onnx` / `_fp16ready.onnx` | ROI heads |
| `framework_output_*.raw`, `cpu_output_*.raw` | per-stage goldens from the framework and from ONNX Runtime |
| `qpc_backbone_fpn/programqpc.bin` | `--compile`, 12-core backbone |
| `qpc_roi_heads/programqpc.bin` | `--compile`, 4-core ROI heads |
| `triton-model-repo/` | only with `--triton-model-repo` (see below) |

## Demo CLI

Run on the accelerator (uses `qpc_backbone_fpn/`, `qpc_roi_heads/`, and the RPN ONNX):

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.demo \
  --image build/mask_rcnn/000000000139.jpg \
  --model build/mask_rcnn \
  --out out.jpg
```

CPU reference path (no device needed): add `--provider CPUExecutionProvider`. `--device-id` pins the QID (default: auto-select; pick one with `Nsp Free` >= 12 from `qaic-util -q`). `--top`, `--score-thresh`, and `--nms-thresh` tune decoding. `--json-out <path>` writes the prediction dict (`instances`, `num_instances`) as JSON. Example output:

```
num detections (after postprocess): 20
chair	score=0.9954	box=(555.8,410.8,664.2,592.4)
tv	score=0.9854	box=(1053.1,392.0,1214.8,549.2)

```

`out.jpg` draws the masks, boxes, and labels.

## Python API

```python
from qaic_models.vision.segmentation.mask_rcnn import MaskRCNN

model = MaskRCNN("build/mask_rcnn", provider="QAicExecutionProvider")

result = model.predict(
    "build/mask_rcnn/000000000139.jpg",
    score_thresh=0.05,
    nms_thresh=0.5,
    topk=20,
    output_path="out.jpg",   # optional annotation
)

pred = result["prediction"]
print(pred["num_instances"])
for inst in pred["instances"]:
    print(inst["label"], inst["score"], inst["box_xyxy"])
print("latency_ms:", result["latency_ms"])
```

For per-stage latency statistics use `model.predict_benchmark(image, warmup=1, runs=10)`, which returns per-stage mean/p50/p90/p99 for `backbone_fpn`, `rpn`, `roi_heads`, and `python_postprocess`.

## Input and output details

- Input: any RGB image path; resized to 1216x800, float32 NCHW, ImageNet-style normalization handled by the recipe preprocessing.
- Result structure: `{"prediction": {"instances": [{"label", "score", "box_xyxy"}], "num_instances": int}, "latency_ms": float}`. Labels are COCO-80 class names. `box_xyxy` is in resized-image (1216x800) pixel coordinates.
- Masks: full-image boolean masks accompany the annotation output; the JSON result carries boxes/scores/labels only.
- Empty result: below `score_thresh` no instances are returned; the annotation is the plain resized image.

## Triton serving

Generate the model repository (needs the compiled QPCs, so run after `--compile`):

```bash
python3 -m qaic_models.vision.segmentation.mask_rcnn.export \
  --compile \
  --triton-model-repo build/mask_rcnn/triton-model-repo
```

The repository contains the `mask_rcnn` ensemble — backbone and ROI heads on the `qaic` backend, the RPN on the `onnxruntime` backend (its memory footprint suits CPU execution), and a `python` postprocess model — plus its four component models. It is self-contained (artifacts are copied into each model's `1/` directory).

Start the Triton container with device access and the repo bind-mounted:

```bash
docker run -it --rm \
  --shm-size=4g \
  --network host \
  --mount type=bind,source=</path/to/cloud-ai-sdk root>,target=/cloud-ai-sdk \
  --device /dev/accel/ \
  ghcr.io/qualcomm/cloud_ai_triton_server:1.23.2.0
```

Inside the container, install the serving dependencies (the python postprocess model imports `qaic_models`), then start the server:

```bash
pip install -e "/cloud-ai-sdk[mask-rcnn-infer,triton]"
tritonserver --model-repository=/cloud-ai-sdk/qaic_models/vision/segmentation/mask_rcnn/build/mask_rcnn/triton-model-repo
```

Wait for readiness: `http://localhost:8000/v2/health/ready` and `http://localhost:8000/v2/models/mask_rcnn/ready`.

From the host (or any client with `pip3 install "/path/to/cloud-ai-sdk-root[mask-rcnn-infer,triton]"`):

```python
import numpy as np
import tritonclient.http as triton_http
from tritonclient.http import InferInput

from qaic_models.vision.segmentation.mask_rcnn.preprocess import MaskRCNNImage

client = triton_http.InferenceServerClient(url="localhost:8000")

img = np.ascontiguousarray(MaskRCNNImage.load("build/mask_rcnn/000000000139.jpg").astype(np.float32))
inputs = [InferInput("image", img.shape, "FP32")]
inputs[0].set_data_from_numpy(img)

result = client.infer(
    model_name="mask_rcnn",
    inputs=inputs,
    outputs=[
        triton_http.InferRequestedOutput("cls_results"),  # [N, 2]: class id, score
        triton_http.InferRequestedOutput("cls_counts"),   # [1]
        triton_http.InferRequestedOutput("boxes"),        # [N, 4]: x1, y1, x2, y2
    ],
)
count = int(result.as_numpy("cls_counts")[0])
for i in range(min(count, 5)):
    cls_id = int(result.as_numpy("cls_results")[i, 0])
    score = float(result.as_numpy("cls_results")[i, 1])
    print(f"{i}: cls={cls_id} score={score:.3f} box={result.as_numpy('boxes')[i]}")
```

## Expected smoke result

On `000000000139.jpg` (COCO's yellow-walled kitchen/dining scene) expect around twenty detections dominated by chairs, the TV, and vases, with the person, dining table, potted plants, clock, and refrigerator also detected, at high confidence for the top entries, with colored mask overlays and matching boxes in `out.jpg`. The CPU and QAIC paths should agree on the top detections and classes (FP16 rounding can shift low-score detections near the threshold).

## Troubleshooting

- `ImportError: qaic module not available` — run inside the SDK container or install the `qaic` wheel from `/opt/qti-aic/dev/lib/x86_64/`.
- Compilation or demo run fails on the backbone with unavailable cores — another process holds the device; check `Nsp Free` with `/opt/qti-aic/tools/qaic-util -q` and pick an idle QID (`--device-id`).
- `Missing QPC model binary or ONNX model ... Please export models first` — the `--model` directory lacks `qpc_backbone_fpn/`/`qpc_roi_heads/` (run the export with `--compile`) or points at the wrong work directory.
- Detectron2 build fails during export — install the system packages above (cmake/protobuf headers) inside the venv environment.
- Triton: `mask_rcnn` fails to load with a `qaic_models` import error — the container was started from the plain image without `pip install -e "/cloud-ai-sdk[mask-rcnn-infer,triton]"`; the QPC copies must exist, so generate the repo after `--compile`.
- Triton: `failed to open programqpc.bin` — the container needs `--device /dev/accel/`.
- Stale results after changing thresholds/compile flags — delete `build/mask_rcnn` (or the affected QPC directories) and re-run the export.

## License and distribution

Detectron2 is Apache-2.0 (pinned commit `48b598b4f61fbb24182a69b521b2a0ba3252b842`); the COCO weights come from the Detectron2 model zoo and the training data (COCO) is CC-BY-4.0. Generated ONNX/QPC files inherit those restrictions if you redistribute them; the license also covers the Detectron2 checkout the export clones into `build/mask_rcnn`.
