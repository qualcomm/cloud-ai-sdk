# `qaic_models`

`qaic_models` is the installable Qualcomm Cloud AI recipe package for exporting, compiling, running, and serving focused vision and speech model workflows, including Triton integrations where supported.

Every cataloged recipe provides a QAIC execution path.

| Category | Recipe | Task | Guide |
|---|---|---|---|
| Vision / classification | YOLO11 Classify | Image classification | [README](vision/ultralytics/yolo11/classify/README.md) |
| Vision / classification | YOLO26 Classify | Image classification | [README](vision/ultralytics/yolo26/classify/README.md) |
| Vision / detection | YOLO11 Detect | Object detection | [README](vision/ultralytics/yolo11/detect/README.md) |
| Vision / detection | YOLO26 Detect | Object detection | [README](vision/ultralytics/yolo26/detect/README.md) |
| Vision / OCR | LPRNet | License-plate OCR | [README](vision/ocr/lprnet/README.md) |
| Vision / OCR | PaddleOCR | Document OCR pipeline | [README](vision/ocr/paddle_ocr/README.md) |
| Vision / segmentation | Mask R-CNN | Instance segmentation | [README](vision/segmentation/mask_rcnn/README.md) |
| Speech / text-to-speech | F5-TTS | Speech generation | [README](speech/f5_tts/README.md) |

Looking for LLMs, VLMs, embedding, text-to-image, text-to-video, speech recognition models? Visit the [Efficient Transformers Library](https://github.com/qualcomm/efficient-transformers).
## Installation and quick start

Run from the repository root in a Python 3.12 environment. This LPRNet path exports deployable ONNX and both recipe QPC variants, then runs the QAIC demo on a Cloud AI device:

```bash
python3.12 -m venv env_lprnet
source env_lprnet/bin/activate
python -m pip install --upgrade pip
pip install -e ".[lprnet]"
/opt/qti-aic/tools/qaic-util -q
python -m qaic_models.vision.ocr.lprnet.export --work-dir build/lprnet
python -m qaic_models.vision.ocr.lprnet.demo --image build/lprnet/repo/data/test/京PL3N67.jpg --model build/lprnet --provider QAicExecutionProvider --out build/lprnet/annotated.jpg
```

For runtime-only use with existing LPRNet artifacts, install `[lprnet-infer]`. See the [LPRNet recipe guide](vision/ocr/lprnet/README.md) for artifact layout, Triton serving, and troubleshooting.

## QAIC-first workflow

Recipes follow this deployment path:

`framework/checkpoint -> recipe export -> deployable ONNX -> QPC -> QAIC demo or Triton serving`

Recipe-local exporters produce recipe-specific deployable ONNX and QPC artifacts according to their flags. Preprocessing and postprocessing remain concerns of the relevant recipe, runtime, or demo components.

Recipe-local demos consume those artifacts and can write annotated images, JSON, or audio results.

Where supported:

- Exporters can produce latency- and throughput-oriented QPC variants and Triton model repositories.
- Multi-stage recipes can separate stages across the accelerator and host; PaddleOCR separates layout, detection, and recognition.
- Each linked recipe guide documents exact stages, device allocation, batching, variants, and Triton contracts.

Install dependencies through the recipe extras in the root `pyproject.toml`: an export/compile extra such as `[lprnet]`, or a corresponding runtime-only extra such as `[lprnet-infer]` when provided. Add `[triton]` for Triton client support; serving-specific extras and pinned upstream dependencies are recipe-specific.

## Python API

Python APIs are recipe-specific. There is no package-wide registry, universal
argument list, or universal result schema.

| Shape | Import | Primary method |
|---|---|---|
| Single-model vision | `from qaic_models.vision.ocr.lprnet import Model` | `predict(...)` |
| PaddleOCR composition | `from qaic_models.vision.ocr.paddle_ocr import Pipeline` | `predict(...)` |
| PaddleOCR stages | `from qaic_models.vision.ocr.paddle_ocr.layout import Model` | stage-specific `predict(...)` |
| F5-TTS | `from qaic_models.speech.f5_tts import Model` | `infer(...)` |

LPRNet example:

```python
from qaic_models.vision.ocr.lprnet import Model

model = Model("build/lprnet", provider="QAicExecutionProvider")
result = model.predict("plate.jpg", output_path="annotated.jpg")
print(result["prediction"])
print(result["latency_ms"])
```

PaddleOCR composition example:

```python
from qaic_models.vision.ocr.paddle_ocr import Pipeline

pipeline = Pipeline("build/paddle_ocr", provider="QAicExecutionProvider")
result = pipeline.predict("document.png")
print(result["prediction"]["texts"])
```

PaddleOCR leaf stages use their own module-local `Model` aliases and input
contracts. F5-TTS `Model` returns its structured audio result from `infer(...)`.

## Artifacts and prerequisites

Exported ONNX files, QPC directories containing `programqpc.bin`, sample outputs, and optional Triton repositories are written to a caller-owned work directory. Keep generated artifacts outside the package source tree; they are not required for package installation.

Checkpoints and source inputs may be supplied externally or resolved through recipe-specific caches. Artifact names and accepted model paths vary by recipe; use the linked guide as the source of truth.

QPC compilation requires a compatible Cloud AI SDK/compiler. QAIC execution requires the compatible QAIC Python/runtime package and a Ready device. Some export steps can run without hardware.

Check device readiness with `/opt/qti-aic/tools/qaic-util -q`; recipe guides document additional SDK container, QID, and Triton requirements.

## Testing and validation

Run the package test suite from the repository root:

```bash
python -m pytest tests/qaic_models/ -q
```

The suite covers hardware-free structure, preprocessing/postprocessing, artifact, Triton configuration, and functional-harness contract tests. Hardware smoke tests skip when QAIC, a Ready device, or generated artifacts are unavailable.

Use the detailed recipe README for recipe-specific behavior, commands, inputs, outputs, and runtime requirements.

See the [qaic_models test guide](../tests/qaic_models/README.md), the repository [Cloud AI Developer Resources guide](../README.md), and the upstream [Cloud AI User Guide](https://qualcomm.github.io/cloud-ai-sdk-pages/latest/) for SDK installation and platform operation.

## Third-party licensing

This package contains recipe code and documentation and does not grant licenses to third-party models, checkpoints, datasets, or upstream source projects.

Before downloading, exporting, serving, or redistributing artifacts, review each recipe's licensing section, model card, checkpoint license, dataset terms, and upstream project license.
