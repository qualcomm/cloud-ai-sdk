# LPRNet — Chinese License Plate Recognition (OCR)

## What this runs

LPRNet is a lightweight CNN (~1.7M parameters) for end-to-end Chinese license
plate recognition. You give it a **cropped plate image** (any size; it is
resized to 94×24) and it returns the **plate string** (e.g. `京PL3N67`) with a
confidence score.

- **Task:** OCR / ALPR — character recognition on a pre-cropped plate. It does
  **not** detect plates in a full scene; run a detector upstream for that.
- **Character set:** 68 Chinese plate characters — 31 province prefixes, 10
  digits, 25 letters, and `-` (which doubles as the CTC blank token).
- **Decoding:** greedy CTC over 18 time steps.
- **Accelerator target:** Qualcomm Cloud AI 100 (FP16).
- **Limitations:**
  - Input must be a cropped plate; aspect-ratio distortion from resizing
    non-94×24 crops can reduce accuracy.
  - Chinese plates only. Must supply own trained weights for other regions.
  - The demo/Python API runs the latency build (batch 1). The Triton
    `lprnet_qpc` model serves the throughput build (batch 8, four instances
    with dynamic batching).

## Generated files

`export.py` writes everything under `--work-dir`:

| File | Description |
|------|-------------|
| `repo/` | Cloned LPRNet_Pytorch source (only if not provided via `--repo`) |
| `input.raw` | Reference input tensor, float32 `1×3×24×94` |
| `framework_output_0.raw` | PyTorch golden output, `1×68×18` |
| `cpu_output.raw` | CPU (ONNX Runtime) golden output from the final ONNX |
| `lprnet.onnx` | Raw ONNX export (opset 16) |
| `lprnet_fp16ready.onnx` | Final deployable ONNX |
| `qpc_latency/` | Latency-tuned QPC; 16 cores, batch 1, `-mos=1`; contains `programqpc.bin` |
| `qpc_throughput/` | Throughput-tuned QPC; 4 cores, batch 8, `-mos=1`; contains `programqpc.bin` |

Pass the **work directory** to `--model` (the recipe resolves the latency
QPC inside it), or point directly at `lprnet_fp16ready.onnx` (CPU) or a
specific QPC directory (Cloud AI 100).

## Environment setup

Run commands from the recipe directory (`qaic_models/vision/ocr/lprnet`), so
`--work-dir build/lprnet` and the Triton repository land inside it.

- Python 3.12, Linux x86_64.

Create a Python environment in a location outside the recipe source tree:

```bash
python3.12 -m venv env_lprnet
source env_lprnet/bin/activate
python -m pip install --upgrade pip
```

- **Export** needs the export extra (CPU-only PyTorch, ONNX tooling,
  OpenCV-headless):

  ```bash
  pip install -e "/path/to/cloud-ai-sdk-root[lprnet]"
  ```

- **Runtime only** (run a demo against already-generated artifacts):

  ```bash
  pip install -e "/path/to/cloud-ai-sdk-root[lprnet-infer]"
  ```

  The QAIC runtime extra additionally requires the Cloud AI 100 SDK
  (`/opt/qti-aic`) and a device for `QAicExecutionProvider` runs.

- Quick checks:

  ```bash
  python3 -c "import torch, onnx, onnxruntime; print(torch.__version__)"
  /opt/qti-aic/tools/qaic-util -q | grep -e Status -e QID   # devices ready
  ```

## Export and compile

```bash
python -m qaic_models.vision.ocr.lprnet.export \
    --work-dir build/lprnet
```

Both QPC builds (latency and throughput) are always compiled, in parallel.

Flags:

- `--work-dir` (required): where all generated files are written. Delete it to
  regenerate from scratch.
- `--repo`: path to an existing LPRNet_Pytorch checkout. Omit to clone
  `https://github.com/sirius-ai/LPRNet_Pytorch` into `--work-dir/repo`
  (shallow clone; weights are expected at `weights/Final_LPRNet_model.pth`).
- `--image`: sample plate image used to build `input.raw` and the CPU golden.
  Defaults to the first sorted image in the source repository's `data/test/`
  directory (`京PL3N67.jpg` — the plate the golden was validated against); the
  recipe ships no images.
- `--triton-model-repo`: optional Triton model repository output directory
  (see "Triton serving").

Both QPC builds — the latency-tuned QPC (16 cores, batch 1) into
`--work-dir/qpc_latency/` and the throughput-tuned QPC (4 cores, batch 8)
into `--work-dir/qpc_throughput/`, both FP16 with `-mos=1` — are always
compiled.

## Triton serving

Generate a Triton model repository from the generated ONNX:

```bash
python -m qaic_models.vision.ocr.lprnet.export \
    --work-dir build/lprnet \
    --triton-model-repo build/lprnet/triton-model-repo
```

The `lprnet_qpc` model always serves the throughput build (batch 8, four
instances, dynamic batching).

This writes four models:

- `lprnet_onnx`: ONNX Runtime backend, input `input`, output `logits` with a dynamic batch shape `[-1, 68, 18]` (the deployable ONNX keeps a symbolic batch axis).
- `lprnet_postprocess`: Python backend that decodes `logits` into `plate` (STRING `[1]`, the full plate), `confidence` (FP32 `[1]`), and `num_chars` (INT32 `[1]`).
- `lprnet_onnx_ensemble`: `lprnet_onnx` → `lprnet_postprocess`.
- `lprnet_qpc`: Python backend that wraps the throughput QPC end-to-end
  (inference + CTC decode). It returns `plate` (STRING `[1]`), `confidence`
  (FP32 `[1]`), and `num_chars` (INT32 `[1]`) per request.

Because a QPC is compiled for a fixed batch, the `lprnet_qpc` model fills the
empty slots up to the QPC's batch by cycling the coalesced requests' inputs,
runs one inference, and drops the filler rows. (Zero-padding is **not** safe:
the compiled QPC has cross-batch interaction, so zero rows shift the real
rows' logits and corrupt their decode — see `triton_model.py`.) The model
declares `max_batch_size: 8` with dynamic batching (5 ms queue delay) and
**four instances** — each instance owns its own `qaic.Session` on the 4-core
QPC, so 4 × 4 = 16 NSP cores fill one SoC and the effective in-flight batch
is 4 × 8 = 32 plates.

Start Triton with the generated repository mounted. Replace `</path/to/cloud-ai-sdk root>` with this repository checkout path. The in-container install is required because the Python backend imports recipe-local postprocess code.

```bash
docker run -it --rm \
  --name qaic-lprnet \
  --shm-size=4g \
  --network host \
  --mount type=bind,source=</path/to/cloud-ai-sdk root>,target=/cloud-ai-sdk \
  --device /dev/accel/ \
  ghcr.io/qualcomm/cloud_ai_triton_server:1.23.2.0 \
  bash -lc 'pip3 install "/cloud-ai-sdk[lprnet-infer,triton]" && \
    /opt/tritonserver/bin/tritonserver --model-repository=/cloud-ai-sdk/qaic_models/vision/ocr/lprnet/build/lprnet/triton-model-repo'
```

Request the QAIC model `lprnet_qpc` (one plate per request; Triton batches
concurrent requests automatically):

```python
import numpy as np
import tritonclient.http as triton_http
from tritonclient.http import InferInput, InferRequestedOutput

from qaic_models.vision.ocr.lprnet.preprocess import preprocess_plate_image

plate_input = preprocess_plate_image("plate.jpg")  # [1, 3, 24, 94]
client = triton_http.InferenceServerClient(url="localhost:8000")
# `lprnet_qpc` has a non-zero max_batch_size, so the input keeps the batch dim.
infer_input = InferInput("input", plate_input.shape, "FP32")  # [1, 3, 24, 94]
infer_input.set_data_from_numpy(plate_input)
result = client.infer(
    model_name="lprnet_qpc",
    inputs=[infer_input],
    outputs=[
        InferRequestedOutput("plate"),
        InferRequestedOutput("confidence"),
        InferRequestedOutput("num_chars"),
    ],
)

# `plate` is a STRING [1] tensor holding the full plate; the element is UTF-8
# bytes (e.g. b'\xe4\xba\xacPL3N67' is 京PL3N67). Decode it to a str.
plate = result.as_numpy("plate")[0].decode("utf-8")
print("plate:", plate)
print("confidence:", result.as_numpy("confidence")[0])
print("num_chars:", int(result.as_numpy("num_chars")[0]))
```

The `plate` output is a `STRING [1]` tensor holding the full decoded plate as a
single string; its element is UTF-8 bytes, so decode it (CJK characters are
multi-byte). `num_chars` gives the decoded length. Request `lprnet_onnx` (or
`lprnet_onnx_ensemble`) for the ONNX path instead.

## Demo CLI

```bash
# Cloud AI 100
python -m qaic_models.vision.ocr.lprnet.demo \
    --image plate.jpg \
    --model build/lprnet \
    --provider QAicExecutionProvider \
    --out annotated.jpg
```

`--out` draws the decoded plate onto an annotated image; `--json-out prediction.json` writes the prediction (plate text, confidence, characters) as JSON.

Example output:

```text
latency_ms: <ms>
plate: 京PL3N67
confidence: 0.9856
wrote: annotated.jpg
```

## Python API

```python
from qaic_models.vision.ocr.lprnet import Model

# Cloud AI 100
m = Model("build/lprnet", provider="QAicExecutionProvider")
result = m.predict("plate.jpg", "annotated.jpg")

print(result)
```

## Input and output details

**Input:** a cropped plate image (JPEG/PNG). Preparation: resize to 94×24,
normalize `(pixel − 127.5) × 0.0078125`, BGR, NCHW float32.

**Result structure** (`result["prediction"]`):

| Field | Type | Meaning |
|-------|------|---------|
| `plate` | str | Decoded plate string (may be empty) |
| `confidence` | float | Mean softmax-max probability over kept CTC steps, 0–1 |
| `characters` | list[str] | Individual decoded characters |

`result["latency_ms"]` is the single-inference latency in milliseconds.

An empty decode (all-blank output) yields `plate: ""`, `confidence: 0.0`,
`characters: []`.

## Expected smoke result

Using the default golden sample (the source repo's `data/test/京PL3N67.jpg`,
the `京PL3N67` plate — see the `--image` flag above):

- **Cloud AI 100:** `plate: 京PL3N67`, confidence ≈ 0.9856.

## Troubleshooting

- **`qaic module not available`** — you're running `QAicExecutionProvider`
  outside a Cloud AI container. Use `--provider CPUExecutionProvider` or run
  inside a container with the SDK.
- **`No generated ONNX artifact found` / `No generated QPC found`** — pass the
  work directory produced by `export.py` (a full run compiles both QPC
  builds). A stale `--work-dir` from a different model will also fail;
  delete it.
- **`--repo ... does not contain model/LPRNet.py`** — point `--repo` at a full
  LPRNet_Pytorch checkout (the default clone does this automatically).
- **Weights not found** — the repo checkout must contain
  `weights/Final_LPRNet_model.pth`.
- **Compile fails** — check `qaic-util -q` for device status; reset devices
  with `sudo /opt/qti-aic/tools/qaic-util -s` and retry.
- **`Triton QPC model requires the generated throughput QPC`** — the
  `qpc_throughput/` artifact is missing from the work dir; rerun `export.py`
  (a full run compiles both QPC builds).
- **CJK glyphs render as boxes in `--out`** — the demo needs a CJK font
  (Noto Sans CJK); the predictor downloads one below `QAIC_WORK_ROOT` (or
  `<cloud-ai>/qaic-work` in a checkout) when missing.
- **`libGL.so.1` import errors** — use the headless OpenCV extra
  (`opencv-python-headless`, included in `[lprnet]`).

## License / distribution

- Model code and weights: **Apache 2.0** (sirius-ai/LPRNet_Pytorch).
- Trained on a Chinese license plate dataset (CCPD-style); verify the
  dataset's terms before redistributing the weights or generated artifacts.
- Generated QPC/ONNX artifacts are build outputs of the Apache-2.0 weights and
  may be distributed subject to that license.
