# Input and Calibration Data

## Contents

- Representative source assets
- RAW plus JSON input format
- Single-input preparation
- PGQ calibration data
- Synthetic-data limitations

## Representative Source Assets

Use one task-representative input for ONNX fidelity, QAIC accuracy, and throughput. A
synthetic input can produce near-uniform activations and hide real-input errors.

Accept an image only when both conditions hold:

1. **Provenance** - the source repository, model documentation, evaluation dataset, or
   bundled fixture identifies it as an input, example, or evaluation asset. A `.jpg` or
   `.png` URL alone is not evidence; documentation may contain diagrams.
2. **Task fit** - the image content matches the model's expected input distribution and
   task.

Use sources in this order and stop at the first valid result:

1. A source-repository test/example asset or official evaluation-dataset asset. Resolve
   dataset identifiers with the documented filename convention or official API. Do not
   invent filenames or URLs, substitute an archive for a single image, or use an
   unvalidated download.
2. The task-appropriate fixture documented by `cloud-ai-vision-demo`.
3. A modality-appropriate, non-degenerate synthetic fallback only when no valid real
   input exists. Never use flat color, all-zero/all-one, silence, empty, or padding-only
   inputs.

Preserve the documented URL scheme; do not rewrite an `http://` asset URL to `https://`.
If a download fails, test the selected URL from the current runtime before changing the
asset. DNS resolution and a successful HTTP response establish connectivity. A TLS
hostname mismatch (`curl` error 60 where the certificate names another host) is an endpoint
certificate problem, not evidence of a container or proxy failure.

For `images.cocodataset.org`, use the documented HTTP image URL or another provenance-valid
HTTPS image. Do not use `-k`, `verify=False`, or another TLS-verification bypass when a valid
source is available. The COCO calibration scripts download the full `val2017.zip` for PGQ;
do not copy their `verify=False` behavior for a single representative image. Validate the
downloaded file as an image and retain the exact source URL in the workspace notes.

## 5. Preparing Inputs: RAW + JSON

**`qaic-compile`** uses the `{"inputs": [...]}` format described here for calibration data and profiling inputs.

> **`qaic-runner` uses a completely different `{"IO-files": [[...]]}` schema** — do NOT pass an `{"inputs": [...]}` file to `qaic-runner`; it will crash with a `nlohmann::json type_error`. See the `cloud-ai-qaic-runner` skill §3.1 for the correct format.

### 5.1 JSON Input Format for `qaic-compile`

JSON describes one or more inputs:

```json
{
  "inputs": [
    {
      "name": "input",
      "path": "./input_000.raw",
      "shape": [1, 3, 224, 224],
      "datatype": "float32"
    }
  ]
}
```

- `name`: must match the model input name exactly.
- `path`: path to the binary `.raw` file.
- `shape`: full shape as a list.
- `datatype`: data type (`float32`, `int64`, etc.).

### 5.2 Generating RAW + JSON for a Single Input

Example for an image model using NCHW, FP32:

```python
import cv2
import numpy as np
import pathlib
import json


def preprocess_image_to_raw(image_path, output_dir, input_name="input",
                            h=224, w=224):
    out = pathlib.Path(output_dir)
    out.mkdir(exist_ok=True)
    entries = []

    img = cv2.imread(str(image_path))            # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h)).astype(np.float32)
    img = img.transpose(2, 0, 1)[np.newaxis]     # HWC -> NCHW, add batch
    img = np.ascontiguousarray(img)

    raw_path = out / "input_000.raw"
    img.tofile(raw_path)

    entries.append({
        "name": input_name,
        "path": str(raw_path),
        "shape": list(img.shape),
        "datatype": "float32",
    })

    json_path = out / "inputs.json"
    json.dump({"inputs": entries}, open(json_path, "w"), indent=2)
    print(f"Wrote {raw_path} and {json_path}")
```

Use `inputs.json` with `qaic-compile`:

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -aic-batch-json-input=inputs.json \
  ...
```

For `qaic-runner` output capture use the `IO-files` format instead — see the `cloud-ai-qaic-runner` skill §3.1.

---

## 6. Calibration Data and Random Inputs (PGQ)

When using **Profile-Guided Quantization (PGQ)** (covered in the Accuracy skill):

- **Recommended**: Use real, representative inputs via `-aic-batch-json-input=calibration_inputs.json`.
  - Build `calibration_inputs.json` using an extended form of the helper above that loops over multiple validation samples.
- **Optional synthetic data**: `-use-random-input-data=gaussian` can generate random Gaussian inputs when you do not yet have calibration data.
  - This is useful for quick experiments but will generally **not** match real image or text distributions.
  - Expect poorer INT8 accuracy with synthetic calibration data; switch to real data for production.

Example PGQ profile generation (high-level):

```bash
/opt/qti-aic/exec/qaic-compile \
  -m=./model.onnx \
  -aic-batch-json-input=calibration_inputs.json \
  -dump-profile=pgq_profile.yaml \
  -quantization-schema-activations=symmetric_with_uint8 \
  -quantization-calibration=KLMinimization
```

Quantization is optional:

- If FP16-only performance meets your latency and throughput requirements (as measured by `qaic-runner` or `qaic.Session`), you can stay in FP16 and skip INT8.

---
