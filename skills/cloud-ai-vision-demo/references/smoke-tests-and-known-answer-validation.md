# Smoke Tests and Known-Answer Validation

## Contents

- [Image selection and execution](#image-selection-and-execution)
- [Preprocessing parity](#preprocessing-parity)
- [Encoder-decoder known-answer validation](#encoder-decoder-known-answer-validation)

## Image selection and execution

```bash
python3 -m py_compile /workspace/<model>/predict.py && echo "syntax OK"
```

### Image selection

Look for real images that shipped with the model source. The source — whether a cloned repository, a pip-installed package, or a downloaded archive — often includes canonical test assets (jpg or png sample images in an `assets/` or `test/` subdirectory). These images match the model's training distribution better than a generic fallback. Use whatever search strategy fits the workspace layout to locate them; copy the most representative one to `test_input.jpg`.

If no suitable image is found in the workspace, fall back by pipeline shape:

| Pipeline shape | Fallback |
|---|---|
| **Single-stage detection / segmentation / classification** | Pre-bundled test image (`smoke_test.jpg`) — COCO + ImageNet classes |
| **Single-stage OCR / text recognition** | `input.raw` is acceptable (convert to JPEG if needed); pre-isolated text-region images also work |
| **Encoder-decoder image-to-text** (TrOCR, Donut, Pix2Struct) | Use the bundled `ocr_smoke_test.jpg` fixture (see copy command below). **Never use a synthetic PIL image** — a default PIL font at typical resolutions is outside the training distribution of all printed-OCR models; the CPU reference also produces garbage, making the test silently vacuous. |
| **Two-stage detect-then-recognize** | **No acceptable fallback** — a real scene image is required; generic fixtures (`smoke_test.jpg`, `input.raw`) don't match the primary stage's training distribution and produce zero detections (silent failure, no error) |

For the pre-bundled test image (detection / segmentation / classification fallback):
  ```bash
  cp "$(find -L /skills "$HOME/.agents" "$HOME/.claude" .agents .claude -path "*/cloud-ai-vision-demo/smoke_test.jpg" 2>/dev/null | head -1)" \
     /workspace/<model>/test_input.jpg
  ```
  This is COCO val2017 image 39769 (cats on sofa, CC-BY 2.0, 640×480, ~173 KB).
  It is always present in the container at that path without any network access.
  **Never use `input.raw`** for detection / segmentation / classification (often random noise → empty detections).
  **Never create a synthetic image with PIL** (same failure mode as `input.raw`).

For single-stage OCR: `input.raw` is acceptable; convert if needed:
  ```bash
  python3 -c "
  import numpy as np; from PIL import Image
  a = np.fromfile('input.raw', dtype=np.float32).reshape(3,<H>,<W>)
  Image.fromarray((a.transpose(1,2,0)*255).clip(0,255).astype('uint8')).save('input.jpg')"
  ```
  Then run with `--image input.jpg`.

For encoder-decoder image-to-text, copy the bundled fixture:
  ```bash
  cp "$(find -L /skills ~/.claude .claude -path "*/cloud-ai-vision-demo/ocr_smoke_test.jpg" 2>/dev/null | head -1)" \
     /workspace/<model>/test_input.jpg
  ```
  This is a real printed-text line image (black text on white, ~640×80 px) representative of the training distribution for all printed-OCR encoder-decoder models. It is always present in the container at that path without network access.

### Run and verify
```bash
python3 /workspace/<model>/predict.py --image test_input.jpg --output test_output.jpg 2>&1
```
Required: valid JSON on stdout, positive `latency_ms`, no traceback, human-readable labels.

## Preprocessing parity

`predict.py` is the only code that runs at demo time, yet the upstream accuracy
validation compared the QPC against `input.raw` — a tensor `predict.py.preprocess()`
never produced.
If `preprocess()` and the export script disagree on normalization, the demo is broken
(uniform/degenerate output, the same for every image) even though every numeric accuracy
check passed. This happens in both directions: `preprocess()` applies mean/std the ONNX
graph already bakes in (double-normalization), or omits mean/std the graph expects.

`input.raw` is the exact tensor the QPC was validated against, and is authoritative.
Print its value distribution:

```bash
cd /workspace/<model> && python3 -c "
import numpy as np
a = np.fromfile('input.raw', dtype=np.float32)
print(f'input.raw: min={a.min():.3f} max={a.max():.3f}')"
```

Classify `input.raw` into one scheme using the decision key (the **min/max signs**, not
the exact numbers — an ImageNet tensor's true bounds are channel-dependent, roughly
`[-2.12, 2.64]`, and never hit a tidy round number):

| Decision key | Scheme | What `preprocess()` must do |
|---|---|---|
| `min < 0` | ImageNet mean/std | divide by 255, then `(x - mean) / std` |
| `min ≈ 0`, `max ≈ 1` | scaled | divide by 255 only — **no mean/std** |
| `min ≈ 0`, `max ≈ 255` | raw pixels | resize only, no division |

Then read `preprocess()` in `predict.py` and confirm it produces that same scheme.
The schemes must match `input.raw` — not the model name, README, or a default habit
like "ViT backbone ⇒ ImageNet mean/std." If they differ, fix `preprocess()` to match the
export script's input construction. Find that script in the workspace — it is the `.py`
file that calls `torch.onnx.export()` (it built the example tensor and wrote `input.raw`);
locate it with `grep -l "torch.onnx.export\|\.tofile" /workspace/<model>/*.py`. Replicate
its exact normalization, including the literal mean/std constants. If no export script is
present, derive the required normalization from `input.raw`'s scheme (the table above)
and use the standard ImageNet constants `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`
for the mean/std case. Do not proceed to the model card until the schemes agree.

---

## Encoder-decoder known-answer validation

First establish the CPU ground truth on the same fixture:
```bash
OCR_SMOKE=$(find -L /skills "$HOME/.agents" "$HOME/.claude" .agents .claude -path "*/cloud-ai-vision-demo/ocr_smoke_test.jpg" 2>/dev/null | head -1)
# Run CPU reference (transformers model.generate) on the fixture image:
CPU_PRED=$(python3 -c "
from transformers import AutoProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
cache = os.path.join(os.path.dirname(os.path.abspath('/workspace/<model>/predict.py')), 'hf_cache')
proc  = AutoProcessor.from_pretrained('<hf_model_id>', cache_dir=cache, local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained('<hf_model_id>', cache_dir=cache, local_files_only=True)
img   = Image.open('$OCR_SMOKE').convert('RGB')
ids   = model.generate(proc(images=img, return_tensors='pt').pixel_values)
print(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
" 2>/dev/null)
echo "CPU ground truth: $CPU_PRED"
```

Then assert QAic matches:
```bash
QAIC_PRED=$(python3 /workspace/<model>/predict.py --image "$OCR_SMOKE" 2>/dev/null \
  | python3 -c "import sys,json; lines=[l for l in sys.stdin if l.strip().startswith('{')]; print(json.loads(lines[-1])['prediction'].strip())")
python3 -c "
cpu  = '''$CPU_PRED'''
qaic = '''$QAIC_PRED'''
assert cpu == qaic, f'Known-answer mismatch:\\n  CPU : {cpu!r}\\n  QAic: {qaic!r}\\nSingle-step cosine can pass while autoregressive output is wrong — check KV-cache wiring (cloud-ai-model-conversion/references/exporting-models.md, "Autoregressive decoder export").'
print(f'Encoder-decoder known-answer OK: {qaic!r}')
"
```

If `CPU_PRED` is garbage on the fixture, the fixture does not match this model's training distribution — find a real document crop from the model's source repo or test assets and use that instead.

A whole-transcript mismatch is a real defect. For a **per-step** token gate, check the score margin at the first mismatch before investigating wiring — see `cloud-ai-accuracy-tuning` [Near-tied discrete outputs](../../cloud-ai-accuracy-tuning/references/output-comparison.md#near-tied-discrete-outputs).
