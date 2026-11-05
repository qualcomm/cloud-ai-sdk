# Detection and Segmentation Postprocessing

## Contents

- [Detector-family inspection](#detector-family-inspection)
- [Instance-segmentation output inspection](#instance-segmentation-output-inspection)
- [Detection and segmentation assertions](#detection-and-segmentation-assertions)
- [Postprocessing failure triage](#postprocessing-failure-triage)

## Detector-family inspection

Run this before writing any postprocess code:

```python
import qaic, numpy as np
session = qaic.Session(model_path="<qpc_dir>/programqpc.bin", num_activations=1)
session.setup()
print("outputs:", session.model_output_names)
print("shapes: ", session.model_output_shape_dict)
raw = np.fromfile("/workspace/<model>/cpu_output.raw", dtype=np.float32).reshape(<shape>)
boxes_raw  = raw[0, :4, :]   # adjust if layout is (N, 4+C)
scores_raw = raw[0, 4:, :]
print(f"box range: {boxes_raw.min():.2f}..{boxes_raw.max():.2f}")
print(f"score range: {scores_raw.min():.4f}..{scores_raw.max():.4f}")
```

**Family (output shape):**
- `(1, 4+C, N)` or `(1, N, 4+C)` where N is large → **Path A** (YOLO/SSD)
- Two tensors `(1, Q, C+1)` + `(1, Q, 4)` → **Path B** (DETR)
- Outputs named `"boxes"`, `"scores"`, `"nmsed_*"` → NMS embedded, use outputs directly

**Score activation (Path A):**
- `scores_raw.max() ≤ 1.0` → already activated. **Use as-is. Do NOT apply sigmoid, softmax, or any other function — even "for safety". Doing so maps near-zero noise scores to ~0.5, causing hundreds of thousands of false positives.**
- `scores_raw.max() > 1.0` → raw logits; apply per-class **sigmoid** (never softmax for dense detectors)
  - Softmax turns `dog=0.85` into `dog≈0.03` — destroys confidence signal

**⚠ Mandatory score-range assertion — run this BEFORE writing any postprocess code:**
```python
import numpy as np
raw = np.fromfile("/workspace/<model>/cpu_output.raw", dtype=np.float32)
# reshape to (1, 4+C, N) or as appropriate for this model
out = raw.reshape(1, -1, <N>)    # adjust shape
scores_raw = out[0, 4:, :]       # rows 4+ are class scores
print(f"score range: {scores_raw.min():.4f}..{scores_raw.max():.4f}")
assert scores_raw.max() <= 1.01 or scores_raw.max() > 1.01  # documents which branch
# ↑ If max ≤ 1.0: scores are already activated — use as-is, no sigmoid
# ↑ If max > 1.0: scores are raw logits — apply per-class sigmoid
```
The score range determines the entire postprocess path. Running this assertion once saves
debugging hours later. Record the exact values in a comment in the postprocess method.

**Coordinate space:**
- `boxes_raw.max() ≤ 1.1` → normalized [0,1], multiply by (INPUT_W, INPUT_H) first
- `boxes_raw.max() > 1.1` → pixel space, use directly

**Coordinate format (xywh vs xyxy) — run this before writing postprocess:**
```python
# YOLO-family (YOLOv5/v8/v11, YOLO-NAS) always outputs cx,cy,w,h — never x1,y1,x2,y2.
# SSD outputs x1,y1,x2,y2 in normalized [0,1].
# Confirm programmatically: in xywh, ch2 (w) and ch3 (h) are object widths/heights,
# so their max is bounded by the image size. Ratio ch0_range/ch2_mean distinguishes formats.
mask = scores_raw.max(axis=0) > 0.1   # high-confidence anchors only
if mask.sum() > 10:
    b = boxes_raw[:, mask]
    fmt = "xywh" if b[0].std() > b[2].std() * 1.5 else "xyxy"
else:
    fmt = "xywh (assumed — YOLO-family)"
print(f"coordinate format: {fmt}")
print(f"  ch0(cx/x1) mean={boxes_raw[0].mean():.1f}  ch2(w/x2) mean={boxes_raw[2].mean():.1f}")
# Expected for xywh: ch0 mean ≈ INPUT_W/2 (anchors spread across image), ch2 mean << INPUT_W/2
```
**If confirmed xywh**: `x1 = cx - w/2; x2 = cx + w/2` (mandatory conversion).
**If confirmed xyxy**: use `x1, y1, x2, y2 = boxes_raw[0], boxes_raw[1], boxes_raw[2], boxes_raw[3]` directly.

**NMS (Path A):**
Check ONNX for embedded NMS: `python3 -c "import onnx; m=onnx.load('<model>.onnx'); print([n.op_type for n in m.graph.node if 'Max' in n.op_type])"`
- Embedded → skip manual NMS
- Not embedded → `_nms()` is already in the template; call it

**Path B (DETR):** No NMS. Apply softmax over `C+1` classes, exclude no-object (last col, index C).
Output boxes are normalized `cxcywh` in [0,1]; convert to pixel `x1y1x2y2` using `orig_w/h`.

**DETR multi-input:** DETR requires `pixel_values` + `pixel_mask`. The template's Path B
`preprocess` already returns both keys — fill it in. Nothing else needs overriding: the
base class `_forward()` passes the whole `preprocess` dict straight to the session.
Store `self._pixel_mask = np.ones((1, INPUT_H, INPUT_W), dtype=np.float32)` in `preprocess`
(`pixel_mask` is promoted to float32 by `-convert-to-fp16`; verify with
`session.model_input_shape_dict`).

---

## Instance-segmentation output inspection

Run this before writing any postprocess code to determine tensor layout and class dimension:
```bash
python3 -c "
import qaic, numpy as np
session = qaic.Session(model_path='<qpc_dir>/programqpc.bin', num_activations=1)
session.setup()
for name in session.model_output_names:
    shape, dtype = session.model_output_shape_dict[name]
    print(f'{name}: {shape} {dtype}')
"
```
Identify which tensor is class logits `(1, Q, C+1)` and which is mask logits `(1, Q, H, W)` by their shapes.
The class dimension `C` determines the label list: C=80 (or 81 with no-object) → `COCO_80_LABELS`; C=91 → `COCO_91_LABELS`.
Record Q (number of queries) — if the smoke test later shows `num_instances == Q`, the no-object class was handled incorrectly (see Instance segmentation postprocess pitfalls).

---

## Detection and segmentation assertions

### Detection assertion script (must pass before proceeding to model card):
```bash
python3 /workspace/<model>/predict.py --image test_input.jpg --output /tmp/smoke_assert.jpg 2>/dev/null \
| python3 -c "
import sys, json
lines = [l for l in sys.stdin.read().splitlines() if l.strip().startswith('{')]
assert lines, 'no JSON line in stdout — check predict.py prints a JSON line (qaic library prints to stdout; pipe filters it)'
d = json.loads(lines[-1]); p = d['prediction']
assert isinstance(p, dict)
for k in ('boxes','labels','scores'): assert k in p, f'missing: {k}'
assert len(p['boxes']) > 0, 'empty boxes on real image — postprocess bug'
assert len(p['boxes']) == len(p['labels']) == len(p['scores'])
assert all(0<=s<=1 for s in p['scores'])
top = max(p['scores'])
assert top >= 0.25, f'top score {top:.3f} < 0.25 — sigmoid applied to activated scores, or CONF_THRESH too low'
assert len(p['boxes']) < 50, f'{len(p[\"boxes\"])} detections — sigmoid applied to activated scores (all ~0.5) or CONF_THRESH too low'
for b in p['boxes']:
    x1,y1,x2,y2 = b
    assert x2 > x1 and y2 > y1, f'invalid box geometry {b} — xywh treated as xyxy (missing cx-w/2 conversion)'
assert not any(l == 'N/A' for l in p['labels']), f'N/A labels — wrong COCO label list or off-by-one (label_idx = class_id + 1); use label_idx = int(class_id)'
print(f'OK: {len(p[\"boxes\"])} detections, top={top:.3f}, labels={p[\"labels\"][:3]}')
"
```

### COCO ground-truth check - run when `LABELS_URL` is `"COCO_80"`/`"COCO_91"`; skip for custom datasets (no bundled ground-truth image).

Structural checks above pass even when an off-by-one shifts every label to a valid neighbour (cat→dog→horse). Only a known-answer test catches it. `smoke_test.jpg` is COCO #39769 — **two cats on a couch** — so `cat` must appear:
```bash
SMOKE=$(find -L /skills "$HOME/.agents" "$HOME/.claude" .agents .claude -path "*/cloud-ai-vision-demo/smoke_test.jpg" 2>/dev/null | head -1)
python3 /workspace/<model>/predict.py --image "$SMOKE" --output /tmp/coco_gt.jpg 2>/dev/null \
| python3 -c "
import sys, json
d = json.loads([l for l in sys.stdin if l.strip().startswith('{')][-1])
labels = [l.lower() for l in d['prediction']['labels']]
assert 'cat' in labels, f\"expected 'cat' (two cats on a couch); got {labels} — off-by-one (drop +1 on class_id) or wrong label list\"
print(f'COCO ground-truth OK: {labels[:5]}')
"
```

---

| Symptom | Cause | Fix |
|---|---|---|
| score range near 0 after activation | softmax applied to sigmoid scores | remove softmax; use scores directly |
| score range [0,1] but boxes empty | threshold too high | lower `CONF_THRESH` to 0.1 |
| boxes empty after NMS | bad coords or IoU too low | print box coords — if huge/negative, letterbox undo is wrong |
| numeric labels | label fetch failed | check `LABELS_URL` returns correct line count |
| boxes look present on smoke test (gray image) but wrong on real images | xywh treated as xyxy — cx/cy used as x1/y1, w/h used as x2/y2 | add `x1=cx-w/2`, `x2=cx+w/2` conversion before letterbox undo |

## Postprocessing failure triage

### Too-many-boxes triage (sigmoid-on-activated-scores)

| Symptom | Cause | Fix |
|---|---|---|
| `top=0.500` in assertion output | sigmoid applied to already-activated scores (maps 0→0.5) | remove sigmoid; `cpu_output.raw` score max ≤ 1.0 means use as-is |
| >50 detections on one real image | sigmoid inflation + CONF_THRESH too low | fix sigmoid first, restore `CONF_THRESH = 0.25` |

### Instance segmentation assertion script (must pass before proceeding to model card):
```bash
python3 /workspace/<model>/predict.py --image test_input.jpg --output /tmp/smoke_assert.jpg 2>/dev/null \
| python3 -c "
import sys, json
lines = [l for l in sys.stdin.read().splitlines() if l.strip().startswith('{')]
assert lines, 'no JSON line in stdout'
d = json.loads(lines[-1]); p = d['prediction']
assert 'num_instances' in p and 'instances' in p, f'missing keys: {list(p)}'
n = p['num_instances']
assert n > 0, 'zero instances on real image — postprocess bug or CONF_THRESH too high'
assert n < 50, f'{n} instances — likely postprocess bug (investigate)'
top = max(i[\"score\"] for i in p[\"instances\"])
assert 'N/A' not in [i[\"label\"] for i in p[\"instances\"][:5]], 'N/A labels — wrong COCO label list'
print(f'OK: {n} instances, top={top:.3f}, labels={[i[\"label\"] for i in p[\"instances\"][:3]]}')
"
```

### Instance-segmentation postprocess pitfalls

*Too many instances (num_instances near or equal to the model's total query count):*
Query-based models like Mask2Former and MaskDINO output a fixed number of object queries (e.g. 200). Each query produces a class distribution over foreground classes plus a special "no-object" class. If the postprocess softmax is applied over only the foreground classes (e.g. 80 logits instead of 81), then every query is forced to choose a foreground label regardless of confidence — all 200 queries fire above any threshold. The fix is to include the no-object class in the softmax and discard queries whose highest-scoring class is no-object. Anchor-based models (Mask RCNN) do not have this issue because NMS eliminates redundant proposals before thresholding.

*Path B mask postprocess: resize to box region and paste into canvas - not directly to image size:*
Anchor-based models (Mask R-CNN, SOLO, CondInst) output mask logits whose spatial dimensions cover the bounding box only, not the full image. Do not write `mask_img.resize((orig_w, orig_h))` - that maps a box-crop mask onto the whole image. Resize to `(box_w, box_h)` in original-image coordinates, then paste into a zeros canvas at `(x1, y1)`:
```python
mask_box  = np.array(Image.fromarray(binary_mask).resize((box_w, box_h), Image.BILINEAR)) > 128
full_mask = np.zeros((self._orig_h, self._orig_w), dtype=bool)
full_mask[y1:y2, x1:x2] = mask_box
```
Path A query-based models output masks at full encoder resolution, so direct resize to `(orig_w, orig_h)` is correct there — the distinction is anchor-based vs query-based, not mask tensor shape.

*Path A smoke test shows oversized blobs regardless of confidence threshold:*
This is a model accuracy symptom, not a postprocess bug. Verify ONNX fidelity (cosine >= 0.9999 vs framework golden) — degraded mask logits produce oversized blobs at any threshold. Fix the export before tuning postprocess.

*N/A labels in output:*
Most modern COCO-trained models (YOLO, DETR variants, Mask2Former) output dense 0-indexed class logits where index 0 = person, 1 = bicycle, ..., 79 = toothbrush — use `COCO_80_LABELS`. The original COCO dataset uses sparse category IDs (1–90) with 11 unused slots filled with "N/A"; `COCO_91_LABELS` maps those IDs directly. Using `COCO_91_LABELS` on a model with dense 0-79 output shifts every label by one and produces N/A entries for unused slots. The class output dimension is the clearest signal: dim == 80 or 81 → `COCO_80_LABELS`; dim == 91 or 92 → `COCO_91_LABELS`.

*Indexing into COCO_91_LABELS — do not add 1 to class_id:*
`COCO_91_LABELS[0]` is `'N/A'` (the background/no-object slot), so `argmax` output maps **directly** to list index: `label_idx = int(class_id)`. Do not write `label_idx = class_id + 1`. The list header comment "1-based COCO category IDs" refers to the original dataset numbering, not to how Python indexing works here — the `'N/A'` at index 0 already absorbs the offset. Adding `+1` silently shifts every class forward by one (cat→dog, dog→horse, etc.) with no error raised.
