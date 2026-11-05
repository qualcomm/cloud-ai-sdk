# Template Routing and Integration

## Contents

- [Template routing and universal integration rules](#template-routing-and-universal-integration-rules)
- [Fill in the template stubs](#fill-in-the-template-stubs)
- [Multi-stage pipeline integration](#multi-stage-pipeline-integration)

## Template routing and universal integration rules

**Do NOT write predict.py from scratch.** `visualize()` and `_nms()` are already
correct in the template — overwriting them is the most common source of rendering bugs.

**Before routing: verify the pipeline is complete.**
Count QPCs in the workspace:
```bash
find /workspace/<model>/ -name programqpc.bin | wc -l
```
If the model is one stage of a multi-stage pipeline and other stages have no QPC yet,
go back and compile the missing stages before writing predict.py. A predict.py that
covers only part of the pipeline cannot produce meaningful end-to-end results.

**Complex source pipelines - preserve upstream orchestration.** When the source
repository provides an authoritative inference entry point with nontrivial control flow,
`predict.py` MUST invoke that entry point. The source pipeline must continue to own
request splitting, preprocessing, tokenization or feature extraction, schedulers and
iterative loops, state/cache updates, dynamic boundary calculations, result ordering,
and final assembly. This applies to TTS, ASR, seq2seq, diffusion, and other
pipeline-driven models.

Integrate QPC execution at the narrowest stable interface that preserves the source call
graph. Depending on the source architecture, use a compatible model adapter, module
forward replacement, hook, or pipeline-component substitution. Replace only heavy
compiled computations and preserve each replaced component's tensor contract.
**Importing source helpers while reproducing its orchestration in a new host loop does
not preserve the source pipeline.** Monkey-patching is one integration mechanism, not a
requirement when another narrow interface better preserves the call graph.

A compiled shape limits one QPC invocation, not necessarily the complete user request.
If the source pipeline supports chunking, windowing, batching, iterative scheduling, or
another bounded-execution strategy, preserve that behavior and consume every resulting
chunk, window, iteration, or result in source-defined order. Add stricter subdivision
only when a compiled bound requires it, and retain the source merge/assembly semantics.
Do not silently clamp or truncate a complete request to one QPC call. Fail only when an
indivisible source unit exceeds the compiled contract or the source pipeline itself does
not support decomposition.

Before declaring `predict.py` complete, record the source inference entry point and show
that the script invokes it rather than only importing its helpers. When the source
supports requests larger than one compiled invocation, smoke-test an input that crosses
that bound and verify that every bounded invocation ran and was assembled. Compare the
integrated output against the unmodified source pipeline on the same input.

Each QPC wrapper may pad to the compiled upper bound internally, but must pass true
lengths/masks when the export has them (see `cloud-ai-model-conversion/references/qaic-model-requirements.md`,
"Valid-Length Inputs for Padded Variable-Length Dimensions") and
slice back to the dynamic shape the upstream caller expects. For sequential
single-device QPC stages, create all `qaic.Session`s with the same
`oversubscription_name` (ProgramGroup) so stages share NSP resources instead of adding
core requirements across stages; MDP/TS multi-SoC QPCs cannot share this way.
**Construct every group member before the first `run()` and call `.setup()` on none of
them** - the first `run()` activates the whole group. Eager per-member `setup()` fails
with `Failed to allocate NSP resources` / `Activation Reservation failed` even when each
stage fits on its own (see `cloud-ai-qaic-python`, **ProgramGroup - Python API**). Use the task templates below only for
simple models without richer source orchestration or for the specific two-stage vision
patterns they cover.

**Add `--warmup-calls N` (default 0) to any ProgramGroup-based multi-stage predict.py** — no template provides it. The group's first `.run()` pays the whole group's activation cost (see `cloud-ai-qaic-python`, **ProgramGroup - Python API**), inflating `latency_ms` on a single-shot run. Run the full inference N times discarding results, then time the final call.

**Routing table:**

| Task | Template | Input mode |
|---|---|---|
| YOLO / SSD / EfficientDet / DETR | `predict_detection.py` | image |
| Classification | `predict_classification.py` | image |
| OCR / single-stage text recognition (one QPC; decodes text from input image — input should already show the text region; no detection stage) | `predict_ocr.py` | input matching the model's training distribution (pre-localized text region) |
| Encoder-decoder image-to-text (two QPCs: image encoder + autoregressive BPE decoder; models based on VisionEncoderDecoderModel — TrOCR, Donut, Pix2Struct, etc.) | `predict_two_stage.py` PATH B | image |
| Two-stage detect-then-recognize (two or more compiled QPC dirs; Stage 1 detects regions, Stage 2 decodes label per region) | `predict_two_stage.py` PATH A | input matching the **primary** stage's training distribution (full-scene, not pre-cropped) |
| Semantic or Instance segmentation | `predict_segmentation.py` — keep Semantic block, Instance Path A (query-based: Mask2Former/MaskDINO), or Instance Path B (anchor-based: Mask R-CNN/SOLO); delete the other two | image |
| Pose estimation (keypoints + skeleton) | `predict_pose.py` | image |
| Monocular depth estimation | `predict_depth.py` | image |
| 3D geometry estimation / point cloud (RGB image in; XYZ point coordinates out — per-pixel spatial grid or unstructured point set; may include a validity mask) | `predict_3d.py` — spatial grid models (H×W output, mask present): use PATH A (full grid + mask_b64 + grid_h/w); unstructured point-set models (N points, no pixel correspondence): use PATH B (subsampled + n_points) | image |
| Face detection with landmarks | `predict_face.py` | image |
| Text classification / NLP (QPC takes tokenized text) | Derive from `QAICPredictor`; override `predict_text(text)` — `prediction` must be `[{"label": str, "score": float}, ...]` sorted by score descending (HuggingFace pipeline standard); scores must be softmax probabilities in [0,1], never raw logits | text |
| Text-to-speech / speech generation / any text-in, audio-out model | Multi-stage pipeline: use the upstream-orchestration pattern above, not `QAICPredictor`. Single QPC: derive from `QAICPredictor`, override `predict_text(text)`. **See the [TTS and ASR contracts](audio-and-generative-contracts.md) before writing predict.py.** | text (`--text`; WAV written to `--output`); set `input_type: "tts"` and `reference_voices` in demo_config.json |
| ASR / audio classification / any waveform-input model | Derive from `QAICPredictor`; override `predict_audio(audio_path)` — receive local WAV/MP3/FLAC/OGG path, preprocess to model's expected format (e.g. 16 kHz mono float32), run inference, return `{'prediction': <model-specific>, 'latency_ms': float}` | audio |
| Any other task | Derive from `QAICPredictor`; implement `preprocess`, `postprocess`, and `visualize` — use the HuggingFace pipeline output format for that task type | image |

**TTS/flow-matching shape budgeting:** if the model conditions on `ref_text + gen_text` inside a fixed-length sequence window (a compiled QPC's static shape), size that window for real sentences, not a short smoke-test phrase. A too-small text-token budget silently truncates the input text after tokenization — the model still runs and returns plausible-looking output, but it mostly or entirely reproduces the reference text instead of the requested text, and a too-small combined ref+generated frame budget silently caps output duration regardless of input length. Both fail without an error: no exception, no truncation warning surfaced to the caller. Before fixing shapes at export time, measure the token/frame count for your real reference audio/text plus a representative multi-sentence request, then size the export shape (and re-check REF_MAX_FRAMES-style host-side truncation constants) to fit comfortably above that, not the default from a short trial run.

```bash
cp "$(find -L /skills "$HOME/.agents" "$HOME/.claude" .agents .claude -path "*/cloud-ai-vision-demo/templates/predict_<task>.py" 2>/dev/null | head -1)" \
   /workspace/<model>/predict.py
cp "$(find -L /skills "$HOME/.agents" "$HOME/.claude" .agents .claude -path "*/cloud-ai-vision-demo/templates/predict_base.py" 2>/dev/null | head -1)" \
   /workspace/<model>/predict_base.py
```

**Read the fill-in section** (predict.py contains only the fill-in section — no base class):
```bash
grep -n "START FILL-IN" /workspace/<model>/predict.py
# everything after that marker is the ModelPredictor stub to be filled in
```

**Write ONLY the ModelPredictor class — never modify predict_base.py:**
```bash
python3 - << 'EOF'
path = "/workspace/<model>/predict.py"
content = open(path).read()
marker = "# ── START FILL-IN SECTION"
prefix = content[:content.index(marker)]
new_suffix = r"""# ── START FILL-IN SECTION ─────────────────────────────────────────────────────
class ModelPredictor(QAICPredictor):
    # FILL_IN all constants and preprocess/postprocess below
    pass

if __name__ == "__main__":
    ModelPredictor().main()
"""
open(path, "w").write(prefix + new_suffix)
EOF
```
Replace the `new_suffix` body with the actual filled-in `ModelPredictor` class.

**Immediately verify the file was not truncated:**
```bash
python3 -m py_compile /workspace/<model>/predict.py && echo "syntax OK"
```
If this fails with `SyntaxError: EOF while scanning string literal` at the last line — the heredoc was truncated. Rewrite the file.

**Never modify `predict_base.py` — `visualize()`, `_nms()`, and `_resolve_qpc()` live there and are correct as-is.**

**Verify the write is correct:**
```bash
python3 -c "
import ast
# QAICPredictor lives in predict_base.py (imported, not defined in predict.py)
base_classes = [n.name for n in ast.walk(ast.parse(open('/workspace/<model>/predict_base.py').read())) if isinstance(n, ast.ClassDef)]
assert 'QAICPredictor' in base_classes, 'ERROR: QAICPredictor missing from predict_base.py!'
# predict.py must have ModelPredictor but must NOT inline QAICPredictor
classes = [n.name for n in ast.walk(ast.parse(open('/workspace/<model>/predict.py').read())) if isinstance(n, ast.ClassDef)]
assert 'ModelPredictor' in classes, 'ERROR: ModelPredictor missing from predict.py!'
assert 'QAICPredictor' not in classes, 'ERROR: QAICPredictor inlined in predict.py — base class was overwritten!'
print('OK — predict_base.py:', base_classes, '| predict.py:', classes)
"
```

**⚠ Rules:**
- Fill in ONLY the constants and `preprocess`/`postprocess` stubs.
- Do NOT touch `visualize()`, `_nms()`, the base class, or `COLORS`.
- `QPC_DIR` must use `os.path.join(os.path.dirname(os.path.abspath(__file__)), "qpc_Xc")`.

---

## Fill in the template stubs

Fill in every `FILL_IN` marker in `ModelPredictor`. Each method's docstring describes exactly what it must do — read those before writing any code. Key reminders:

- `LABELS_URL = "COCO_80"` or `"COCO_91"` activates built-in label lists (no download, no override needed).
- For custom datasets: set `LABELS_URL` to a real URL and `LABELS_TMP` to a unique `/tmp/` path.
- **Detection only:** delete the Path A or Path B `ModelPredictor` class that does not apply.
- **Classification only:** `visualize()` must stay as bare `return` — do not implement it.

## Multi-stage pipeline integration

Use when compilation produced two or more QPC directories — one for the primary stage (localizer: detector or segmenter) and one or more for secondary stages (recognizer, classifier, or text decoder applied per detected region).

Copy `predict_two_stage.py` as the starting template using the copy command above. Its method docstrings describe every fill-in step. The section below highlights the two constraints most likely to produce silent failures.

**Mandatory: verify secondary stage input format before writing `_run_secondary` preprocess**

Every stage's input format has four independent axes; wrong values on any axis produce degenerate output silently (e.g., every sequence argmax = pad/blank index; no error raised):

| Axis | Possibilities |
|---|---|
| Layout | NCHW, NHWC |
| Channels | 1-channel, 3-channel grayscale stack, 3-channel RGB |
| Normalization | raw `[0, 255]`, normalized `[0, 1]`, ImageNet-normalized, custom mean/std |
| Resolution | per the model's documented input shape — confirm against source |

Sweep `(layout, channels, normalization)` with one fixed resolution using a real crop (not random noise); the combination whose output is non-degenerate is the correct preprocess. If every combination produces degenerate output, the resolution is wrong — check the source repo's reference inference code.

```python
# FILL_IN: session_path, input_key, output_key, H, W, seq_len
import qaic, numpy as np
from PIL import Image

def _run(arr, session_path, input_key, output_key):
    s = qaic.Session(model_path=session_path, num_activations=1); s.setup()
    return s.run({input_key: np.ascontiguousarray(arr, dtype=np.float32)})[output_key]

crop = Image.open("FILL_IN_real_image.jpg").crop((FILL_IN_x1, FILL_IN_y1, FILL_IN_x2, FILL_IN_y2))
for channels in ["L", "RGB"]:
    img = crop.resize((FILL_IN_W, FILL_IN_H)).convert(channels)
    arr_raw = np.array(img, dtype=np.float32)
    arr_norm = arr_raw / 255.0
    for norm, arr in [("raw", arr_raw), ("norm", arr_norm)]:
        for layout in ["NHWC", "NCHW"]:
            if channels == "L":
                a = np.stack([arr, arr, arr], axis=-1)[np.newaxis]   # (1,H,W,3)
            else:
                a = arr[np.newaxis]                                    # (1,H,W,3)
            if layout == "NCHW":
                a = a.transpose(0, 3, 1, 2)                            # (1,3,H,W)
            out = _run(a, "FILL_IN_secondary_qpc/programqpc.bin", "FILL_IN_input_key", "FILL_IN_output_key")
            print(f"channels={channels} norm={norm} layout={layout}: argmax={np.argmax(out.reshape(FILL_IN_seq_len, -1), axis=-1).tolist()}")
# Non-degenerate output (argmax varies across positions, not all pad/blank) = correct format.
```

**Session access: build both sessions in `__init__` with the same `oversubscription_name`**

```python
# Correct — both stages share NSPs via ProgramGroup (max of their requirements, not sum).
# This is the pattern already in predict_two_stage.py __init__; fill in the QPC dirs
# and the group name. _resolve_qpc() is the base-class helper that appends
# programqpc.bin and anchors relative paths next to predict.py.
def __init__(self):
    import qaic
    self.detector = qaic.Session(
        model_path=self._resolve_qpc("FILL_IN_detector_qpc"),
        num_activations=1, oversubscription_name="pipeline")
    self.recognizer = qaic.Session(
        model_path=self._resolve_qpc("FILL_IN_recognizer_qpc"),
        num_activations=1, oversubscription_name="pipeline")  # same name = same group
    self._warmed_up = False
    # No .setup() on either session — see the no-setup() rule above.

# Wrong — omitting oversubscription_name creates an independent session that holds
# NSPs exclusively (may overflow device)
s = qaic.Session(model_path=self._resolve_qpc("FILL_IN_recognizer_qpc"),
                 num_activations=1)
```

Both sessions must pass the **same** `oversubscription_name` string; nothing sets it for you. Override `_forward()` to orchestrate the named sessions (its base-class docstring describes this) — do not add a session-factory helper. Single-device QPCs only — MDP/TS QPCs cannot use `oversubscription_name`.

**Smoke test: use a real test image from the source repo**

Generic fixtures (`smoke_test.jpg`, random `input.raw`) do not match the primary stage's training distribution and produce zero detections — the secondary stage is never invoked and the result is empty JSON. Empty output from a generic fixture is an input-distribution mismatch, not a passing smoke test. Use a real test asset from the source repo (`src/<model>/assets/` or referenced by the source's example inference script).

**Common pitfall — input distribution mismatch:** the primary detector is typically trained on full scenes and expects the target at realistic scale within scene context. Using a pre-cropped or synthesized image that isolates just the target (e.g., a close-up plate crop for a license-plate pipeline) is the most common source of silent empty output — the detector produces no boxes because the input doesn't match its training distribution. Use a test image from the source repo that reflects the model's actual input distribution.
