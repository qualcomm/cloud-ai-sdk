---
name: cloud-ai-vision-demo
description: Use when a Cloud AI 100 model compiled to QPC needs a predict.py demo or has preprocessing, postprocessing, output-contract, or multi-stage integration requirements.
---

# Cloud AI 100 Vision Demo Script

## Purpose

Create a relocatable `predict.py` after QPC compilation. It must use packages
already available in the runtime, invoke the compiled QPC through `qaic.Session`,
and emit exactly one JSON object on stdout with a structured `prediction` and
`latency_ms`. Image demos also write the requested annotated image; TTS demos
write the requested WAV.

This skill covers image, text, audio, OCR, detection, segmentation, pose, depth,
3D, face, ASR, and TTS demos. It does not replace model export or QPC compilation.

## Activation and universal contract

Use this skill when the model already has a usable QPC directory and the task is
to build or repair the host-side demo. Before editing, identify:

- `input_shape`, `input_dtype`, normalization, and output tensor metadata from the
  export artifacts and `qaic.Session`;
- the best QPC directory from the core-count sweep;
- the model task and whether one request uses one QPC or a source-controlled
  multi-stage pipeline;
- the authoritative source inference entry point when the source has orchestration.

CLI entry points are:

| Task | Required input | Output |
|---|---|---|
| Image | `--image <file|url|base64>` and image tasks also `--output <jpg>` | One JSON line; image tasks write the annotated output |
| Text classification/NLP | `--text <string>` | One JSON line with task-specific prediction |
| ASR/audio | `--audio <file|url|base64>` | One JSON line with `prediction.text` for ASR |
| TTS | `--text`, `--output`, and `--voice-id` | One JSON line plus a 16-bit PCM WAV |

Keep diagnostics and progress on stderr. Do not silently truncate a request to a
compiled bound. A QPC shape bounds one invocation; source-supported chunking,
windowing, batching, and iterative generation still need to consume and assemble
the complete request.

## Routing table

Choose a template only after checking the model's actual outputs and source
pipeline. Read every reference listed for the selected row; do not load unrelated
task references.

| Model/task | Starting point | Read these references |
|---|---|---|
| YOLO, SSD, EfficientDet, or DETR | `templates/predict_detection.py` | [Template routing](references/template-routing-and-integration.md), [detection/segmentation](references/detection-and-segmentation.md), [smoke tests](references/smoke-tests-and-known-answer-validation.md), [demo config](references/demo-config-and-fonts.md) |
| Classification | `templates/predict_classification.py` | [Template routing](references/template-routing-and-integration.md), [smoke tests](references/smoke-tests-and-known-answer-validation.md), [demo config](references/demo-config-and-fonts.md) |
| Single-stage OCR/text recognition | `templates/predict_ocr.py` | [Template routing](references/template-routing-and-integration.md), [smoke tests](references/smoke-tests-and-known-answer-validation.md), [demo config](references/demo-config-and-fonts.md) |
| Encoder-decoder image-to-text | `templates/predict_two_stage.py` Path B | [Template routing](references/template-routing-and-integration.md), [smoke tests](references/smoke-tests-and-known-answer-validation.md), [audio/generative contracts](references/audio-and-generative-contracts.md), [demo config](references/demo-config-and-fonts.md) |
| Detect-then-recognize pipeline | `templates/predict_two_stage.py` Path A | [Template routing](references/template-routing-and-integration.md), [detection/segmentation](references/detection-and-segmentation.md), [smoke tests](references/smoke-tests-and-known-answer-validation.md), [demo config](references/demo-config-and-fonts.md) |
| Semantic or instance segmentation | `templates/predict_segmentation.py` | [Template routing](references/template-routing-and-integration.md), [detection/segmentation](references/detection-and-segmentation.md), [smoke tests](references/smoke-tests-and-known-answer-validation.md), [demo config](references/demo-config-and-fonts.md) |
| Pose, depth, 3D, or face landmarks | `templates/predict_pose.py`, `predict_depth.py`, `predict_3d.py`, or `predict_face.py` | [Template routing](references/template-routing-and-integration.md), [smoke tests](references/smoke-tests-and-known-answer-validation.md), [demo config](references/demo-config-and-fonts.md) |
| Text classification | Derive from `QAICPredictor`, implement `predict_text` | [Template routing](references/template-routing-and-integration.md), [demo config](references/demo-config-and-fonts.md) |
| ASR or audio classification | Derive from `QAICPredictor`, implement `predict_audio` | [Template routing](references/template-routing-and-integration.md), [audio/generative contracts](references/audio-and-generative-contracts.md), [demo config](references/demo-config-and-fonts.md) |
| TTS or speech generation | Preserve the upstream pipeline; use a single-stage template only for a genuinely single-QPC model | [Template routing](references/template-routing-and-integration.md), [audio/generative contracts](references/audio-and-generative-contracts.md), [demo config](references/demo-config-and-fonts.md) |

## Workflow

1. Count and inspect all `programqpc.bin` files. Compile missing stages before
   writing a demo. Record the source inference entry point for orchestrated models.
2. Copy the closest template. Keep `predict_base.py` unchanged; its base class,
   `visualize()`, `_nms()`, and `_resolve_qpc()` are shared implementation.
3. Fill only the intended `ModelPredictor` constants and preprocessing/postprocessing
   stubs. Use script-relative QPC paths. For multi-stage single-device sessions,
   construct every session before the first run with the same
   `oversubscription_name`; do not call `.setup()` on ProgramGroup members.
4. Inspect real output shapes and value ranges before choosing score activation,
   coordinate layout, labels, masks, or decoder handling. Validate secondary-stage
   layout, channels, normalization, and resolution with a real crop.
5. Run syntax, device smoke, preprocessing-parity, and task-specific known-answer
   checks. Use source test assets when available; generic fixtures are only fallbacks
   for the task shapes documented in the [smoke-test reference](references/smoke-tests-and-known-answer-validation.md).
6. Write `demo_config.json` beside the script. Set the correct `input_type`, keep
   runtime paths relative, and register every accepted TTS voice ID.

## Reference map

- [Template routing and integration](references/template-routing-and-integration.md): template filling, source orchestration, multi-stage integration, and ProgramGroup behavior.
- [Detection and segmentation](references/detection-and-segmentation.md): detector and mask output inspection, postprocessing, assertions, and triage.
- [Smoke tests](references/smoke-tests-and-known-answer-validation.md): fixtures, preprocessing parity, COCO checks, and encoder-decoder validation.
- [Audio and generative contracts](references/audio-and-generative-contracts.md): output cropping, TTS/ASR contracts, long audio, voices, and shape budgeting.
- [Demo configuration and fonts](references/demo-config-and-fonts.md): runtime metadata, QID declarations, TTS voice metadata, and rendering.

## Completion gates

- `predict.py` compiles and the file still contains `ModelPredictor` without an
  inlined `QAICPredictor`.
- The script invokes the compiled QPC and emits one final JSON line with a finite
  latency; task-specific prediction fields match the selected contract.
- Multi-stage and bounded-request tests prove that all required stages/chunks run
  and source ordering/assembly is preserved.
- Image, OCR, detection, segmentation, ASR, and TTS assertions from the routed
  references pass as applicable.
- `demo_config.json` is colocated, relocatable, and consistent with the CLI.

For export or ONNX issues, route to `cloud-ai-model-conversion`; for runtime API
details use `cloud-ai-qaic-python`; for numerical mismatches use
`cloud-ai-accuracy-tuning`.
