#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> multi-QPC inference on Cloud AI 100.

Two pipeline patterns — keep the one that matches your model, delete the other:

PATH A — Detect-then-recognize (two QPCs: detector + per-region recognizer)
  Stage 1 (primary):   localizes regions (bounding boxes) in the full scene.
  Stage 2 (secondary): recognition/classification applied per detected region.
  Use cases: license plate recognition, text spotting (detect box -> OCR text),
             face recognition (detect face -> classify identity), logo recognition.

PATH B — Encoder-decoder image-to-text (two QPCs: image encoder + autoregressive decoder)
  Stage 1 (encoder): image -> encoder_hidden_states (ViT, BEiT, CNN, etc.)
  Stage 2 (decoder): autoregressive BPE decode using encoder_hidden_states.
  Use cases: TrOCR (printed/handwritten OCR), Donut (document -> JSON),
             Pix2Struct (chart/table -> text), any VisionEncoderDecoderModel.

TEMPLATE: cp this file to /workspace/<model>/predict.py, fill every FILL_IN marker,
then smoke-test before generating the MODEL_CARD.
DELETE the class you are not using. Each path is a self-contained class — no other
code needs to change when you delete one. Rename the kept class to ModelPredictor.

Session management (both paths):
  Both QPCs share NSPs via ProgramGroup (oversubscription_name=_PIPELINE_GROUP).
  Resources reserved = max(stage1_NSPs, stage2_NSPs), not the sum.
  Both QPCs share NSPs via ProgramGroup (oversubscription_name set in __init__).

PATH A fill-in guide:
  Step A — Set all FILL_IN constants. _SECONDARY_VOCAB from source repo tokenizer/head
            config. Leave "" if the secondary is a classifier (not sequence output).
  Step B — Verify secondary stage input format BEFORE writing _run_secondary preprocess.
            See references/template-routing-and-integration.md "Multi-stage pipeline
            integration" for the mandatory 4-axis verification snippet (layout x
            channels x normalization x resolution).
            Wrong normalization produces silent degenerate output with no error raised.
  Step C — Fill in preprocess(): letterbox primary input, store transform state on self.
  Step D — Fill in _run_secondary(): crop -> format per Step B -> run session -> decode.
  Step E — Fill in postprocess(): parse primary output, undo letterbox, call _run_secondary
            per box. Labels MUST be secondary decoded output, never static class names.

PATH B fill-in guide:
  Step A — Set QPC_DIR (encoder), DECODER_QPC_DIR (decoder), INPUT_H/W, SEQ_LEN.
  Step B — Set token IDs from generation_config.json in the hf_cache snapshot:
            cat hf_cache/hub/models--<org>--<model>/snapshots/<sha>/generation_config.json
  Step C — Confirm encoder output key and decoder input/output key names:
            python3 -c "import qaic; s=qaic.Session('encoder_qpc/programqpc.bin',
            num_activations=1); print(s.model_input_shape_dict,
            s.model_output_shape_dict)"
  Step D — Fill in preprocess() and postprocess(). _forward() is complete as-is.
  Step E — Smoke-test with a real image that matches the model's training distribution.
            Do NOT use a synthetically generated PIL image — see
            references/smoke-tests-and-known-answer-validation.md for guidance on
            what constitutes a valid test image per model type.

(_nms() and all base methods live in predict_base.py — never modify that file.)"""

import os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict_base import QAICPredictor, hex_to_rgb  # hex_to_rgb used by PATH A only; remove if using PATH B


# ── PATH A — Detect-then-recognize ───────────────────────────────────────────
# DELETE this entire class if using PATH B below.

class ModelPredictorDetectRecognize(QAICPredictor):
    # ── Stage 1 (detector) ───────────────────────────────────────────────────
    INPUT_H        = 0     # FILL_IN: compiled input height (e.g. 384)
    INPUT_W        = 0     # FILL_IN: compiled input width  (e.g. 384)
    CONF_THRESH    = 0.45
    NMS_IOU_THRESH = 0.45
    OVERLAY_ALPHA  = 160
    LABELS_URL     = None
    LABELS_TMP     = ""

    # ── Stage 2 (recognizer) ─────────────────────────────────────────────────
    RECOGNIZER_H   = 0     # FILL_IN: recognizer input height (e.g. 64)
    RECOGNIZER_W   = 0     # FILL_IN: recognizer input width  (e.g. 128)

    # FILL_IN: recognizer vocabulary in index order, e.g. '0123456789ABC_'
    # (last char = CTC blank/pad). Leave "" if recognizer outputs class probs.
    _RECOGNIZER_VOCAB = "FILL_IN"

    _LABELS = ["FILL_IN"]
    _COLORS = [
        "#FF3333", "#33FF33", "#3399FF", "#FF9900", "#CC33FF",
        "#FF66CC", "#00CCCC", "#FFFF33", "#FF6633", "#33CCFF",
    ]

    def __init__(self):
        import qaic  # noqa: PLC0415
        self.detector   = qaic.Session(
            model_path=self._resolve_qpc("FILL_IN_detector_qpc"),
            num_activations=1, oversubscription_name="pipeline")
        self.recognizer = qaic.Session(
            model_path=self._resolve_qpc("FILL_IN_recognizer_qpc"),
            num_activations=1, oversubscription_name="pipeline")
        self._warmed_up = False

    def _get_labels(self):
        return self._LABELS

    def preprocess(self, image_path: str) -> dict:
        """Letterbox-resize to (INPUT_W, INPUT_H) on a grey (114,114,114) canvas.
        Normalize to [0,1] float32. Return {"input_name": NCHW array (1,3,H,W)}.
        Also store on self for _forward coordinate unmap:
          self._orig_w, self._orig_h      — original image dimensions
          self._letterbox_scale           — uniform scale applied
          self._letterbox_offset          — (off_x, off_y) padding added
          self._orig_image                — original PIL Image (for recognizer crops)
        """
        raise NotImplementedError

    def _forward(self, model_inputs: dict) -> dict:
        """Run detector on full image, then per-box: crop -> recognizer -> decode label.

        FILL_IN this method:
          Stage 1 - detector:
            det_out = self.detector.run({k: np.ascontiguousarray(v) for k, v in model_inputs.items()})
            rows = det_out["FILL_IN_key"].reshape(-1, 7)
              row format: [class_id, x1, y1, x2, y2, unused, conf]
              (see predict_detection.py for YOLO-raw / DETR variants)

          Per box (skip if conf < CONF_THRESH or degenerate geometry):
            Undo letterbox (using self._letterbox_scale, self._letterbox_offset):
              x1_orig = max(0, (x1 - off_x) / scale)
              y1_orig = max(0, (y1 - off_y) / scale)
              x2_orig = min(self._orig_w, (x2 - off_x) / scale)
              y2_orig = min(self._orig_h, (y2 - off_y) / scale)

            Crop: crop = self._orig_image.crop((x1_orig, y1_orig, x2_orig, y2_orig))

          Stage 2 - recognizer (verify layout/channels/normalization before coding):
            Resize crop to (RECOGNIZER_W, RECOGNIZER_H).
            Normalize. Build NCHW or NHWC input array.
            rec_name = self.recognizer.model_input_names[0]
            rec_out = self.recognizer.run({rec_name: np.ascontiguousarray(crop_arr)})
            Decode:
              Sequence (CTC): argmax over vocab dim, collapse repeats, strip blank,
                lookup in _RECOGNIZER_VOCAB.
              Classifier: softmax -> argmax -> label from _get_labels().

          Return:
            {"_boxes":  [[x1,y1,x2,y2], ...],   # original-image coords
             "_labels": [str, ...],
             "_scores": [float, ...]}

        NOTE: np.ascontiguousarray() is mandatory before every session.run() call.
        PIL->numpy->transpose->newaxis produces non-contiguous views; qaic.Session.run()
        silently returns empty/wrong output without raising an exception.
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        return {
            "prediction": {
                "boxes":  outputs["_boxes"],
                "labels": outputs["_labels"],
                "scores": outputs["_scores"],
            },
            "latency_ms": latency_ms,
        }

    def visualize(self, image_path: str, prediction, output_path: str):
        self._draw_detections(image_path, prediction, output_path,
                              self.OVERLAY_ALPHA, self.get_font, self._COLORS)

    @staticmethod
    def _draw_detections(image_path, prediction, output_path, overlay_alpha, get_font, colors):
        """Shared visualize implementation. Do NOT modify."""
        img     = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(img)
        ov_draw = ImageDraw.Draw(overlay)
        short_side = min(img.width, img.height)
        font_size  = max(24, short_side // 30)
        boxes = prediction.get("boxes", [])
        if boxes:
            min_box_h = min(max(1, b[3] - b[1]) for b in boxes)
            font_size = max(12, min(font_size, int(min_box_h * 0.6)))
        box_width  = max(2, min(4, short_side // 200))
        pad        = max(3, font_size // 4)
        labels = prediction.get("labels", [])
        font = get_font(font_size, text=" ".join(labels))
        labels_to_draw = []
        for i, (box, label, score) in enumerate(zip(
                prediction.get("boxes", []), labels, prediction.get("scores", []))):
            r, g, b = hex_to_rgb(colors[i % len(colors)])
            x1, y1, x2, y2 = box
            ov_draw.rectangle([x1, y1, x2, y2],
                              outline=(r, g, b, overlay_alpha), width=box_width)
            text = f"{label}: {score:.0%}"
            lx = x1 + pad
            try:
                _mbox = draw.textbbox((lx, 0), text, font=font)
                label_h = (_mbox[3] - _mbox[1]) + pad * 2
            except Exception:
                label_h = font_size + pad * 2
            ly = max(0, y1 - label_h)
            try:
                bbox = draw.textbbox((lx, ly), text, font=font)
                ov_draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                                  fill=(r, g, b, overlay_alpha))
            except Exception:
                pass
            labels_to_draw.append((lx, ly, text))
        img = Image.alpha_composite(img, overlay)
        final_draw = ImageDraw.Draw(img)
        for lx, ly, text in labels_to_draw:
            final_draw.text((lx, ly), text, fill="white", font=font)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.convert("RGB").save(output_path, "JPEG", quality=92)


# ── PATH B — Encoder-decoder image-to-text ────────────────────────────────────
# DELETE this entire class if using PATH A above.

class ModelPredictorEncDec(QAICPredictor):
    # ── Stage 1: Image Encoder ───────────────────────────────────────────────
    INPUT_H         = 0    # FILL_IN: compiled encoder input height (e.g. 384)
    INPUT_W         = 0    # FILL_IN: compiled encoder input width  (e.g. 384)

    # ── Stage 2: Autoregressive Text Decoder ─────────────────────────────────
    SEQ_LEN         = 0    # FILL_IN: compiled decoder sequence length (e.g. 128)

    # ── Token IDs — read from generation_config.json in the hf_cache snapshot
    DECODER_START_TOKEN_ID = 0   # FILL_IN: e.g. 2 for TrOCR (</s>)
    EOS_TOKEN_ID           = 0   # FILL_IN: e.g. 2 for TrOCR
    PAD_TOKEN_ID           = 0   # FILL_IN: e.g. 1 for TrOCR (<pad>)

    # ── Normalization: (pixel/255 - MEAN) / STD ──────────────────────────────
    # FILL_IN MEAN and STD from the model's documented preprocessing — export
    # script, README, model card, image_processor_config.json, or reference
    # inference script. Do NOT assume values from architecture family name alone.
    MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # FILL_IN
    STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # FILL_IN

    OVERLAY_ALPHA = 200
    LABELS_URL    = None
    LABELS_TMP    = ""
    _tokenizer    = None  # cached at class level

    def __init__(self):
        import qaic  # noqa: PLC0415
        self.encoder = qaic.Session(
            model_path=self._resolve_qpc("FILL_IN_encoder_qpc"),
            num_activations=1, oversubscription_name="FILL_IN_pipeline_name")
        self.decoder = qaic.Session(
            model_path=self._resolve_qpc("FILL_IN_decoder_qpc"),
            num_activations=1, oversubscription_name="FILL_IN_pipeline_name")
        self._warmed_up = False

    def preprocess(self, image_path: str) -> dict:
        """Resize to INPUT_W x INPUT_H (BILINEAR), normalize, return {"pixel_values": NCHW array}."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.INPUT_W, self.INPUT_H), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD
        return {"pixel_values": np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis])}

    def _forward(self, model_inputs: dict) -> dict:
        """Run encoder then autoregressive decoder; return {"_generated_ids": [...]}."""
        enc_hidden = self.encoder.run({k: np.ascontiguousarray(v) for k, v in model_inputs.items()})["FILL_IN_encoder_output_key"]  # e.g. "encoder_hidden_states"

        input_ids = np.full((1, self.SEQ_LEN), self.PAD_TOKEN_ID, dtype=np.int64)
        input_ids[0, 0] = self.DECODER_START_TOKEN_ID
        generated_ids = []
        for step in range(self.SEQ_LEN - 1):
            logits = self.decoder.run({
                "FILL_IN_enc_hidden_key": np.ascontiguousarray(enc_hidden),
                "input_ids": np.ascontiguousarray(input_ids),
            })["FILL_IN_logits_key"]                    # e.g. "logits" (1, SEQ_LEN, vocab)
            next_id = int(np.argmax(logits[0, step]))
            if next_id == self.EOS_TOKEN_ID:
                break
            generated_ids.append(next_id)
            input_ids[0, step + 1] = next_id

        return {"_generated_ids": generated_ids}

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Decode token IDs to text string.

        FILL_IN: replace AutoTokenizer with the correct tokenizer class and model ID.
        """
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
        from transformers import AutoTokenizer
        if self.__class__._tokenizer is None:
            try:
                self.__class__._tokenizer = AutoTokenizer.from_pretrained(
                    "FILL_IN_hf_model_id", cache_dir=cache_dir, local_files_only=True)
            except Exception:
                self.__class__._tokenizer = AutoTokenizer.from_pretrained(
                    "FILL_IN_hf_model_id", cache_dir=cache_dir)
        text = self.__class__._tokenizer.decode(
            outputs.get("_generated_ids", []), skip_special_tokens=True).strip()
        return {"prediction": text, "latency_ms": latency_ms}

    def visualize(self, image_path: str, prediction, output_path: str):
        """Overlay recognized text on the image."""
        img     = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(img)
        ov_draw = ImageDraw.Draw(overlay)
        short_side = min(img.width, img.height)
        font_size  = max(28, short_side // 24)
        pad        = max(8,  short_side // 60)
        label = str(prediction) if prediction else "(no text)"
        font = self.get_font(font_size, text=label)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = font_size * len(label) // 2, font_size
        w, h = img.size
        ry0 = h - th - 2 * pad
        ov_draw.rectangle([0, ry0, min(tw + 2 * pad, w), h],
                          fill=(0, 0, 0, self.OVERLAY_ALPHA))
        img = Image.alpha_composite(img, overlay)
        ImageDraw.Draw(img).text((pad, ry0 + pad), label, fill=(0, 255, 80), font=font)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.convert("RGB").save(output_path, "JPEG", quality=92)


# Rename the kept class to ModelPredictor before using.
ModelPredictor = ModelPredictorDetectRecognize  # FILL_IN: swap to ModelPredictorEncDec if using PATH B

if __name__ == "__main__":
    ModelPredictor().main()
