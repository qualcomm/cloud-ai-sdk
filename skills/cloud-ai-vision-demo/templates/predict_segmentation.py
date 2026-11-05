#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> segmentation inference on Cloud AI 100.

TEMPLATE: cp this file to /workspace/<model>/predict.py, fill every FILL_IN marker,
keep the matching sub-type block (Semantic OR Instance), delete the other, then smoke-test.
Do NOT rewrite visualize() — it is correct as-is."""

import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict_base import QAICPredictor, hex_to_rgb

# ── Segmentation helpers ──────────────────────────────────────────────────────
COLORS = [
    "#FF3333", "#33FF33", "#3399FF", "#FF9900", "#CC33FF",
    "#FF66CC", "#00CCCC", "#FFFF33", "#FF6633", "#33CCFF",
]

# Fixed-seed palette for semantic segmentation (256 colours, stable across calls)
_PALETTE = np.random.RandomState(42).randint(0, 256, (256, 3), dtype=np.uint8)



# ── START FILL-IN SECTION ─────────────────────────────────────────────────────

# ┌─ Semantic segmentation (DeepLabV3+, SegFormer, FCN) ───────────────────────
# │  Use for: per-pixel class label output — single class map, no instances.
# │  Delete this class if the model is instance segmentation — keep the Instance block instead.
class ModelPredictor(QAICPredictor):
    INPUT_H       = 0    # FILL_IN: compiled input height (e.g. 520)
    INPUT_W       = 0    # FILL_IN: compiled input width  (e.g. 520)
    OVERLAY_ALPHA = 160

    # FILL_IN: URL to a one-label-per-line text file matching the model's class vocabulary.
    # VOC-21: 21 classes including background at index 0
    # ADE20K: 150 classes (no background); Cityscapes: 19 classes
    LABELS_URL = "FILL_IN"
    LABELS_TMP = "/tmp/FILL_IN_labels.txt"   # FILL_IN: use a unique /tmp name per model

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Resize to (INPUT_W, INPUT_H). Apply ImageNet normalization:
        mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
        Store self._orig_size = (orig_W, orig_H) for upscaling in visualize.
        Return {"input_name": NCHW float32 (1,3,INPUT_H,INPUT_W)}.
        Use self._session.model_input_names[0] for the key.
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Unpack first output. Two possible shapes:
        (1, C, H, W) — per-pixel class logits: class_map = np.argmax(output[0], axis=0) → (H, W).
        (1, H, W)    — class indices already computed (some SegFormer exports): class_map = output[0].
        Check output.ndim to distinguish; do NOT apply argmax on the batch axis.
        Store self._class_map = class_map (used by visualize).
        Use self._get_labels() for label names. Collect present classes (excluding background index 0).
        Return {'prediction': {'classes_detected': [str, ...]}, 'latency_ms': float}.
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        color_mask = _PALETTE[self._class_map]           # (H, W, 3)
        orig    = Image.open(image_path).convert("RGB")
        overlay = Image.fromarray(color_mask.astype(np.uint8)).resize(orig.size, Image.NEAREST)
        blended = Image.blend(orig, overlay.convert("RGB"), alpha=0.5)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        blended.save(output_path, "JPEG", quality=92)
# └─────────────────────────────────────────────────────────────────────────────


# ┌─ Instance segmentation Path A: Query-based (Mask2Former, MaskDINO) ────────
# │  Use for: models with a fixed set of object queries; output is class logits
# │  (1,Q,C+1) + mask logits (1,Q,H,W). No NMS. No-object class at index C.
# │  Delete this class if the model is anchor-based — keep Path B instead.
class ModelPredictor(QAICPredictor):  # noqa: F811
    INPUT_H       = 0    # FILL_IN: compiled input height
    INPUT_W       = 0    # FILL_IN: compiled input width
    CONF_THRESH   = 0.25  # minimum class score to accept a query
    OVERLAY_ALPHA = 160

    # COCO instance segmentation (Mask2Former, Mask R-CNN trained on COCO): LABELS_URL = "COCO_80"
    # Models with 91-class output (original COCO category IDs): LABELS_URL = "COCO_91"
    # Other datasets: set to a URL returning one label per line, set LABELS_TMP to a unique /tmp path
    LABELS_URL = "FILL_IN"
    LABELS_TMP = "/tmp/FILL_IN_labels.txt"   # FILL_IN: use a unique /tmp name per model

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Resize to (INPUT_W, INPUT_H). Apply ImageNet normalization:
        mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
        Store self._orig_w, self._orig_h.
        Return {"input_name": NCHW float32 (1,3,INPUT_H,INPUT_W)}.
        Use self._session.model_input_names[0] for the key.
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Identify class logits (1,Q,C+1) and mask logits (1,Q,H_enc,W_enc) by tensor shape.
        Apply softmax over ALL C+1 class logits (including no-object at index C) — do NOT
        slice off the no-object column before softmax; doing so forces every query to pick
        a foreground label and produces num_instances == Q (all queries firing).

        For each query:
          1. cls_id = argmax over softmax — if == C (no-object index), skip.
          2. score = softmax value at cls_id — if < CONF_THRESH, skip.
          3. sigmoid(mask_logit[q]) > 0.5 → binary mask at encoder resolution (H_enc, W_enc).
          4. Upsample binary mask to original image size using Image.NEAREST (NOT BILINEAR).
             Encode as 0/255 uint8 before upsampling so NEAREST gives clean boolean result.
             mask_img = Image.fromarray((mask * 255).astype(np.uint8))
             mask_resized = np.array(mask_img.resize((self._orig_w, self._orig_h), Image.NEAREST)) > 128
          5. Append {'label': str, 'score': float} and the resized bool array to lists.

        Use self._get_labels() for label names (check class dim: C=80→COCO_80, C=91→COCO_91).
        Sort instances by score descending.
        Store self._instance_masks (list of bool arrays, same order as instances).
        Return {'prediction': {'instances': [{'label': str, 'score': float}, ...],
                               'num_instances': int}, 'latency_ms': float}.
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        orig    = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", orig.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(orig)
        short_side = min(orig.width, orig.height)
        font_size  = max(20, short_side // 40)
        font = self.get_font(font_size)
        labels_to_draw = []
        for i, (mask_arr, inst) in enumerate(
                zip(self._instance_masks, prediction["instances"])):
            color = COLORS[i % len(COLORS)]
            r, g, b = hex_to_rgb(color)
            alpha = np.zeros((orig.height, orig.width), dtype=np.uint8)
            alpha[mask_arr] = self.OVERLAY_ALPHA
            color_layer = Image.new("RGBA", orig.size, (r, g, b, 0))
            color_layer.putalpha(Image.fromarray(alpha))
            overlay = Image.alpha_composite(overlay, color_layer)
            ys, xs = np.where(mask_arr)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"{inst['label']}: {inst['score']:.2f}"
                try:
                    bbox = draw.textbbox((cx, cy), text, font=font)
                    labels_to_draw.append((cx, cy, text, (r, g, b), bbox))
                except Exception:
                    labels_to_draw.append((cx, cy, text, (r, g, b), None))
        result     = Image.alpha_composite(orig, overlay)
        final_draw = ImageDraw.Draw(result)
        pad = max(4, font_size // 6)
        for cx, cy, text, (r, g, b), bbox in labels_to_draw:
            if bbox:
                final_draw.rectangle(
                    [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                    fill=(r, g, b, 200))
            final_draw.text((cx, cy), text, fill="white", font=font)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        result.convert("RGB").save(output_path, "JPEG", quality=92)
# └─────────────────────────────────────────────────────────────────────────────


# ┌─ Instance segmentation Path B: Anchor-based (Mask R-CNN, SOLO, CondInst) ──
# │  Use for: models with NMS embedded in ONNX; output is named detection tensors
# │  (boxes, labels, scores, masks). Scores already activated [0,1] — no softmax.
# │  Delete this class if the model is query-based — keep Path A instead.
class ModelPredictor(QAICPredictor):  # noqa: F811
    INPUT_H       = 0    # FILL_IN: compiled input height
    INPUT_W       = 0    # FILL_IN: compiled input width
    CONF_THRESH   = 0.5
    OVERLAY_ALPHA = 160

    LABELS_URL = "COCO_80"  # FILL_IN: "COCO_80" for COCO-trained; URL for other datasets
    LABELS_TMP = ""

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Resize to (INPUT_W, INPUT_H). Apply ImageNet normalization:
        mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
        Store self._orig_w, self._orig_h and scale factors for box undo.
        Return {"input_name": NCHW float32 (1,3,INPUT_H,INPUT_W)}.
        Use self._session.model_input_names[0] for the key.
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Outputs are named tensors — locate by name from session.model_output_names:
          boxes:  (N, 4) or (1, N, 4) — xyxy pixel coords in input-image space
          labels: (N,)   or (1, N)    — integer class indices (0-indexed)
          scores: (N,)   or (1, N)    — confidence [0,1], already activated, no softmax
          masks:  (N, 1, H_mask, W_mask) or (N, H_mask, W_mask) — raw mask logits per instance
        H_mask x W_mask is the model's mask head output size (e.g. 28x28 for Mask R-CNN,
        varies by model — do NOT hardcode). Apply sigmoid and threshold at 0.5.
        Filter detections by CONF_THRESH.
        Undo input resize: scale boxes by (orig_w/INPUT_W, orig_h/INPUT_H).
        IMPORTANT — anchor-based mask scope: each mask covers its bounding box region only,
        not the full image. Resize the binary mask to (box_w, box_h) in original-image coords
        using Image.BILINEAR, then paste into a zeros canvas of shape (orig_h, orig_w) at (x1, y1).
        Do NOT resize directly to (orig_w, orig_h) — that maps a box-crop mask onto the whole image.
        Pattern:
            box_w, box_h = x2 - x1, y2 - y1
            mask_box = np.array(Image.fromarray(binary_mask).resize((box_w, box_h), Image.BILINEAR)) > 128
            full_mask = np.zeros((self._orig_h, self._orig_w), dtype=bool)
            full_mask[y1:y2, x1:x2] = mask_box
        Store self._instance_masks (list of bool arrays shape (orig_h, orig_w), same order as instances).
        Use self._get_labels() for label names.
        Return {'prediction': {'instances': [{'label': str, 'score': float}, ...],
                               'num_instances': int}, 'latency_ms': float}.
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        orig    = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", orig.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(orig)
        short_side = min(orig.width, orig.height)
        font_size  = max(20, short_side // 40)
        font = self.get_font(font_size)
        labels_to_draw = []
        for i, (mask_arr, inst) in enumerate(
                zip(self._instance_masks, prediction["instances"])):
            color = COLORS[i % len(COLORS)]
            r, g, b = hex_to_rgb(color)
            alpha = np.zeros((orig.height, orig.width), dtype=np.uint8)
            alpha[mask_arr] = self.OVERLAY_ALPHA
            color_layer = Image.new("RGBA", orig.size, (r, g, b, 0))
            color_layer.putalpha(Image.fromarray(alpha))
            overlay = Image.alpha_composite(overlay, color_layer)
            ys, xs = np.where(mask_arr)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
                text = f"{inst['label']}: {inst['score']:.2f}"
                try:
                    bbox = draw.textbbox((cx, cy), text, font=font)
                    labels_to_draw.append((cx, cy, text, (r, g, b), bbox))
                except Exception:
                    labels_to_draw.append((cx, cy, text, (r, g, b), None))
        result     = Image.alpha_composite(orig, overlay)
        final_draw = ImageDraw.Draw(result)
        pad = max(4, font_size // 6)
        for cx, cy, text, (r, g, b), bbox in labels_to_draw:
            if bbox:
                final_draw.rectangle(
                    [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                    fill=(r, g, b, 200))
            final_draw.text((cx, cy), text, fill="white", font=font)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        result.convert("RGB").save(output_path, "JPEG", quality=92)
# └─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    ModelPredictor().main()
