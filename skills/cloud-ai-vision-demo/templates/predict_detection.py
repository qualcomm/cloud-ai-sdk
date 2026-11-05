#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> detection inference on Cloud AI 100.

TEMPLATE: cp this file to /workspace/<model>/predict.py, fill every FILL_IN marker,
then smoke-test before generating the MODEL_CARD.

Keep the ModelPredictor class that matches your detector family; delete the other.
Do NOT modify _draw_detections() — it is correct as-is.
(_nms() and all other base methods live in predict_base.py — never modify that file.)"""

import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict_base import QAICPredictor, hex_to_rgb

# ── Detection helpers ─────────────────────────────────────────────────────────
COLORS = [
    "#FF3333", "#33FF33", "#3399FF", "#FF9900", "#CC33FF",
    "#FF66CC", "#00CCCC", "#FFFF33", "#FF6633", "#33CCFF",
]



def _draw_detections(image_path, prediction, output_path, overlay_alpha, get_font):
    """Shared visualize implementation for all detection models. Do NOT modify."""
    img     = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(img)       # used for textbbox measurement only
    ov_draw = ImageDraw.Draw(overlay)
    short_side = min(img.width, img.height)
    font_size  = max(24, short_side // 30)
    # For small-object detectors (e.g. license plates, faces) the image-proportional
    # font size can dwarf tiny boxes. Cap to 60% of the smallest box height so labels
    # always fit the objects they annotate.
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
        r, g, b = hex_to_rgb(COLORS[i % len(COLORS)])
        x1, y1, x2, y2 = box
        ov_draw.rectangle([x1, y1, x2, y2],
                          outline=(r, g, b, overlay_alpha), width=box_width)
        text = f"{label}: {score:.2f}"
        lx, ly = x1 + pad, y1 + box_width // 2
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


# ── START FILL-IN SECTION ─────────────────────────────────────────────────────

# ┌─ PATH A: YOLO / SSD / two-stage ───────────────────────────────────────────
# │  Use for: YOLOv5/v8/v11, YOLO-NAS, SSD, EfficientDet, RT-DETR (YOLO-style),
# │           Faster R-CNN, Mask R-CNN (two-stage; no separate Path exists for these).
# │  Delete this class if the model is DETR-family — keep Path B instead.
class ModelPredictor(QAICPredictor):  # noqa: F811
    INPUT_H        = 0     # FILL_IN: compiled input height (e.g. 640)
    INPUT_W        = 0     # FILL_IN: compiled input width  (e.g. 640)
    CONF_THRESH    = 0.25
    NMS_IOU_THRESH = 0.45  # verify on real images; lower if duplicate boxes survive per object
    OVERLAY_ALPHA  = 160
    LABELS_URL     = "COCO_80"  # FILL_IN: "COCO_80", or a real URL for other datasets
    LABELS_TMP     = ""         # only needed for real URLs

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Letterbox-resize to (INPUT_W, INPUT_H) on a (114,114,114) grey canvas.
        Normalize to [0,1] float32. Return {"input_name": NCHW float32 (1,3,INPUT_H,INPUT_W)}.
        Use self._session.model_input_names[0] for the key.
        Store: self._orig_w, self._orig_h, self._letterbox_scale,
               self._letterbox_offset = (off_x, off_y).
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Unpack first output — shape (1,4+C,N) or (1,N,4+C); use (4+C,N) layout.
        Score activation: if scores_raw.max() > 1.0 → per-class sigmoid (NOT softmax).
                          if scores_raw.max() ≤ 1.0 → already activated, use as-is.
        Filter by CONF_THRESH.

        xywh → x1y1x2y2 (MANDATORY — YOLO always outputs cx,cy,w,h):
            boxes_raw = out[0, :4, :]   # (4, N)
            x1 = boxes_raw[0] - boxes_raw[2] / 2
            y1 = boxes_raw[1] - boxes_raw[3] / 2
            x2 = boxes_raw[0] + boxes_raw[2] / 2
            y2 = boxes_raw[1] + boxes_raw[3] / 2

        ⚠ Anti-pattern (silent wrong boxes — smoke test still passes):
            boxes = raw[:, :4]              # WRONG: these are cx,cy,w,h
            x1 = (boxes[:,0] - off_x) / scale  # applies letterbox undo to cx

        Undo letterbox: x_orig = (x - off_x) / scale. Clip to orig dims.
        Call self._nms(x1,y1,x2,y2,scores,self.NMS_IOU_THRESH,classes=classes).
        Use self._get_labels() for label names.
        Return {'prediction':{'boxes':[[x1,y1,x2,y2],...],'labels':[str,...],'scores':[float,...]},'latency_ms':float}.
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        _draw_detections(image_path, prediction, output_path, self.OVERLAY_ALPHA, self.get_font)
# └─────────────────────────────────────────────────────────────────────────────


# ┌─ PATH B: DETR ──────────────────────────────────────────────────────────────
# │  Use for: DETR, Deformable DETR, DAB-DETR, DINO-DETR.
# │  Delete this class if the model is YOLO/SSD-family — keep Path A instead.
class ModelPredictor(QAICPredictor):  # noqa: F811
    INPUT_H       = 0    # FILL_IN: compiled input height (e.g. 800)
    INPUT_W       = 0    # FILL_IN: compiled input width  (e.g. 800)
    CONF_THRESH   = 0.7  # DETR scores are bimodal; 0.7 is a safe start
    OVERLAY_ALPHA = 160
    LABELS_URL    = "COCO_91"  # 1-based COCO category IDs; base class handles automatically
    LABELS_TMP    = ""

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Fixed resize to (INPUT_W, INPUT_H) — NO letterbox.
        ImageNet norm: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
        Store: self._orig_w, self._orig_h.
        Return {"pixel_values": NCHW float32 (1,3,H,W),
                "pixel_mask":   ones float32 (1,H,W)}.
        pixel_mask dtype is float32 in the QPC (int64 promoted by -convert-to-fp16).
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Identify class logits (1,Q,C+1) and box coords (1,Q,4) by tensor shape.
        Softmax over last dim of logits. Exclude no-object (last column, index C).
        Filter by CONF_THRESH. Do NOT apply NMS.
        Convert normalized cxcywh → pixel x1y1x2y2: x1=(cx-w/2)*orig_w, clip to [0,orig_w].
        Use self._get_labels(); skip detections where label == 'N/A'.
        Return {'prediction':{'boxes':[[x1,y1,x2,y2],...],'labels':[str,...],'scores':[float,...]},'latency_ms':float}.
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        _draw_detections(image_path, prediction, output_path, self.OVERLAY_ALPHA, self.get_font)
# └─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    ModelPredictor().main()
