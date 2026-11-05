#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> face detection inference on Cloud AI 100.

TEMPLATE: cp this file to /workspace/<model>/predict.py, fill every FILL_IN marker,
then smoke-test before generating the MODEL_CARD.
Do NOT rewrite visualize() — it is correct as-is.

Covers face detectors that output bounding boxes with optional 5-point landmarks
(eyes, nose, mouth corners): SCRFD, RetinaFace, YOLO-Face, etc.
For pure classification on already-cropped faces (age/gender/emotion), use
predict_classification.py instead."""

import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict_base import QAICPredictor, hex_to_rgb

# ── Face detection helpers ────────────────────────────────────────────────────

# 5-point landmark names (standard for most face detectors)
_LM_NAMES = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]

# Landmark dot colours
_LM_COLORS = ["#33CCFF", "#33CCFF", "#FFFF33", "#FF6633", "#FF6633"]

FACE_COLOR = "#FF3333"



# ── START FILL-IN SECTION ─────────────────────────────────────────────────────
class ModelPredictor(QAICPredictor):
    INPUT_H        = 0     # FILL_IN: compiled input height (e.g. 640)
    INPUT_W        = 0     # FILL_IN: compiled input width  (e.g. 640)
    CONF_THRESH    = 0.5
    NMS_IOU_THRESH = 0.4   # face detectors benefit from tighter suppression
    OVERLAY_ALPHA  = 200

    # Face detectors are single-class — no label file needed.
    LABELS_URL = ""
    LABELS_TMP = ""

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Letterbox-resize to (INPUT_W, INPUT_H) on a (114,114,114) grey canvas
        (or (0,0,0) — check model docs). Normalize to [0,1] float32.
        Store: self._orig_w, self._orig_h, self._letterbox_scale, self._letterbox_offset.
        Return {"input_name": NCHW float32 (1,3,INPUT_H,INPUT_W)}.
        Use self._session.model_input_names[0] for the key.
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Output layout varies by model family:
            SCRFD:      3 tensors — scores (1,N,1), boxes (1,N,4), landmarks (1,N,10).
            RetinaFace: similar 3-tensor split, or fused (1,N,15) = box4+score1+lm10.
            YOLO-Face:  (1, 5+10, N) = box4+score1+lm10 (x,y per landmark, stride ×2).
        Steps:
        1. Extract scores, boxes (xywh or xyxy — check model), landmarks (optional).
        2. Activate scores if logits (sigmoid). Filter by CONF_THRESH.
        3. Convert boxes to x1y1x2y2. Undo letterbox on boxes AND landmarks:
               x_orig = (x_input - off_x) / scale. Clip to orig dims.
        4. Run _nms().
        5. Build faces list — landmarks are optional; omit key if not available:
               {'box': [x1,y1,x2,y2], 'score': float,
                'landmarks': [[x,y], ...]}  # 5 points: leye,reye,nose,lmouth,rmouth
        Return {'prediction': {'faces': [...], 'num_faces': int}, 'latency_ms': float}.
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        img     = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(img)
        ov_draw = ImageDraw.Draw(overlay)
        short_side = min(img.width, img.height)
        font_size  = max(20, short_side // 35)
        box_w      = max(3,  short_side // 180)
        lm_r       = max(3,  short_side // 150)   # landmark dot radius
        pad        = max(4,  font_size // 4)
        font = self.get_font(font_size)
        fr, fg, fb = hex_to_rgb(FACE_COLOR)

        labels_to_draw = []
        for face in prediction.get("faces", []):
            box   = face.get("box", [])
            score = face.get("score", 0.0)
            lms   = face.get("landmarks", [])   # [[x,y], ...] up to 5 points

            if len(box) == 4:
                x1, y1, x2, y2 = box
                ov_draw.rectangle([x1, y1, x2, y2],
                                  outline=(fr, fg, fb, self.OVERLAY_ALPHA), width=box_w)
                text = f"face: {score:.2f}"
                lx, ly = x1 + pad, y1 + box_w // 2
                try:
                    bbox = draw.textbbox((lx, ly), text, font=font)
                    ov_draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                                      fill=(fr, fg, fb, self.OVERLAY_ALPHA))
                except Exception:
                    pass
                labels_to_draw.append((lx, ly, text))

            for lm_idx, lm in enumerate(lms[:5]):
                if len(lm) < 2:
                    continue
                x, y = lm[0], lm[1]
                r, g, b = hex_to_rgb(_LM_COLORS[lm_idx % len(_LM_COLORS)])
                ov_draw.ellipse([x - lm_r, y - lm_r, x + lm_r, y + lm_r],
                                fill=(r, g, b, 230))

        img = Image.alpha_composite(img, overlay)
        final_draw = ImageDraw.Draw(img)
        for lx, ly, text in labels_to_draw:
            final_draw.text((lx, ly), text, fill="white", font=font)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.convert("RGB").save(output_path, "JPEG", quality=92)


if __name__ == "__main__":
    ModelPredictor().main()
