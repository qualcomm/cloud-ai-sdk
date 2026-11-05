#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> pose estimation inference on Cloud AI 100.

TEMPLATE: cp this file to /workspace/<model>/predict.py, fill every FILL_IN marker,
then smoke-test before generating the MODEL_CARD.
Do NOT rewrite visualize() — it is correct as-is."""

import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict_base import QAICPredictor, hex_to_rgb

# ── Pose helpers ──────────────────────────────────────────────────────────────

# COCO-17 keypoint names (index → name)
COCO17_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO-17 skeleton — pairs of keypoint indices to connect with a line
COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# Limb colours — one per skeleton edge (cycles if model has more connections)
LIMB_COLORS = [
    "#FF3333", "#FF3333", "#33FF33", "#33FF33",
    "#3399FF",
    "#FF9900", "#FF9900", "#FFFF33", "#FFFF33",
    "#CC33FF", "#CC33FF", "#3399FF",
    "#FF66CC", "#FF66CC", "#00CCCC", "#00CCCC",
]

PERSON_COLOR = "#3399FF"



# ── START FILL-IN SECTION ─────────────────────────────────────────────────────
class ModelPredictor(QAICPredictor):
    INPUT_H        = 0     # FILL_IN: compiled input height (e.g. 640)
    INPUT_W        = 0     # FILL_IN: compiled input width  (e.g. 640)
    CONF_THRESH    = 0.25  # person detection confidence threshold
    NMS_IOU_THRESH = 0.45
    KP_THRESH      = 0.3   # keypoint visibility threshold (below → skip drawing)
    NUM_KEYPOINTS  = 17    # FILL_IN: number of keypoints per instance (COCO = 17)
    OVERLAY_ALPHA  = 200

    # Set to "" if the model is single-class (person only) — no download needed.
    # For multi-class pose: URL to one-label-per-line class file.
    LABELS_URL = ""
    LABELS_TMP = ""

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Letterbox-resize to (INPUT_W, INPUT_H) on a (114,114,114) grey canvas.
        Normalize to [0,1] float32. Return {"input_name": NCHW array (1,3,INPUT_H,INPUT_W)}.
        Use self._session.model_input_names[0] for the key.
        Store: self._orig_w, self._orig_h, self._letterbox_scale,
               self._letterbox_offset = (off_x, off_y).
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Decode model outputs to keypoints in original image space.

        For YOLO-Pose: output shape (1, 4+C+K*3, N). Split: boxes=raw[:4],
        scores=raw[4:4+C], keypoints=raw[4+C:].reshape(K,3,N). Activate scores
        if logits (sigmoid). Filter by CONF_THRESH. Run _nms(). For heatmap
        models (HRNet/ViTPose): argmax per heatmap channel + sub-pixel shift.

        Undo letterbox on all coordinates:
            x_orig = (x_input - off_x) / scale

        Return HuggingFace-aligned schema — parallel arrays per pose:
            {
              'prediction': {
                'poses': [
                  {
                    'keypoints': [[x, y], ...],   # (K, 2) original image coords
                    'scores':    [float, ...],     # (K,)   per-keypoint confidence
                    'labels':    [int, ...],       # (K,)   keypoint indices 0..K-1
                    'bbox':      [x1, y1, x2, y2] # pascal-voc, original image coords
                  },
                  ...
                ],
                'num_poses': int
              },
              'latency_ms': float
            }
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        img     = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        short_side = min(img.width, img.height)
        kp_r    = max(4,  short_side // 120)
        limb_w  = max(3,  short_side // 180)
        pr, pg, pb = hex_to_rgb(PERSON_COLOR)

        for pose in prediction.get("poses", []):
            kps    = pose.get("keypoints", [])   # [[x, y], ...]
            scores = pose.get("scores", [])       # [score, ...]

            # skeleton limbs
            skeleton = COCO17_SKELETON if len(kps) == 17 else []
            for edge_idx, (i, j) in enumerate(skeleton):
                if i >= len(kps) or j >= len(kps):
                    continue
                ci = scores[i] if i < len(scores) else 0.0
                cj = scores[j] if j < len(scores) else 0.0
                if ci < self.KP_THRESH or cj < self.KP_THRESH:
                    continue
                xi, yi = kps[i]; xj, yj = kps[j]
                color = LIMB_COLORS[edge_idx % len(LIMB_COLORS)]
                r, g, b = hex_to_rgb(color)
                ov_draw.line([(xi, yi), (xj, yj)], fill=(r, g, b, 220), width=limb_w)

            # keypoint dots
            for kp, sc in zip(kps, scores):
                if sc < self.KP_THRESH:
                    continue
                x, y = kp
                ov_draw.ellipse([x - kp_r, y - kp_r, x + kp_r, y + kp_r],
                                fill=(255, 255, 255, 230),
                                outline=(pr, pg, pb, 255), width=max(1, kp_r // 3))

        img = Image.alpha_composite(img, overlay)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.convert("RGB").save(output_path, "JPEG", quality=92)


if __name__ == "__main__":
    ModelPredictor().main()
