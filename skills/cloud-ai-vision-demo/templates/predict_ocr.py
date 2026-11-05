#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> OCR/ALPR inference on Cloud AI 100.

TEMPLATE: cp this file to /workspace/<model>/predict.py, fill every FILL_IN marker,
then smoke-test before generating the MODEL_CARD.

PDF inputs: predict_base handles PDF rasterization automatically. When --image
receives a .pdf path, each page is rasterized and passed through predict() in
sequence. The result is a list of {page, prediction, latency_ms} dicts, one per
page (capped at 20 pages). No changes to preprocess/postprocess/visualize needed.

Do NOT rewrite visualize() — it is correct as-is."""

import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict_base import QAICPredictor

# ── START FILL-IN SECTION ─────────────────────────────────────────────────────
class ModelPredictor(QAICPredictor):
    INPUT_H    = 0    # FILL_IN: compiled input height (e.g. 24 for LPRNet)
    INPUT_W    = 0    # FILL_IN: compiled input width  (e.g. 94 for LPRNet)
    BLANK_IDX  = 0    # FILL_IN: CTC blank token index (usually 0 or last index)
    OVERLAY_ALPHA = 200

    # FILL_IN: URL to a one-line file containing the character set as a single string,
    # OR a one-character-per-line file. Each character is one class index.
    # Example single-line: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    # self._get_labels() returns a list of single-character strings; join to get the charset.
    LABELS_URL = "FILL_IN"
    LABELS_TMP = "/tmp/FILL_IN_charset.txt"   # FILL_IN: use a unique /tmp name per model

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Open image with PIL. Convert to grayscale (or RGB if model expects 3 channels).
        Resize to (INPUT_W, INPUT_H). Normalize to [0,1] float32.
        Return {"input_name": NCHW array (1, channels, INPUT_H, INPUT_W)}.
        Use self._session.model_input_names[0] for the key.
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Unpack first output — shape (T, 1, C) or (1, T, C) where T=time steps.
        Apply softmax over C. Greedy CTC decode: argmax over C at each T step,
        collapse consecutive repeats, strip BLANK_IDX.
        Confidence = mean of max-softmax at non-blank steps (store as self._confidence).
        Charset from self._get_labels() — join single chars: ''.join(self._get_labels()).
        Return {'prediction': plate_text_str, 'confidence': float, 'latency_ms': float}.
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        img     = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(img)
        ov_draw = ImageDraw.Draw(overlay)
        short_side = min(img.width, img.height)
        font_size = max(28, short_side // 24)
        pad       = max(8,  short_side // 60)
        confidence = getattr(self, "_confidence", 0.0)
        label = f"{prediction}  ({confidence*100:.1f}%)"
        font = self.get_font(font_size, text=label)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = font_size * len(label) // 2, font_size
        w, h = img.size
        ry0 = h - th - 2 * pad
        ov_draw.rectangle([0, ry0, tw + 2 * pad, h], fill=(0, 0, 0, self.OVERLAY_ALPHA))
        img = Image.alpha_composite(img, overlay)
        ImageDraw.Draw(img).text((pad, ry0 + pad), label, fill=(0, 255, 80), font=font)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        img.convert("RGB").save(output_path, "JPEG", quality=92)


if __name__ == "__main__":
    ModelPredictor().main()
