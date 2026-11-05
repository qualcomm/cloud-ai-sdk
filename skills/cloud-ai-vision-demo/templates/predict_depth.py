#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> monocular depth estimation inference on Cloud AI 100.

TEMPLATE: cp this file to /workspace/<model>/predict.py, fill every FILL_IN marker,
then smoke-test before generating the MODEL_CARD.
Do NOT rewrite visualize() — it is correct as-is."""

import os, sys
import base64, io
import numpy as np
from PIL import Image
from predict_base import QAICPredictor

# ── Depth colormap (MAGMA-style, no matplotlib dependency) ───────────────────
_MAGMA_STOPS = np.array([
    [0,   0,   4],    # 0.00 — black
    [44,  15,  74],   # 0.25 — dark purple
    [150, 37,  79],   # 0.50 — red-purple
    [250, 120, 20],   # 0.75 — orange
    [252, 253, 191],  # 1.00 — pale yellow
], dtype=np.float32)


def _apply_magma(depth_norm: np.ndarray) -> np.ndarray:
    """Map [0,1] float array to RGB uint8 using a MAGMA-like 5-stop gradient."""
    n = len(_MAGMA_STOPS) - 1
    idx = depth_norm * n
    lo  = np.clip(idx.astype(np.int32), 0, n - 1)
    hi  = np.clip(lo + 1, 0, n)
    frac = (idx - lo)[..., np.newaxis]
    rgb = (_MAGMA_STOPS[lo] * (1 - frac) + _MAGMA_STOPS[hi] * frac)
    return rgb.clip(0, 255).astype(np.uint8)


# ── START FILL-IN SECTION ─────────────────────────────────────────────────────
class ModelPredictor(QAICPredictor):
    INPUT_H = 0   # FILL_IN: compiled input height (e.g. 384 for MiDaS-small)
    INPUT_W = 0   # FILL_IN: compiled input width  (e.g. 384 for MiDaS-small)

    # Depth models produce relative depth — no label file needed.
    LABELS_URL = ""
    LABELS_TMP = ""

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Resize to (INPUT_W, INPUT_H). Store self._orig_size = (orig_W, orig_H) for upscaling in visualize.
        Normalization: match the export script EXACTLY — replicate the code that builds the
            tensor passed to torch.onnx.export(). Do NOT default to ImageNet mean/std: many
            depth exports bake normalization into the ONNX graph (the wrapper registers
            image_mean/image_std as buffers and applies them before tracing), so the QPC
            expects /255-only pixels and applying mean/std here double-normalizes into
            degenerate output. Confirm against input.raw's range: [0,1] = divide by 255 only;
            min < 0 (~[-2.5,2.5]) = mean/std applied (0.485,0.456,0.406 / 0.229,0.224,0.225).
        Return {"input_name": NCHW float32 (1,3,INPUT_H,INPUT_W)}.
        Use self._session.model_input_names[0] for the key.
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Unpack first output — shape (1,1,H,W) or (1,H,W) or (H,W). Squeeze to 2-D (H,W).
        Store self._depth_map = depth_2d (raw float values — used by visualize).
        Compute summary stats:
            min_depth = float(depth_2d.min()), max_depth = float(depth_2d.max()),
            mean_depth = float(depth_2d.mean()).
        Values are relative (unitless) unless the model produces metric depth.
        Normalize to [0,1], invert (near=bright), encode as 8-bit grayscale PNG, base64:
            depth_norm = (depth_2d - min) / (max - min)  # or zeros if flat
            depth_u8 = (255 * (1.0 - depth_norm)).clip(0,255).astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(depth_u8, mode='L').save(buf, 'PNG')
            depth_png_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        Return {'prediction': {'min_depth': float, 'max_depth': float,
                               'mean_depth': float, 'depth_unit': 'relative',
                               'depth_png_b64': str}, 'latency_ms': float}.
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        depth = self._depth_map                         # (H, W) float
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin < 1e-6:
            norm = np.zeros_like(depth)
        else:
            norm = (depth - dmin) / (dmax - dmin)       # [0, 1]

        # Invert so near=bright (more intuitive for monocular depth)
        rgb = _apply_magma(1.0 - norm)                  # (H, W, 3)

        # Resize coloured depth map back to original image dimensions
        orig_w, orig_h = self._orig_size
        depth_img = Image.fromarray(rgb).resize((orig_w, orig_h), Image.BILINEAR)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        depth_img.save(output_path, "JPEG", quality=92)


if __name__ == "__main__":
    ModelPredictor().main()
