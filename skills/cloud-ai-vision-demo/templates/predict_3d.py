#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> 3D geometry estimation (point cloud) on Cloud AI 100.

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

_MAX_POINTS = 262144  # subsample cap — increase if the model's spatial resolution exceeds 512×512


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
    INPUT_H = 0   # FILL_IN: compiled input height (e.g. 512 for MoGe-2)
    INPUT_W = 0   # FILL_IN: compiled input width  (e.g. 512 for MoGe-2)

    # Point-cloud models produce geometry — no label file needed.
    LABELS_URL = ""
    LABELS_TMP = ""

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Load the image from image_path.
        Resize to (INPUT_W, INPUT_H) with Image.BILINEAR.
        Store:
            self._orig_size = (orig_W, orig_H)   — for visualize() resize
            self._orig_img  = resized RGB uint8 (INPUT_H, INPUT_W, 3) — for RGB sampling
        Normalization: find the code in the export script that constructs the example
            input tensor passed to torch.onnx.export() — that is the complete preprocessing
            pipeline the compiled QPC expects; match it exactly in this method.
            If input.raw exists in the workspace, cross-check by inspecting its value
            range (e.g. [0,1] means /255 only; [-2.5,+2.5] means mean/std was applied).
            If no export script exists, consult the model source repo or model card.
            Do not infer normalization from the model name or README alone.
        Return NCHW float32 (1, 3, INPUT_H, INPUT_W).
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Decode the model output tensors to a subsampled point cloud.

        Output tensor structure — inspect session.model_output_names and shapes:
            XYZ tensor: float32 array of 3D coordinates.
                        Shape is typically (1, H, W, 3) for spatial grid models
                        or (1, N, 3) for unstructured point-set models.
            Mask tensor (optional): float32 or bool validity flags, same spatial
                        extent as XYZ. If absent, treat all points as valid:
                            valid_flat = np.ones(N, dtype=bool)

        Coordinate system — verify from the model source repo or export script:
            Output raw model-space coordinates; do not apply rendering transforms.
            Add a comment on the xyz tensor line naming the convention
            (e.g. # OpenCV camera space: Y-down, Z-forward) so downstream steps
            can scan it. Most monocular geometry models output OpenCV camera space
            (Y-down, Z-forward).

        Steps:
        1.  Squeeze XYZ tensor to (..., 3) and mask to (...).
        2.  Store self._depth_map = xyz[..., 2]   (raw Z, NOT multiplied by mask)
                  self._valid_mask = (mask > 0.5)   (bool, same spatial shape)
            Do NOT multiply Z by the mask — that collapses dmin to 0, inverting the colormap.
        3.  Choose output path based on model type:

            PATH A — spatial grid model (H×W points map 1-to-1 to input pixels):
                Send the full H×W grid with mask_b64 + grid_h + grid_w.
                This enables mesh rendering with depth-discontinuity culling,
                producing clean edges without scattered boundary pixels.
                Most monocular geometry models (MoGe, DepthPro, etc.) are grid models.

                all_xyz  = np.ascontiguousarray(xyz.reshape(-1, 3), dtype=np.float32)
                all_rgb  = np.ascontiguousarray(self._orig_img.reshape(-1, 3), dtype=np.uint8)
                all_mask = (mask.reshape(-1) > 0.5).astype(np.uint8)
                return {"prediction": {"points_xyz_b64": b64(all_xyz),
                                       "points_rgb_b64": b64(all_rgb),
                                       "mask_b64":       b64(all_mask),
                                       "grid_h": H, "grid_w": W},
                        "latency_ms": latency_ms}

            PATH B — unstructured point set (N points, N ≠ H×W, no pixel correspondence):
                Subsample to _MAX_POINTS valid points.
                Send n_points + point_map_h + point_map_w (no mask_b64).

                idx = np.where(valid_flat)[0]
                if len(idx) > _MAX_POINTS:
                    idx = np.sort(np.random.choice(idx, _MAX_POINTS, replace=False))
                points_xyz = np.ascontiguousarray(xyz_flat[idx], dtype=np.float32)
                points_rgb = np.full((len(idx), 3), 128, dtype=np.uint8)  # constant if no pixel map
                return {"prediction": {"points_xyz_b64": b64(points_xyz),
                                       "points_rgb_b64": b64(points_rgb),
                                       "n_points":    int(len(idx)),
                                       "point_map_h": H,
                                       "point_map_w": W},
                        "latency_ms": latency_ms}

            where b64 = lambda a: base64.b64encode(a.tobytes()).decode('ascii')
        """
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        depth = self._depth_map                          # (H, W) raw Z, NOT masked
        valid = self._valid_mask                         # (H, W) bool
        valid_depths = depth[valid]
        if valid.any() and valid_depths.max() - valid_depths.min() > 1e-6:
            dmin, dmax = float(valid_depths.min()), float(valid_depths.max())
            norm = np.clip((depth - dmin) / (dmax - dmin), 0.0, 1.0)
        else:
            norm = np.zeros_like(depth)

        rgb = _apply_magma(1.0 - norm)                   # (H, W, 3) — near=bright
        rgb[~valid] = 0                                  # invalid pixels → black

        orig_w, orig_h = self._orig_size
        depth_img = Image.fromarray(rgb).resize((orig_w, orig_h), Image.BILINEAR)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        depth_img.save(output_path, "JPEG", quality=92)


if __name__ == "__main__":
    ModelPredictor().main()
