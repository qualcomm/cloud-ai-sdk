####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

"""LPRNet input preparation.

Validated preprocessing (matches the LPRNet_Pytorch training pipeline):
  - load BGR image with OpenCV (PIL fallback converts RGB -> BGR)
  - resize to (94, 24) with cv2.resize (bilinear)
  - normalize: (pixel - 127.5) * 0.0078125
  - HWC -> CHW, add batch dim -> NCHW float32 [1, 3, 24, 94]

The base ``load_image_nchw_rgb`` helper is intentionally NOT used: it loads
RGB via PIL and normalizes by /255, which does not match the validated
BGR / 127.5 / 0.0078125 pipeline.
"""

from __future__ import annotations

import numpy as np

INPUT_NAME = "input"
INPUT_H = 24
INPUT_W = 94
INPUT_SHAPE = (1, 3, INPUT_H, INPUT_W)

MEAN = 127.5
SCALE = 0.0078125  # 1 / 128.0


def preprocess_plate_image(image_path: str) -> np.ndarray:
    """Load a license plate image and return the model input tensor.

    Returns a contiguous float32 array of shape (1, 3, 24, 94) keyed to the
    model input tensor name ``input``.
    """
    import cv2  # headless-safe; kept local so runtime installs stay light

    img = cv2.imread(image_path)
    if img is None:
        # Fallback: try PIL, convert RGB -> BGR to keep the cv2 convention.
        from PIL import Image

        pil_img = Image.open(image_path).convert("RGB")
        img = np.array(pil_img)[:, :, ::-1]

    img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img.astype(np.float32)
    img -= MEAN
    img *= SCALE
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.ascontiguousarray(img[np.newaxis])  # NCHW
    return img
