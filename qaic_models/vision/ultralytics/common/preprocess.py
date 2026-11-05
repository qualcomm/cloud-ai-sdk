####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from qaic_models.base.preprocess import ImageSize

ULTRALYTICS_DETECT_IMAGE_SIZE = ImageSize(width=640, height=640)
ULTRALYTICS_CLASSIFY_IMAGE_SIZE = ImageSize(width=224, height=224)

_LETTERBOX_PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry needed to map letterboxed detections back to the original image."""

    orig_width: int
    orig_height: int
    scale: float
    pad_x: int
    pad_y: int


@dataclass(frozen=True)
class ClassifyPreprocessInfo:
    """Geometry recorded for Ultralytics classification preprocessing."""

    orig_width: int
    orig_height: int
    resized_width: int
    resized_height: int
    crop_x: int
    crop_y: int


def load_ultralytics_letterbox_image(
    image_path: str,
    size: ImageSize = ULTRALYTICS_DETECT_IMAGE_SIZE,
) -> tuple[np.ndarray, LetterboxInfo]:
    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size
    target_width, target_height = size.wh

    scale = min(target_width / orig_width, target_height / orig_height)
    new_width = round(orig_width * scale)
    new_height = round(orig_height * scale)
    resized = img.resize((new_width, new_height), Image.BILINEAR)

    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2

    canvas = Image.new("RGB", (target_width, target_height), (_LETTERBOX_PAD_VALUE,) * 3)
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)

    return np.ascontiguousarray(arr), LetterboxInfo(
        orig_width=orig_width,
        orig_height=orig_height,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )


def load_ultralytics_classify_image(
    image_path: str,
    size: ImageSize = ULTRALYTICS_CLASSIFY_IMAGE_SIZE,
) -> tuple[np.ndarray, ClassifyPreprocessInfo]:
    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size
    target_width, target_height = size.wh

    scale = max(target_width / orig_width, target_height / orig_height)
    resized_width = round(orig_width * scale)
    resized_height = round(orig_height * scale)
    resized = img.resize((resized_width, resized_height), Image.BILINEAR)

    crop_x = (resized_width - target_width) // 2
    crop_y = (resized_height - target_height) // 2
    cropped = resized.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))

    arr = np.array(cropped).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)

    return np.ascontiguousarray(arr), ClassifyPreprocessInfo(
        orig_width=orig_width,
        orig_height=orig_height,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_x=crop_x,
        crop_y=crop_y,
    )
