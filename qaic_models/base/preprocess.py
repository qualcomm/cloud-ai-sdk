####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageSize:
    """Image size in the recipe-level width/height convention."""

    width: int
    height: int

    @property
    def wh(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def hw(self) -> Tuple[int, int]:
        return (self.height, self.width)


ImageSizeInput = ImageSize | Tuple[int, int]


def image_size_wh(size: ImageSizeInput) -> Tuple[int, int]:
    """Return (width, height) for either ImageSize or a legacy WH tuple."""
    return size.wh if isinstance(size, ImageSize) else size


def image_size_hw(size: ImageSizeInput) -> Tuple[int, int]:
    """Return (height, width) for either ImageSize or a legacy WH tuple."""
    if isinstance(size, ImageSize):
        return size.hw
    width, height = size
    return (height, width)


def load_image_nchw_rgb(
    image_path: str,
    input_size: ImageSizeInput = ImageSize(width=640, height=640),
    batch: int | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Load an RGB image, resize it, and return contiguous CHW/NCHW float32."""
    img = Image.open(image_path).convert("RGB")
    new_w, new_h = image_size_wh(input_size)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    arr = np.array(img_resized).astype(np.float32)
    if normalize:
        arr /= 255.0
    arr = np.transpose(arr, (2, 0, 1))
    if batch:
        arr = np.expand_dims(arr, 0)

    return np.ascontiguousarray(arr)
