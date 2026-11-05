####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import numpy as np

from qaic_models.base.preprocess import ImageSize, ImageSizeInput, image_size_wh, load_image_nchw_rgb


MASK_RCNN_IMAGE_SIZE = ImageSize(width=1216, height=800)


class MaskRCNNImage:
    """Utility for loading and preprocessing images for Mask R-CNN."""

    def load(image_path: str, size: ImageSizeInput = MASK_RCNN_IMAGE_SIZE, batch: int = 1) -> np.ndarray:
        return load_image_nchw_rgb(image_path, image_size_wh(size), batch, normalize=False)
