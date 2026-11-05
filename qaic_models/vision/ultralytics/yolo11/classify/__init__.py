####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from .model import YOLO11Classify, YOLO11ClassifyOnnxCPU, YOLO11ClassifyQaic

Model = YOLO11Classify

__all__ = [
    "YOLO11Classify",
    "YOLO11ClassifyOnnxCPU",
    "YOLO11ClassifyQaic",
    "Model",
]
