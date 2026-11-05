####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from .model import YOLO26Classify, YOLO26ClassifyOnnxCPU, YOLO26ClassifyQaic

Model = YOLO26Classify

__all__ = [
    "YOLO26Classify",
    "YOLO26ClassifyOnnxCPU",
    "YOLO26ClassifyQaic",
    "Model",
]
