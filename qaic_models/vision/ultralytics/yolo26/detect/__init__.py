####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from .model import YOLO26Detect, YOLO26DetectOnnxCPU, YOLO26DetectQaic

Model = YOLO26Detect

__all__ = [
    "YOLO26Detect",
    "YOLO26DetectOnnxCPU",
    "YOLO26DetectQaic",
    "Model",
]
