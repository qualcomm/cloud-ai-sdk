####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from .model import YOLO11, YOLO11OnnxCPU, YOLO11Qaic

Model = YOLO11

__all__ = [
    "YOLO11",
    "YOLO11OnnxCPU",
    "YOLO11Qaic",
    "Model",
]
