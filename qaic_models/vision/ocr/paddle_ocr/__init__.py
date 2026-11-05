####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from .model import (
    PaddleOCRLayout, PaddleOCRDetection, PaddleOCRRecognition,
    PaddleOCRLayoutOnnxCPU, PaddleOCRLayoutQaic,
    PaddleOCRDetOnnxCPU, PaddleOCRDetQaic,
    PaddleOCRRecOnnxCPU, PaddleOCRRecQaic,
)
from .pipeline import PaddleOCRPipeline

Pipeline = PaddleOCRPipeline

__all__ = [
    "PaddleOCRLayout",
    "PaddleOCRDetection",
    "PaddleOCRRecognition",
    "PaddleOCRPipeline",
    "Pipeline",
    "PaddleOCRLayoutOnnxCPU",
    "PaddleOCRLayoutQaic",
    "PaddleOCRDetOnnxCPU",
    "PaddleOCRDetQaic",
    "PaddleOCRRecOnnxCPU",
    "PaddleOCRRecQaic",
]
