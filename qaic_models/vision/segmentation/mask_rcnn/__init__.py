####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from .model import LatencyStats, MaskRCNN, MaskRCNNOnnxCPU, MaskRCNNQaic, PredictResult

Model = MaskRCNN

__all__ = [
    "LatencyStats",
    "MaskRCNN",
    "MaskRCNNOnnxCPU",
    "MaskRCNNQaic",
    "PredictResult",
    "Model",
]
