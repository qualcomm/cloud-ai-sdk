####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from .model import LPRNet, LPRNetOnnxCPU, LPRNetQaic

Model = LPRNet

__all__ = [
    "LPRNet",
    "LPRNetOnnxCPU",
    "LPRNetQaic",
    "Model",
]
