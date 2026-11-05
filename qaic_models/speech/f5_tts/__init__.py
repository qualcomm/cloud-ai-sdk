####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from .model import F5TTS, F5TTSConfig, F5TTSQaic, InferResult, LatencyStats

Model = F5TTS

__all__ = ["F5TTS", "F5TTSConfig", "F5TTSQaic", "InferResult", "LatencyStats", "Model"]
