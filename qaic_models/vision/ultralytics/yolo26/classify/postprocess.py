####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

from typing import Sequence

from qaic_models.vision.ultralytics.common.postprocess import extract_named_output, postprocess_classification_probs

OUTPUT_NAME = "probs"
NUM_CLASSES = 1000


def postprocess_classification(outputs, labels: Sequence[str], topk: int = 5) -> dict:
    return postprocess_classification_probs(extract_named_output(outputs, OUTPUT_NAME), labels, topk=topk)
