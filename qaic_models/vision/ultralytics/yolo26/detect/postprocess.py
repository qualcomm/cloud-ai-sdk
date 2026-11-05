####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qaic_models.vision.ultralytics.common.output_contract import (
    load_raw_record_contract,
)
from qaic_models.vision.ultralytics.common.postprocess import (
    extract_named_output,
    label_for_class,
)
from qaic_models.vision.ultralytics.common.preprocess import LetterboxInfo

RAW_CONTRACT = load_raw_record_contract(Path(__file__).with_name("model.yaml"), "output0")
OUTPUT_NAME = RAW_CONTRACT.output_name
OUTPUT_SLOTS = 300
OUTPUT_CHANNELS = RAW_CONTRACT.field_width
CONF_THRESH = 0.25


@dataclass(frozen=True)
class PostprocessConfig:
    conf_thresh: float = CONF_THRESH


DEFAULT_CONFIG = PostprocessConfig()


def _validate_output(output0) -> np.ndarray:
    arr = np.asarray(output0)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"expected {OUTPUT_NAME} shape [1, {OUTPUT_SLOTS}, {OUTPUT_CHANNELS}], got {arr.shape}")
    try:
        records, _ranks = RAW_CONTRACT.records_and_ranks(arr)
    except ValueError as exc:
        raise ValueError(f"expected {OUTPUT_NAME} shape [1, {OUTPUT_SLOTS}, {OUTPUT_CHANNELS}], got {arr.shape}") from exc
    if records.shape != (OUTPUT_SLOTS, OUTPUT_CHANNELS):
        raise ValueError(f"expected {OUTPUT_NAME} shape [1, {OUTPUT_SLOTS}, {OUTPUT_CHANNELS}], got {arr.shape}")
    return records


def postprocess_detections(
    outputs,
    letterbox: LetterboxInfo,
    labels: Sequence[str],
    cfg: PostprocessConfig = DEFAULT_CONFIG,
) -> dict:
    detections = _validate_output(extract_named_output(outputs, OUTPUT_NAME))
    confidence_index = RAW_CONTRACT.field_index("confidence")
    class_index = RAW_CONTRACT.field_index("class_id")
    coordinate_indices = [RAW_CONTRACT.field_index(name) for name in ("x1", "y1", "x2", "y2")]
    conf = detections[:, confidence_index]
    keep = conf > cfg.conf_thresh
    if not np.any(keep):
        return {"boxes": [], "labels": [], "scores": []}

    kept = detections[keep]
    boxes = kept[:, coordinate_indices].astype(np.float32)
    scores = kept[:, confidence_index].astype(np.float32)
    classes = kept[:, class_index].astype(np.int64)

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - letterbox.pad_x) / letterbox.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - letterbox.pad_y) / letterbox.scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, letterbox.orig_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, letterbox.orig_height)

    return {
        "boxes": boxes.tolist(),
        "labels": [label_for_class(int(class_id), labels) for class_id in classes.tolist()],
        "scores": scores.tolist(),
    }
