####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qaic_models.base.predict_base import PredictorBase
from qaic_models.vision.ultralytics.common.output_contract import (
    load_raw_record_contract,
)
from qaic_models.vision.ultralytics.common.preprocess import LetterboxInfo

# Validated against the onboarded yolo11m export: scores in output0 are already
# activated (observed max ~0.94), so sigmoid is only applied as a safety branch
# for models/exports where scores come through as raw logits (max > 1.0).
CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.45
RAW_CONTRACT = load_raw_record_contract(Path(__file__).with_name("model.yaml"), "output0")
OUTPUT_NAME = RAW_CONTRACT.output_name
_CLASS_SCORE_SLICE = RAW_CONTRACT.field_slice("class_scores")
OUTPUT_CHANNELS = _CLASS_SCORE_SLICE.stop
NUM_CLASSES = _CLASS_SCORE_SLICE.stop - _CLASS_SCORE_SLICE.start


@dataclass(frozen=True)
class PostprocessConfig:
    conf_thresh: float = CONF_THRESH
    nms_iou_thresh: float = NMS_IOU_THRESH


DEFAULT_CONFIG = PostprocessConfig()


def _extract_output0(outputs) -> np.ndarray:
    """Normalize raw predictor output to the validated output0 tensor.

    QAICPredictor._forward() returns a dict keyed by tensor name; CPUPredictor's
    default _forward() returns the ONNX Runtime list output. Rejecting missing
    names and extra list outputs keeps training/debug outputs from being silently
    decoded as detections.
    """
    if isinstance(outputs, dict):
        if OUTPUT_NAME not in outputs:
            raise ValueError(f"missing expected QAIC output {OUTPUT_NAME!r}; got {sorted(outputs)}")
        return outputs[OUTPUT_NAME]
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise ValueError(f"expected exactly one ONNX output ({OUTPUT_NAME!r}); got {len(outputs)}")
        return outputs[0]
    return outputs


def decode_yolo11_detections(
    output0: np.ndarray,
    letterbox: LetterboxInfo,
    cfg: PostprocessConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw YOLO11 detect output into original-image-space detections.

    `output0` is `[1, 84, 8400]` (or the transposed `[1, 8400, 84]`): 4 box
    channels (cx, cy, w, h) followed by 80 COCO class scores. Returns NMS-filtered
    (boxes_xyxy, scores, classes) in original image coordinates.
    """
    arr = np.asarray(output0)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected output0 shape [1, {OUTPUT_CHANNELS}, N] or [1, N, {OUTPUT_CHANNELS}], got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected output0 shape [1, {OUTPUT_CHANNELS}, N] or [1, N, {OUTPUT_CHANNELS}], got {arr.shape}")
    if arr.shape[0] == OUTPUT_CHANNELS:
        canonical = arr[np.newaxis, :, :]
    elif arr.shape[-1] == OUTPUT_CHANNELS:
        arr = arr.T
        canonical = arr[np.newaxis, :, :]
    else:
        raise ValueError(f"expected output0 shape [1, {OUTPUT_CHANNELS}, N] or [1, N, {OUTPUT_CHANNELS}], got {output0.shape}")

    records, _ranks = RAW_CONTRACT.records_and_ranks(canonical)
    boxes_cxcywh = records[:, [RAW_CONTRACT.field_index(name) for name in ("cx", "cy", "width", "height")]].T
    class_scores = records[:, _CLASS_SCORE_SLICE].T
    if class_scores.shape[0] != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} class-score channels, got {class_scores.shape[0]}")

    if class_scores.size and class_scores.max() > 1.0:
        class_scores = 1.0 / (1.0 + np.exp(-class_scores))

    classes = np.argmax(class_scores, axis=0)
    scores = class_scores[classes, np.arange(class_scores.shape[1])]

    keep_conf = scores >= cfg.conf_thresh
    if not np.any(keep_conf):
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes_cxcywh = boxes_cxcywh[:, keep_conf]
    scores = scores[keep_conf]
    classes = classes[keep_conf]

    cx, cy, w, h = boxes_cxcywh
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    # Undo letterbox: forward pass was x' = x * scale + pad, so x = (x' - pad) / scale.
    x1 = (x1 - letterbox.pad_x) / letterbox.scale
    y1 = (y1 - letterbox.pad_y) / letterbox.scale
    x2 = (x2 - letterbox.pad_x) / letterbox.scale
    y2 = (y2 - letterbox.pad_y) / letterbox.scale

    x1 = np.clip(x1, 0, letterbox.orig_width)
    x2 = np.clip(x2, 0, letterbox.orig_width)
    y1 = np.clip(y1, 0, letterbox.orig_height)
    y2 = np.clip(y2, 0, letterbox.orig_height)

    keep_idx = PredictorBase._nms(x1, y1, x2, y2, scores, cfg.nms_iou_thresh, classes=classes)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)[keep_idx].astype(np.float32)
    return boxes_xyxy, scores[keep_idx].astype(np.float32), classes[keep_idx].astype(np.int64)


def postprocess_detections(
    outputs,
    letterbox: LetterboxInfo,
    labels: Sequence[str],
    cfg: PostprocessConfig = DEFAULT_CONFIG,
) -> dict:
    """Build the canonical detection prediction dict from raw predictor output."""
    output0 = _extract_output0(outputs)
    boxes_xyxy, scores, classes = decode_yolo11_detections(output0, letterbox, cfg)
    return {
        "boxes": boxes_xyxy.tolist(),
        "labels": [labels[c] if labels and c < len(labels) else str(c) for c in classes.tolist()],
        "scores": scores.tolist(),
    }
