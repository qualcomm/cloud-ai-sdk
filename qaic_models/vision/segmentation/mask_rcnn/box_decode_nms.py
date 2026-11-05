####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .preprocess import MASK_RCNN_IMAGE_SIZE


@dataclass(frozen=True)
class DecodeNMSConfig:
    num_classes: int = 80
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    topk_per_image: int = 100
    clip_boxes: bool = True


@dataclass(frozen=True)
class DecodeNMSResult:
    boxes_xyxy: np.ndarray      # (N,4) float32
    scores: np.ndarray          # (N,) float32
    classes: np.ndarray         # (N,) int64 in [0,num_classes)
    keep_indices: np.ndarray    # (N,) int64 indices into proposals (0..K-1)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _clip_xyxy_inplace(boxes: np.ndarray, h: int, w: int) -> None:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)


def _apply_deltas_to_proposals(
    proposals_xyxy: np.ndarray,
    deltas: np.ndarray,
    weights: Tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> np.ndarray:
    """Detectron2 Box2BoxTransform apply_deltas for XYXY boxes.

    proposals_xyxy: (K,4)
    deltas: (K,4)
    returns: (K,4)
    """
    wx, wy, ww, wh = weights
    boxes = proposals_xyxy.astype(np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    ctr_x = x1 + 0.5 * w
    ctr_y = y1 + 0.5 * h

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh

    # Prevent sending too large values into exp()
    dw = np.clip(dw, a_min=None, a_max=np.log(1000.0 / 16.0))
    dh = np.clip(dh, a_min=None, a_max=np.log(1000.0 / 16.0))

    pred_ctr_x = dx * w + ctr_x
    pred_ctr_y = dy * h + ctr_y
    pred_w = np.exp(dw) * w
    pred_h = np.exp(dh) * h

    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    return np.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=1)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        suppressed = []
        for j in rest:
            if _iou_xyxy(boxes[i], boxes[int(j)]) > iou_thresh:
                suppressed.append(int(j))
        if suppressed:
            mask = np.ones(rest.shape[0], dtype=bool)
            sup_set = set(suppressed)
            for k, j in enumerate(rest):
                if int(j) in sup_set:
                    mask[k] = False
            order = rest[mask]
        else:
            order = rest
    return np.array(keep, dtype=np.int64)


def decode_and_nms(
    proposals_xyxy: np.ndarray,
    cls_logits: np.ndarray,
    bbox_deltas: np.ndarray,
    image_hw: Tuple[int, int] = MASK_RCNN_IMAGE_SIZE.hw,
    cfg: DecodeNMSConfig = DecodeNMSConfig(),
) -> DecodeNMSResult:
    """Decode class-specific box deltas and run per-class NMS.

    Assumes class-specific regression with bbox_deltas shaped (K, 4*num_classes).
    """
    h, w = image_hw
    K = int(proposals_xyxy.shape[0])

    probs = _softmax(cls_logits, axis=1).astype(np.float32)  # (K, C+1)
    probs_fg = probs[:, : cfg.num_classes]  # drop background

    # bbox_deltas is (K, 4*num_classes) for class-specific; some models may include background.
    deltas = bbox_deltas.reshape(K, -1, 4).astype(np.float32)
    if deltas.shape[1] == cfg.num_classes + 1:
        deltas = deltas[:, : cfg.num_classes, :]
    elif deltas.shape[1] != cfg.num_classes:
        raise ValueError(f"Unexpected bbox_deltas classes: {deltas.shape}")

    all_boxes = []
    all_scores = []
    all_classes = []
    all_keep_idx = []

    # per-class threshold + NMS
    for c in range(cfg.num_classes):
        sc = probs_fg[:, c]
        keep = sc >= cfg.score_thresh
        if not np.any(keep):
            continue
        idxs = np.nonzero(keep)[0].astype(np.int64)

        boxes_c = _apply_deltas_to_proposals(proposals_xyxy[idxs], deltas[idxs, c, :])
        scores_c = sc[idxs]

        if cfg.clip_boxes:
            _clip_xyxy_inplace(boxes_c, h=h, w=w)

        keep_local = _nms_xyxy(boxes_c, scores_c, cfg.nms_thresh)
        idxs_kept = idxs[keep_local]

        all_boxes.append(boxes_c[keep_local])
        all_scores.append(scores_c[keep_local])
        all_classes.append(np.full((keep_local.shape[0],), c, dtype=np.int64))
        all_keep_idx.append(idxs_kept)

    if not all_boxes:
        return DecodeNMSResult(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.int64),
            keep_indices=np.zeros((0,), dtype=np.int64),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    keep_indices = np.concatenate(all_keep_idx, axis=0)

    # global top-k
    if cfg.topk_per_image is not None and scores.shape[0] > cfg.topk_per_image:
        order = np.argsort(-scores)[: cfg.topk_per_image]
        boxes = boxes[order]
        scores = scores[order]
        classes = classes[order]
        keep_indices = keep_indices[order]

    return DecodeNMSResult(
        boxes_xyxy=boxes.astype(np.float32, copy=False),
        scores=scores.astype(np.float32, copy=False),
        classes=classes.astype(np.int64, copy=False),
        keep_indices=keep_indices.astype(np.int64, copy=False),
    )
