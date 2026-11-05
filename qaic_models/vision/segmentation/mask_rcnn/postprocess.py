####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .box_decode_nms import DecodeNMSConfig, decode_and_nms
from .preprocess import MASK_RCNN_IMAGE_SIZE


@dataclass(frozen=True)
class PostprocessConfig:
    num_classes: int = 80
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    topk: int = 100
    clip_boxes: bool = True
    mask_thresh: float = 0.5  # threshold applied to selected mask logits


@dataclass(frozen=True)
class PostprocessResult:
    boxes_xyxy: np.ndarray         # (N,4) float32
    scores: np.ndarray             # (N,) float32
    classes: np.ndarray            # (N,) int64
    masks: Optional[np.ndarray]    # (N,H,W) bool OR None (full-image masks at image_hw)


def _clip_xyxy_inplace(boxes: np.ndarray, h: int, w: int) -> None:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)


def postprocess_detections(
    proposals_xyxy: np.ndarray,
    cls_logits: np.ndarray,
    bbox_deltas: np.ndarray,
    mask_logits: Optional[np.ndarray],
    image_hw: Tuple[int, int] = MASK_RCNN_IMAGE_SIZE.hw,
    cfg: PostprocessConfig = PostprocessConfig(),
) -> PostprocessResult:
    """
        CPU postprocess for fixed-shape outputs from model_cpu_heads.onnx.

    Inputs are per-proposal outputs for a fixed K proposals (e.g. K=200):
      - proposals_xyxy: (K,4)
      - cls_logits: (K, num_classes+1)
      - bbox_deltas: (K, 4*num_classes) or (K, 4*(num_classes+1))
      - mask_logits: (K, num_classes, 28, 28) (typical)

    This helper performs:
      1) softmax + score threshold
      2) per-class bbox decode
      3) per-class NMS
      4) global top-k

    Mask handling:
      - selects the mask logits for the predicted class for each kept detection
      - resizes/pastes each ROI mask into full-image resolution according to the
        decoded box, producing (N,H,W) boolean masks in image coordinates
      - thresholds logits at mask_thresh
    """

    det = decode_and_nms(
        proposals_xyxy=np.asarray(proposals_xyxy),
        cls_logits=np.asarray(cls_logits),
        bbox_deltas=np.asarray(bbox_deltas),
        image_hw=image_hw,
        cfg=DecodeNMSConfig(
            num_classes=cfg.num_classes,
            score_thresh=cfg.score_thresh,
            nms_thresh=cfg.nms_thresh,
            topk_per_image=cfg.topk,
            clip_boxes=cfg.clip_boxes,
        ),
    )

    masks_out: Optional[np.ndarray] = None
    if mask_logits is not None and det.boxes_xyxy.shape[0] > 0:
        ml = np.asarray(mask_logits)  # (K,C,Hm,Wm)
        if ml.ndim != 4:
            raise ValueError(f"mask_logits expected (K,C,Hm,Wm). Got {ml.shape}")

        # select class-specific mask for each kept detection (N, Hm, Wm)
        sel = ml[det.keep_indices, det.classes]
        N, Hm, Wm = sel.shape

        # Paste ROI masks into full-image masks
        H_img, W_img = image_hw
        masks_full = np.zeros((N, H_img, W_img), dtype=bool)
        from PIL import Image as PILImage

        for i in range(N):
            x1, y1, x2, y2 = det.boxes_xyxy[i]
            # clamp box to image bounds
            x1i = int(max(0, min(W_img - 1, x1)))
            x2i = int(max(0, min(W_img, x2)))
            y1i = int(max(0, min(H_img - 1, y1)))
            y2i = int(max(0, min(H_img, y2)))
            if x2i <= x1i or y2i <= y1i:
                continue

            box_h = y2i - y1i
            box_w = x2i - x1i
            roi_mask = sel[i]

            if (box_w, box_h) != (Wm, Hm):
                # Resize ROI mask to match the box size
                roi_img = PILImage.fromarray(roi_mask.astype(np.float32))
                roi_resized = roi_img.resize((box_w, box_h), resample=PILImage.BILINEAR)
                roi_arr = np.array(roi_resized)
            else:
                roi_arr = roi_mask

            m_small = roi_arr > cfg.mask_thresh
            masks_full[i, y1i:y2i, x1i:x2i] = m_small

        masks_out = masks_full

    return PostprocessResult(
        boxes_xyxy=det.boxes_xyxy,
        scores=det.scores,
        classes=det.classes,
        masks=masks_out,
    )
