####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from typing import List

import numpy as np
import json
import triton_python_backend_utils as pb_utils

from qaic_models.vision.segmentation.mask_rcnn.postprocess import PostprocessConfig, postprocess_detections
from qaic_models.vision.segmentation.mask_rcnn.preprocess import MASK_RCNN_IMAGE_SIZE


class TritonPythonModel:
    """Triton Python backend for Mask R-CNN postprocessing.

    Inputs (per config.pbtxt):
      - cls_logits:  [200, 81]
      - bbox_deltas: [200, 320]
      - mask_logits: [200, 80, 28, 28]

    Outputs:
      - cls_results: [N, 2]  (class_id, score) as float32
      - cls_counts:  [1]     (N) as int32
    """

    def __init__(self):
        # Will be initialized in initialize()
        self._cfg = None
        self._image_hw = MASK_RCNN_IMAGE_SIZE.hw
        self._topk = 20  # default, can be overridden via model config parameter

    def initialize(self, args: dict) -> None:
        # Read optional topk from model configuration parameters
        try:
            model_config = json.loads(args["model_config"])
            params = model_config.get("parameters", {})
            topk_param = params.get("topk", {})
            raw = topk_param.get("string_value", None)
            if raw is not None:
                self._topk = int(raw)
        except Exception:
            # Fall back to default _topk if parsing fails
            pass

        # Match the CPU reference postprocess configuration, but with configurable topk
        self._cfg = PostprocessConfig(
            num_classes=80,
            score_thresh=0.05,
            nms_thresh=0.5,
            topk=self._topk,
            clip_boxes=True,
            mask_thresh=0.5,
        )
        return

    def execute(self, requests):
        responses = []

        for request in requests:
            # Extract inputs from Triton
            cls_logits_in = pb_utils.get_input_tensor_by_name(request, "cls_logits")
            bbox_deltas_in = pb_utils.get_input_tensor_by_name(request, "bbox_deltas")
            mask_logits_in = pb_utils.get_input_tensor_by_name(request, "mask_logits")
            proposals_in = pb_utils.get_input_tensor_by_name(request, "proposals")

            cls_logits = cls_logits_in.as_numpy()
            bbox_deltas = bbox_deltas_in.as_numpy()
            mask_logits = mask_logits_in.as_numpy()
            proposals_xyxy = proposals_in.as_numpy()

            # Remove batch dimension: [1, K, ...] -> [K, ...]
            if cls_logits.ndim == 3 and cls_logits.shape[0] == 1:
                cls_logits = cls_logits[0]
            if bbox_deltas.ndim == 3 and bbox_deltas.shape[0] == 1:
                bbox_deltas = bbox_deltas[0]
            if mask_logits.ndim == 5 and mask_logits.shape[0] == 1:
                mask_logits = mask_logits[0]
            if proposals_xyxy.ndim == 3 and proposals_xyxy.shape[0] == 1:
                proposals_xyxy = proposals_xyxy[0]

            result = postprocess_detections(
                proposals_xyxy=proposals_xyxy,
                cls_logits=cls_logits,
                bbox_deltas=bbox_deltas,
                mask_logits=mask_logits,
                image_hw=self._image_hw,
                cfg=self._cfg,
            )

            # Apply same top-k behavior as MaskRCNN: keep only top-K detections by score
            TOPK = self._topk
            scores_all = result.scores
            if scores_all.shape[0] > TOPK:
                top_idx = np.argsort(scores_all)[-TOPK:][::-1]
            else:
                top_idx = np.arange(scores_all.shape[0])

            boxes_xyxy = result.boxes_xyxy[top_idx]
            scores = scores_all[top_idx]
            classes = result.classes[top_idx]

            num_det = boxes_xyxy.shape[0]

            # Build [N,2] tensor: [class_id, score]
            if num_det > 0:
                cls_ids = classes.astype(np.float32).reshape(-1, 1)
                score_cols = scores.reshape(-1, 1)
                cls_results_np = np.concatenate([cls_ids, score_cols], axis=1)
            else:
                cls_results_np = np.zeros((0, 2), dtype=np.float32)

            cls_counts_np = np.array([num_det], dtype=np.int32)
            boxes_np = boxes_xyxy.astype(np.float32) if num_det > 0 else np.zeros((0, 4), dtype=np.float32)

            out_cls_results = pb_utils.Tensor("cls_results", cls_results_np)
            out_cls_counts = pb_utils.Tensor("cls_counts", cls_counts_np)
            out_boxes = pb_utils.Tensor("boxes", boxes_np)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_cls_results, out_cls_counts, out_boxes])
            )

        return responses

    def finalize(self) -> None:
        # Nothing to clean up
        return
