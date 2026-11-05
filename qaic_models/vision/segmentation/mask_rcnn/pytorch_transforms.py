# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations


def _static_roi_pooler_forward(self, x, box_lists):
    import torch
    from detectron2.layers import nonzero_tuple
    from detectron2.modeling.poolers import _create_zeros, assign_boxes_to_levels, convert_boxes_to_pooler_format

    num_level_assignments = len(box_lists)
    if num_level_assignments == 0:
        return _create_zeros([], x[0].shape[1], self.output_size, self.output_size, x[0])

    pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
    level_assignments = assign_boxes_to_levels(
        box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
    )

    num_channels = x[0].shape[1]
    output_size = self.output_size[0] if isinstance(self.output_size, (tuple, list)) else self.output_size
    output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])

    if torch.onnx.is_in_onnx_export():
        output_sum = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])
        for level, pooler in enumerate(self.level_poolers):
            pooled_all = pooler(x[level], pooler_fmt_boxes)
            mask = (level_assignments == level).to(dtype=pooled_all.dtype)
            output_sum = output_sum + pooled_all * mask[:, None, None, None]
        return output_sum

    for level, pooler in enumerate(self.level_poolers):
        inds = nonzero_tuple(level_assignments == level)[0]
        pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
        output[inds] = pooler(x[level], pooler_fmt_boxes_level)
    return output


def static_find_top_rpn_proposals(
    proposals,
    pred_objectness_logits,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_size,
    training,
):
    import torch
    from detectron2.layers import batched_nms
    from detectron2.modeling.proposal_generator.proposal_utils import move_device_like
    from detectron2.structures import Boxes, Instances

    num_images = len(image_sizes)
    device = proposals[0].device

    topk_scores = []
    topk_proposals = []
    level_ids = []
    batch_idx = move_device_like(torch.arange(num_images, device=device), proposals[0])
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        if torch.onnx.is_in_onnx_export():
            num_proposals_i = pre_nms_topk
        else:
            hi_wi_a = logits_i.shape[1]
            num_proposals_i = torch.clamp(hi_wi_a, max=pre_nms_topk) if isinstance(hi_wi_a, torch.Tensor) else min(hi_wi_a, pre_nms_topk)

        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]
        topk_scores.append(topk_scores_i)
        topk_proposals.append(topk_proposals_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    topk_scores = torch.cat(topk_scores, dim=1)
    topk_proposals = torch.cat(topk_proposals, dim=1)
    level_ids = torch.cat(level_ids, dim=0)

    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        if torch.onnx.is_in_onnx_export():
            boxes.clip(image_size)
        else:
            valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
            if not valid_mask.all():
                if training:
                    raise FloatingPointError("Predicted boxes or scores contain Inf/NaN. Training has diverged.")
                boxes = boxes[valid_mask]
                scores_per_img = scores_per_img[valid_mask]
                lvl = lvl[valid_mask]
            boxes.clip(image_size)

            keep_nonempty = boxes.nonempty(threshold=min_box_size)
            if keep_nonempty.sum().item() != len(boxes):
                boxes, scores_per_img, lvl = boxes[keep_nonempty], scores_per_img[keep_nonempty], lvl[keep_nonempty]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]
        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


def apply_mask_rcnn_export_overlays(model):
    from detectron2.modeling import poolers
    from detectron2.modeling.proposal_generator import proposal_utils, rpn
    from qaic_models.base.pytorch_transforms import FunctionPatchTransform, MethodPatchTransform

    class StaticROIPoolerTransform(MethodPatchTransform):
        _target_class = poolers.ROIPooler
        _method_name = "forward"
        replacement = staticmethod(_static_roi_pooler_forward)

    class StaticRPNProposalUtilsTransform(FunctionPatchTransform):
        _target_module = proposal_utils
        _function_name = "find_top_rpn_proposals"
        _replacement = static_find_top_rpn_proposals

    class StaticRPNModuleTransform(FunctionPatchTransform):
        _target_module = rpn
        _function_name = "find_top_rpn_proposals"
        _replacement = static_find_top_rpn_proposals

    transforms = [StaticROIPoolerTransform, StaticRPNProposalUtilsTransform, StaticRPNModuleTransform]
    changed = False
    for transform in transforms:
        model, did_change = transform.apply(model)
        changed = changed or did_change
    return model, transforms, changed


def undo_mask_rcnn_export_overlays(transforms) -> None:
    for transform in reversed(transforms):
        transform.undo()
