# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import torch
from torch import nn


class BackboneFPNExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_height: int = 800, input_width: int = 1216):
        super().__init__()
        self.model = model
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.eval()

    def forward(self, image_nchw: torch.Tensor):
        inputs = [{"image": image_nchw[0], "height": self.input_height, "width": self.input_width}]
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        return features["p2"], features["p3"], features["p4"], features["p5"], features["p6"]


class RPNExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_height: int = 800, input_width: int = 1216):
        super().__init__()
        self.model = model
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.eval()

    def forward(self, p2, p3, p4, p5, p6):
        class _ImagesForRPN:
            def __init__(self, height: int, width: int, device, dtype):
                self.image_sizes = [(height, width)]
                self.tensor = torch.zeros((1, 3, height, width), device=device, dtype=dtype)

        features = {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}
        images = _ImagesForRPN(self.input_height, self.input_width, p2.device, p2.dtype)
        proposals, _ = self.model.proposal_generator(images, features, None)
        return proposals[0].proposal_boxes.tensor


class ROIHeadsExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_height: int = 800, input_width: int = 1216):
        super().__init__()
        self.model = model
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.eval()

    def forward(self, p2, p3, p4, p5, proposals_xyxy):
        from detectron2.structures import Boxes, Instances

        roi_features = {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
        inst = Instances((self.input_height, self.input_width))
        inst.proposal_boxes = Boxes(proposals_xyxy)
        inst.objectness_logits = torch.zeros(
            (proposals_xyxy.shape[0],), dtype=proposals_xyxy.dtype, device=proposals_xyxy.device
        )
        proposals = [inst]

        box_features = self.model.roi_heads.box_pooler(
            [roi_features[f] for f in self.model.roi_heads.box_in_features],
            [x.proposal_boxes for x in proposals],
        )
        box_features = self.model.roi_heads.box_head(box_features)
        cls_logits, bbox_deltas = self.model.roi_heads.box_predictor(box_features)

        if getattr(self.model.roi_heads, "mask_on", False) and self.model.roi_heads.mask_pooler is not None:
            mask_features = self.model.roi_heads.mask_pooler(
                [roi_features[f] for f in self.model.roi_heads.mask_in_features],
                [x.proposal_boxes for x in proposals],
            )
            mask_logits = self.model.roi_heads.mask_head.layers(mask_features)
        else:
            mask_logits = torch.empty((cls_logits.shape[0], 80, 28, 28), device=cls_logits.device)

        return cls_logits, bbox_deltas, mask_logits
