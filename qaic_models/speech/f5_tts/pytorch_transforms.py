####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations


def _vocos_backbone_forward(self, x, **kwargs):
    import torch

    x = self.embed(x)
    x = self.norm.weight.data * x / torch.norm(x, p=2, dim=1, keepdim=True) + self.norm.bias.data
    for conv_block in self.convnext:
        x = conv_block(x, cond_embedding_id=None)
    return self.final_layer_norm.weight.data * x / torch.norm(x, p=2, dim=1, keepdim=True) + self.final_layer_norm.bias.data


def _vocos_convnext_forward(self, x, cond_embedding_id=None):
    import torch

    residual = x
    x = self.dwconv(x)
    x = self.norm.weight.data * x / torch.norm(x, p=2, dim=1, keepdim=True) + self.norm.bias.data
    x = torch.matmul(self.pwconv1.weight.data, x) + self.pwconv1.bias.data
    x = self.act(x)
    x = torch.matmul(self.pwconv2.weight.data, x) + self.pwconv2.bias.data
    return residual + x


def _vocos_istft_head_forward(self, x):
    import torch

    x = torch.matmul(self.out.weight.data, x) + self.out.bias.data
    mag, p = x.chunk(2, dim=1)
    mag = torch.exp(mag)
    mag = torch.clip(mag, max=1e2)
    return (mag, p)


def _reshape_vocos_layer_norm(norm):
    import torch

    norm.weight.data = (norm.weight.data * torch.sqrt(torch.tensor(norm.weight.data.shape[0], dtype=torch.float32))).view(1, -1, 1)
    norm.bias.data = norm.bias.data.view(1, -1, 1)


def _reshape_vocos_weights_for_onnx_export(vocos_model):
    """Irreversible weight-shape surgery for the ONNX-friendly Vocos forward methods.

    Not a method patch — these are one-time tensor reshapes (layer-norm scale folding,
    pointwise-conv weight unsqueezing, gamma folding into pwconv2), not undoable via
    MethodPatchTransform.undo(). Call only on a model instance dedicated to export.
    """
    _reshape_vocos_layer_norm(vocos_model.backbone.norm)
    _reshape_vocos_layer_norm(vocos_model.backbone.final_layer_norm)
    vocos_model.head.out.bias.data = vocos_model.head.out.bias.data.view(1, -1, 1)
    for block in vocos_model.backbone.convnext:
        _reshape_vocos_layer_norm(block.norm)
        block.pwconv1.weight.data = block.pwconv1.weight.data.unsqueeze(0)
        block.pwconv1.bias.data = block.pwconv1.bias.data.view(1, -1, 1)
        if block.gamma is not None:
            block.pwconv2.weight.data = (block.gamma.data.unsqueeze(-1) * block.pwconv2.weight.data).unsqueeze(0)
            block.pwconv2.bias.data = (block.gamma.data * block.pwconv2.bias.data).view(1, -1, 1)
        else:
            block.pwconv2.weight.data = block.pwconv2.weight.data.unsqueeze(0)
            block.pwconv2.bias.data = block.pwconv2.bias.data.view(1, -1, 1)


def apply_vocos_export_overlays(vocos_model):
    """Patch Vocos's backbone/convnext/head forward methods for ONNX-friendly export.

    Replaces normalized-layer ops (torch.norm-based RMSNorm, grouped conv) with
    tensor-op equivalents that trace cleanly to ONNX. Uses MethodPatchTransform
    (class-level, undoable) instead of inline `model.forward = ...` assignment.
    """
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone
    from vocos.modules import ConvNeXtBlock

    from qaic_models.base.pytorch_transforms import MethodPatchTransform

    class VocosBackboneForwardTransform(MethodPatchTransform):
        _target_class = VocosBackbone
        _method_name = "forward"
        replacement = staticmethod(_vocos_backbone_forward)

    class VocosConvNeXtForwardTransform(MethodPatchTransform):
        _target_class = ConvNeXtBlock
        _method_name = "forward"
        replacement = staticmethod(_vocos_convnext_forward)

    class VocosISTFTHeadForwardTransform(MethodPatchTransform):
        _target_class = ISTFTHead
        _method_name = "forward"
        replacement = staticmethod(_vocos_istft_head_forward)

    transforms = [VocosBackboneForwardTransform, VocosConvNeXtForwardTransform, VocosISTFTHeadForwardTransform]
    changed = False
    for transform in transforms:
        vocos_model, did_change = transform.apply(vocos_model)
        changed = changed or did_change

    _reshape_vocos_weights_for_onnx_export(vocos_model)

    return vocos_model, transforms, changed


def undo_vocos_export_overlays(transforms) -> None:
    for transform in reversed(transforms):
        transform.undo()
