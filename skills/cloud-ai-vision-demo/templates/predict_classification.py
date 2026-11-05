#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict.py — <MODEL NAME> classification inference on Cloud AI 100.

TEMPLATE: cp this file to /workspace/<model>/predict.py, fill every FILL_IN marker,
then smoke-test before generating the MODEL_CARD.
Classification models produce no output image — visualize() is intentionally a no-op."""

import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict_base import QAICPredictor

# ── START FILL-IN SECTION ─────────────────────────────────────────────────────
class ModelPredictor(QAICPredictor):
    INPUT_H   = 0   # FILL_IN: compiled input height (e.g. 224)
    INPUT_W   = 0   # FILL_IN: compiled input width  (e.g. 224)
    # FILL_IN: URL to a one-label-per-line text file matching the model's class vocabulary.
    # ImageNet-1k synsets: "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    # Verify: len(self._get_labels()) must equal the number of logits from the model.
    LABELS_URL = "FILL_IN"
    LABELS_TMP = "/tmp/FILL_IN_labels.txt"   # FILL_IN: use a unique /tmp name per model

    def __init__(self):
        super().__init__("FILL_IN_qpc_dir")

    def preprocess(self, image_path: str) -> dict:
        """Verify the resize target and normalization against export_onnx.py. Standard torchvision pipeline:
        1. Resize: scale = RESIZE_SHORT / min(w, h)  ← shortest side, NOT width alone
                   new_w = int(w * scale), new_h = int(h * scale)
        2. Centre-crop to INPUT_W × INPUT_H:
                   left = (new_w - INPUT_W) // 2     ← use POST-resize dims, not INPUT constants
                   top  = (new_h - INPUT_H) // 2
        3. Normalize: (arr/255 - mean) / std; ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        4. HWC → NCHW float32.
        Return {"input_name": NCHW array} — use self._session.model_input_names[0] for the key.
        Some models use direct resize to (INPUT_W, INPUT_H) instead of shortest-side + crop.
        """
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        """Flatten first output to 1-D logits. Apply softmax.
        Get top-5 (or top-10) indices by descending prob.
        Use self._get_labels() for label name lookup (fall back to str index).
        Return {'prediction': [{'label': str, 'score': float}, ...], 'latency_ms': float}.
        List must be sorted highest-score first.
        """
        raise NotImplementedError

    # ⚠ DO NOT implement visualize() — classification produces no output image.
    # The --output argument exists for CLI consistency only; nothing is written here.
    def visualize(self, image_path: str, prediction, output_path: str):
        return


if __name__ == "__main__":
    ModelPredictor().main()
