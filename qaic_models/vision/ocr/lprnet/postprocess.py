####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

"""LPRNet output interpretation: greedy CTC decode of the logits tensor.

The model outputs a single tensor ``logits`` of shape (1, 68, 18):
  - dim 1: 68 character classes (index 67 = '-' = CTC blank)
  - dim 2: 18 time steps

Decode: argmax over classes at each time step, collapse consecutive repeats,
strip blank tokens. Confidence is the mean of the softmax-max probability over
the kept (non-blank, non-repeat) steps.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

OUTPUT_NAME = "logits"
NUM_CLASSES = 68
NUM_TIMESTEPS = 18
BLANK_IDX = 67  # '-' is the CTC blank token (last index in CHARS)

# Chinese license plate character set (68 chars, index 67 = '-' = CTC blank).
# From LPRNet_Pytorch/data/load_data.py. Embedded directly — no download needed.
CHARS = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "I", "O", "-",
]


def _extract_logits(outputs) -> np.ndarray:
    """Normalize raw predictor output to the validated logits tensor.

    QAICPredictor._forward() returns a dict keyed by tensor name; CPUPredictor's
    default _forward() returns the ONNX Runtime list output.
    """
    if isinstance(outputs, dict):
        if OUTPUT_NAME not in outputs:
            raise ValueError(f"missing expected QAIC output {OUTPUT_NAME!r}; got {sorted(outputs)}")
        arr = outputs[OUTPUT_NAME]
    elif isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise ValueError(f"expected exactly one ONNX output ({OUTPUT_NAME!r}); got {len(outputs)}")
        arr = outputs[0]
    else:
        arr = outputs

    arr = np.asarray(arr)
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    if arr.shape != (1, NUM_CLASSES, NUM_TIMESTEPS):
        raise ValueError(f"expected logits shape (1, {NUM_CLASSES}, {NUM_TIMESTEPS}); got {arr.shape}")
    return arr


def decode_ctc(logits: np.ndarray, chars: Sequence[str] = CHARS) -> dict:
    """Greedy CTC decode of a (1, 68, 18) logits tensor.

    Returns the canonical prediction dict:
        {"plate": str, "confidence": float, "characters": [str]}
    """
    logits = logits[0]  # (68, 18)

    # Greedy argmax over classes at each time step -> [18]
    preds = np.argmax(logits, axis=0)

    # Softmax for confidence (per time step)
    exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
    softmax = exp_logits / exp_logits.sum(axis=0, keepdims=True)  # (68, 18)
    max_probs = softmax.max(axis=0)  # (18,)

    characters: list[str] = []
    confidences: list[float] = []
    prev = -1
    for t, c in enumerate(preds):
        if c != BLANK_IDX and c != prev:
            characters.append(chars[c])
            confidences.append(float(max_probs[t]))
        prev = c

    plate_text = "".join(characters)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return {
        "plate": plate_text,
        "confidence": round(confidence, 4),
        "characters": characters,
    }


def postprocess_lprnet(outputs, chars: Sequence[str] = CHARS) -> dict:
    """Build the canonical LPRNet prediction dict from raw predictor output."""
    logits = _extract_logits(outputs)
    return decode_ctc(logits, chars)
