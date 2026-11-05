####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def extract_named_output(outputs, output_name: str) -> np.ndarray:
    if isinstance(outputs, dict):
        if output_name not in outputs:
            raise ValueError(f"missing expected QAIC output {output_name!r}; got {sorted(outputs)}")
        return outputs[output_name]
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise ValueError(f"expected exactly one ONNX output ({output_name!r}); got {len(outputs)}")
        return outputs[0]
    return outputs


def label_for_class(class_id: int, labels: Sequence[str]) -> str:
    return labels[class_id] if labels and class_id < len(labels) else str(class_id)


def postprocess_classification_probs(probs, labels: Sequence[str], topk: int = 5) -> dict:
    arr = np.asarray(probs)
    if arr.ndim == 2:
        if arr.shape[0] != 1:
            raise ValueError(f"expected classification output shape [1, 1000], got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] != 1000:
        raise ValueError(f"expected classification output shape [1, 1000] or [1000], got {arr.shape}")

    k = min(topk, arr.shape[0])
    order = np.argsort(arr)[::-1][:k]
    topk_items = [
        {
            "class_id": int(class_id),
            "label": label_for_class(int(class_id), labels),
            "score": float(arr[class_id]),
        }
        for class_id in order
    ]
    return {"top1": topk_items[0] if topk_items else None, "topk": topk_items}
