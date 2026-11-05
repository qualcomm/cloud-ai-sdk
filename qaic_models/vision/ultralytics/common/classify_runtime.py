####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import json
import os

try:
    import qaic
except ImportError:
    qaic = None

from qaic_models.base.predict_base import CPUPredictor, QAICPredictor
from qaic_models.vision.ultralytics.common.preprocess import load_ultralytics_classify_image


def resolve_classify_onnx_path(model_path: str, variant: str) -> str:
    if model_path.endswith(".onnx"):
        return os.path.abspath(model_path)
    candidate = os.path.join(model_path, f"{variant}_fp16ready.onnx")
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"No generated ONNX artifact found at {candidate!r}.")


def resolve_classify_qpc_dir(qpc_dir: str, default_qpc_subdir: str) -> str:
    if os.path.exists(os.path.join(qpc_dir, "programqpc.bin")):
        return os.path.abspath(qpc_dir)
    candidate = os.path.join(qpc_dir, default_qpc_subdir)
    if os.path.exists(os.path.join(candidate, "programqpc.bin")):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"No generated QPC found at {qpc_dir!r} or {candidate!r}.")


class UltralyticsClassifyMixin:
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    LABELS_TMP = "imagenet_classes.txt"

    def preprocess(self, image_path: str) -> dict:
        image, _info = load_ultralytics_classify_image(image_path)
        return {"images": image}

    def postprocess(self, outputs, latency_ms: float) -> dict:
        prediction = self.postprocess_classification(outputs, self._get_labels())
        return {"prediction": prediction, "latency_ms": latency_ms}

    def visualize(self, image_path: str, prediction, output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=2)


class UltralyticsClassifyOnnxCPU(UltralyticsClassifyMixin, CPUPredictor):
    DEFAULT_VARIANT = ""
    SUPPORTED_VARIANTS: tuple[str, ...] = ()

    def __init__(self, model_path: str, variant: str | None = None, providers: list[str] | None = None):
        variant = self._require_variant(variant or self.DEFAULT_VARIANT)
        super().__init__(self._resolve_onnx_path(model_path, variant), providers=providers)

    @classmethod
    def _require_variant(cls, variant: str) -> str:
        if variant not in cls.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported variant: {variant!r}. Supported variants: {', '.join(cls.SUPPORTED_VARIANTS)}.")
        return variant

    @staticmethod
    def _resolve_onnx_path(model_path: str, variant: str) -> str:
        return resolve_classify_onnx_path(model_path, variant)


class UltralyticsClassifyQaic(UltralyticsClassifyMixin, QAICPredictor):
    DEFAULT_QPC_SUBDIR = "qpc_latency"

    def __init__(self, qpc_dir: str, providers: list[str] | None = None, device_id: int | None = None):
        if qaic is None:
            raise ImportError(f"qaic module not available; install the QAIC SDK to use {type(self).__name__}.")
        super().__init__(self._resolve_qpc_dir(qpc_dir), device_id=device_id)
        self.providers = providers or ["QAicExecutionProvider"]

    @classmethod
    def _resolve_qpc_dir(cls, qpc_dir: str) -> str:
        return resolve_classify_qpc_dir(qpc_dir, cls.DEFAULT_QPC_SUBDIR)
