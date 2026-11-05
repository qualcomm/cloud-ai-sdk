####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import os
from typing import Optional, Protocol

try:
    import qaic
except ImportError:
    qaic = None

from qaic_models.base.predict_base import CPUPredictor, QAICPredictor
from qaic_models.vision.ultralytics.common.preprocess import load_ultralytics_letterbox_image
from qaic_models.vision.ultralytics.yolo11.detect.visualize import visualize_detections

from .postprocess import PostprocessConfig, postprocess_detections

SUPPORTED_VARIANTS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")
DEFAULT_VARIANT = "yolo26m"


class _YOLO26DetectPredictor(Protocol):
    def predict(self, image_path: str, output_path: Optional[str]) -> dict: ...


class _YOLO26DetectRecipeMixin:
    LABELS_URL = "COCO_80"
    LABELS_TMP = ""

    def preprocess(self, image_path: str) -> dict:
        image, letterbox = load_ultralytics_letterbox_image(image_path)
        self._letterbox = letterbox
        return {"images": image}

    def postprocess(self, outputs, latency_ms: float) -> dict:
        prediction = postprocess_detections(outputs, self._letterbox, self._get_labels(), cfg=PostprocessConfig())
        return {"prediction": prediction, "latency_ms": latency_ms}

    def visualize(self, image_path: str, prediction, output_path: str):
        visualize_detections(image_path, prediction, output_path, self.get_font)


def _require_variant(variant: str) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported YOLO26 detect variant: {variant!r}. Supported variants: {', '.join(SUPPORTED_VARIANTS)}.")
    return variant


class YOLO26DetectOnnxCPU(_YOLO26DetectRecipeMixin, CPUPredictor):
    def __init__(self, model_path: str, variant: str = DEFAULT_VARIANT, providers: Optional[list[str]] = None):
        variant = _require_variant(variant)
        super().__init__(self._resolve_onnx_path(model_path, variant), providers=providers)

    @staticmethod
    def _resolve_onnx_path(model_path: str, variant: str) -> str:
        if model_path.endswith(".onnx"):
            return os.path.abspath(model_path)
        candidate = os.path.join(model_path, f"{variant}_fp16ready.onnx")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        raise FileNotFoundError(f"No generated ONNX artifact found at {candidate!r}.")


class YOLO26DetectQaic(_YOLO26DetectRecipeMixin, QAICPredictor):
    DEFAULT_QPC_SUBDIR = "qpc_latency"

    def __init__(self, qpc_dir: str, providers: Optional[list[str]] = None, device_id: int | None = None):
        if qaic is None:
            raise ImportError("qaic module not available; install the QAIC SDK to use YOLO26DetectQaic.")
        super().__init__(self._resolve_qpc_dir(qpc_dir), device_id=device_id)
        self.providers = providers or ["QAicExecutionProvider"]

    @classmethod
    def _resolve_qpc_dir(cls, qpc_dir: str) -> str:
        if os.path.exists(os.path.join(qpc_dir, "programqpc.bin")):
            return os.path.abspath(qpc_dir)
        candidate = os.path.join(qpc_dir, cls.DEFAULT_QPC_SUBDIR)
        if os.path.exists(os.path.join(candidate, "programqpc.bin")):
            return os.path.abspath(candidate)
        raise FileNotFoundError(f"No generated QPC found at {qpc_dir!r} or {candidate!r}.")


class YOLO26Detect:
    @classmethod
    def from_compiled(cls, artifact_dir: str, variant: str = DEFAULT_VARIANT, provider: str = "QAicExecutionProvider", device_id: int | None = None, **kwargs):
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        return cls(artifact_dir, variant=variant, provider=provider, device_id=device_id)

    def __init__(self, artifact_dir: str, variant: str = DEFAULT_VARIANT, provider: str = "QAicExecutionProvider", device_id: int | None = None):
        provider = provider.strip()
        if provider == "CPUExecutionProvider":
            self._impl: _YOLO26DetectPredictor = YOLO26DetectOnnxCPU(artifact_dir, variant=variant, providers=[provider])
        elif provider == "QAicExecutionProvider":
            self._impl = YOLO26DetectQaic(artifact_dir, providers=[provider], device_id=device_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def predict(self, image_path: str, output_path: Optional[str] = None) -> dict:
        return self._impl.predict(image_path, output_path)
