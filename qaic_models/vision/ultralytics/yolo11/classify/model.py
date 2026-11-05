####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

from typing import Optional, Protocol

from qaic_models.vision.ultralytics.common.classify_runtime import (
    UltralyticsClassifyOnnxCPU,
    UltralyticsClassifyQaic,
)

from .postprocess import postprocess_classification

SUPPORTED_VARIANTS = ("yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls")
DEFAULT_VARIANT = "yolo11s-cls"


class _YOLO11ClassifyPredictor(Protocol):
    def predict(self, image_path: str, output_path: Optional[str]) -> dict: ...


class YOLO11ClassifyOnnxCPU(UltralyticsClassifyOnnxCPU):
    DEFAULT_VARIANT = DEFAULT_VARIANT
    SUPPORTED_VARIANTS = SUPPORTED_VARIANTS
    postprocess_classification = staticmethod(postprocess_classification)


class YOLO11ClassifyQaic(UltralyticsClassifyQaic):
    DEFAULT_QPC_SUBDIR = "qpc_latency"
    postprocess_classification = staticmethod(postprocess_classification)


def _require_variant(variant: str) -> str:
    return YOLO11ClassifyOnnxCPU._require_variant(variant)


class YOLO11Classify:
    @classmethod
    def from_compiled(cls, artifact_dir: str, variant: str = DEFAULT_VARIANT, provider: str = "QAicExecutionProvider", device_id: int | None = None, **kwargs):
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        return cls(artifact_dir, variant=variant, provider=provider, device_id=device_id)

    def __init__(self, artifact_dir: str, variant: str = DEFAULT_VARIANT, provider: str = "QAicExecutionProvider", device_id: int | None = None):
        variant = _require_variant(variant)
        provider = provider.strip()
        if provider == "CPUExecutionProvider":
            self._impl: _YOLO11ClassifyPredictor = YOLO11ClassifyOnnxCPU(artifact_dir, variant=variant, providers=[provider])
        elif provider == "QAicExecutionProvider":
            self._impl = YOLO11ClassifyQaic(artifact_dir, providers=[provider], device_id=device_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def predict(self, image_path: str, output_path: Optional[str] = None) -> dict:
        return self._impl.predict(image_path, output_path)
