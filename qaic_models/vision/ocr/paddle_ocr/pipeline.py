# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Run the PaddleOCR layout -> det -> crop -> rec pipeline on one image."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Protocol, Sequence, TypedDict, cast

from . import postprocess, preprocess, visualize
from .model import (
    PaddleOCRLayoutOnnxCPU,
    PaddleOCRLayoutQaic,
    PaddleOCRDetOnnxCPU,
    PaddleOCRDetQaic,
    PaddleOCRRecOnnxCPU,
    PaddleOCRRecQaic,
    REC_BATCH,
)
from .workers import _WorkerPool


class OCRPrediction(TypedDict):
    """Prediction payload returned by the layout/det/rec pipeline."""

    texts: list[dict[str, Any]]
    regions: list[dict[str, Any]]


class _OCRPipelineResultBase(TypedDict):
    prediction: OCRPrediction
    latency_ms: float


class OCRPipelineResult(_OCRPipelineResultBase, total=False):
    """Stable outer result with an optional rendered-output list."""

    output_paths: list[str]


class _DetPredictor(Protocol):
    get_font: Any

    def run_image(self, image: Any) -> tuple[list, list]:
        """Run one detector pass on an in-memory image crop."""


class _RecPredictor(Protocol):
    def run_batch(self, crops: Sequence[Any]) -> list:
        """Run one recognition batch on line crops."""


class _DetSessionWorker:
    """Adapter that binds one detector predictor/session to one serial queue."""

    def __init__(self, predictor: _DetPredictor) -> None:
        self.predictor = predictor

    def run(self, crop: Any) -> tuple[list, list]:
        return self.predictor.run_image(crop)


class _RecSessionWorker:
    """Adapter that binds one recognizer predictor/session to one serial queue."""

    def __init__(self, predictor: _RecPredictor) -> None:
        self.predictor = predictor

    def run(self, group: Sequence[Any]) -> list:
        return self.predictor.run_batch(group)


class PaddleOCRPipeline:
    """Run layout, detection, cropping, and recognition for one image.

    The instance owns one layout predictor, one detector predictor per
    ``det_devices`` entry, and ``rec_sessions`` recognizer predictors for its
    entire lifetime. Timed detector and recognizer calls use one serial worker
    queue per predictor/session, so a session is never invoked concurrently;
    calls on the same pipeline instance are also serialized. The pool owns
    only temporary executor threads and does not close predictor sessions.

    Results have the form ``{"prediction": {"texts": [...],
    "regions": [...]}, "latency_ms": float}``, with ``output_paths`` added
    as a one-item list when ``output_path`` is supplied.
    """

    def __init__(
        self,
        model_dir: str,
        provider: str = "QAicExecutionProvider",
        *,
        layout_device_id: int = 0,
        det_devices: tuple[int, ...] = (1, 2),
        rec_device_id: int = 3,
        rec_sessions: int = 4,
    ) -> None:
        if provider not in {"CPUExecutionProvider", "QAicExecutionProvider"}:
            raise ValueError(f"Unsupported provider: {provider}")
        if provider == "QAicExecutionProvider":
            if not det_devices:
                raise ValueError("det_devices must be non-empty")
            if len(set(det_devices)) != len(det_devices):
                raise ValueError(f"det_devices must not contain duplicates: {det_devices}")
            if rec_sessions <= 0:
                raise ValueError("rec_sessions must be positive")
        self.provider = provider
        self._predict_lock = Lock()
        if provider == "CPUExecutionProvider":
            self.layout = PaddleOCRLayoutOnnxCPU(model_dir, providers=[provider])
            self.dets = [PaddleOCRDetOnnxCPU(model_dir, providers=[provider])]
            self.recs = [PaddleOCRRecOnnxCPU(model_dir, providers=[provider])]
        else:
            self.layout = PaddleOCRLayoutQaic(model_dir, device_id=layout_device_id)
            self.dets = [PaddleOCRDetQaic(model_dir, device_id=device) for device in det_devices]
            self.recs = [PaddleOCRRecQaic(model_dir, device_id=rec_device_id) for _ in range(rec_sessions)]

    def predict(self, image_path: str, *, output_path: str | None = None) -> OCRPipelineResult:
        """Run one serialized prediction and return the structured result."""
        with self._predict_lock:
            return self._predict_impl(image_path, output_path=output_path)

    def _predict_impl(self, image_path: str, *, output_path: str | None = None) -> OCRPipelineResult:
        """Run the warmup and timed pass while the instance lock is held."""
        img_bgr = preprocess._load_bgr(image_path)
        # Warm every session: the first QPC inference pays one-time device setup,
        # so the timed pass must never be the first one. The rec sessions warm on
        # a small corner crop (line content is irrelevant to the warmup).
        self.layout.run_image(img_bgr)
        for det in self.dets:
            det.run_image(img_bgr)
        warmup_crop = preprocess.crop_box(
            img_bgr, [0, 0, min(img_bgr.shape[1], 100), min(img_bgr.shape[0], 30)]
        )
        for rec in self.recs:
            rec.run_batch([warmup_crop])

        t0 = time.perf_counter()
        # 1. Layout: the document's regions, in reading order.
        layout_pred = self.layout.run_image(img_bgr)
        layout_regions, ocr_regions, placeholders = postprocess.split_layout_regions(layout_pred)
        # 2. Det: spread the OCR-eligible region crops over the det sessions;
        # offset each region's boxes into image coordinates, keeping its
        # reading order.
        region_crops = []
        for region in ocr_regions:
            region_crop = preprocess.crop_box(img_bgr, region["box"])
            if region_crop is not None:
                region_crops.append((region, region_crop))
        det_results = _run_det_groups(self.dets, [crop for _, crop in region_crops])
        detections = []
        for (region, _), (boxes, scores) in zip(region_crops, det_results):
            x0, y0 = region["box"][:2]
            detections.extend(
                ([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0], score, region)
                for box, score in postprocess.reading_order(boxes, scores)
            )
        # 3. Rec: crop each line, then fan the two-line batches out over the rec
        # sessions; the decodes come back in group (reading) order.
        kept_pairs, kept_regions, crops = [], [], []
        for box, score, region in detections:
            crop = preprocess.crop_box(img_bgr, box)
            if crop is not None:
                kept_pairs.append((box, score))
                kept_regions.append(region)
                crops.append(crop)
        groups = [crops[i : i + REC_BATCH] for i in range(0, len(crops), REC_BATCH)]
        decodes = _run_rec_groups(self.recs, groups)
        prediction = cast(
            OCRPrediction,
            postprocess.build_prediction(
                layout_regions, placeholders, kept_pairs, kept_regions, decodes,
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result: OCRPipelineResult = {"prediction": prediction, "latency_ms": latency_ms}
        if output_path is not None:
            visualize.visualize_rec(img_bgr, prediction, output_path, self.dets[-1].get_font)
            result["output_paths"] = [output_path]
        return result


def _run_det_groups(
    dets: Sequence[_DetPredictor], region_crops: Sequence[Any],
) -> list[tuple[list, list]]:
    """Run detector crops on per-session serial queues in crop order."""
    if not region_crops:
        return []
    with _WorkerPool([_DetSessionWorker(det) for det in dets]) as pool:
        futures = [pool.submit(crop) for crop in region_crops]
        return [future.result() for future in futures]


def _run_rec_groups(
    recs: Sequence[_RecPredictor], groups: Sequence[Sequence[Any]],
) -> list:
    """Run recognizer groups on per-session serial queues in group order."""
    if not groups:
        return []
    with _WorkerPool([_RecSessionWorker(rec) for rec in recs]) as pool:
        futures = [pool.submit(group) for group in groups]
        return [decode for future in futures for decode in future.result()]
