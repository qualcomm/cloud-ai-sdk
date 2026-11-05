####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from .postprocess import PostprocessConfig, PostprocessResult, postprocess_detections
from .preprocess import MASK_RCNN_IMAGE_SIZE


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float


@dataclass(frozen=True)
class PredictResult:
    post: PostprocessResult
    latencies: Dict[str, LatencyStats]


def latency_stats(samples_ms: list[float]) -> LatencyStats:
    a = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        mean_ms=float(a.mean()),
        p50_ms=float(np.percentile(a, 50)),
        p90_ms=float(np.percentile(a, 90)),
        p99_ms=float(np.percentile(a, 99)),
    )


def time_callable(run_once: Callable[[], object], warmup: int, runs: int) -> Tuple[object, LatencyStats]:
    for _ in range(max(0, warmup)):
        run_once()

    samples = []
    out = None
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        out = run_once()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)

    return out, latency_stats(samples)


def time_onnx_session(sess, feed: dict, warmup: int, runs: int) -> Tuple[list, LatencyStats]:
    out, stats = time_callable(lambda: sess.run(None, feed), warmup=warmup, runs=runs)
    return out, stats


def time_qaic_session(sess, feed: dict, warmup: int, runs: int) -> Tuple[dict, LatencyStats]:
    out, stats = time_callable(lambda: sess.run(feed), warmup=warmup, runs=runs)
    return out, stats


def time_mask_rcnn_postprocess(
    proposals,
    cls_logits,
    bbox_deltas,
    mask_logits,
    cfg: PostprocessConfig,
    warmup: int,
    runs: int,
) -> Tuple[PostprocessResult, LatencyStats]:
    post, stats = time_callable(
        lambda: postprocess_detections(
            proposals_xyxy=proposals,
            cls_logits=cls_logits,
            bbox_deltas=bbox_deltas,
            mask_logits=mask_logits,
            image_hw=MASK_RCNN_IMAGE_SIZE.hw,
            cfg=cfg,
        ),
        warmup=warmup,
        runs=runs,
    )
    assert post is not None
    return post, stats
