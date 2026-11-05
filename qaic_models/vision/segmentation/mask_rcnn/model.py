####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import os
from typing import Dict, Optional, Protocol

try:
    import qaic
except ImportError:  # QAIC SDK is only available in Cloud AI runtime containers.
    qaic = None

from qaic_models.base.predict_base import CPUPredictor, QAICPredictor

from .benchmark import (
    LatencyStats,
    PredictResult,
    time_mask_rcnn_postprocess,
    time_onnx_session,
    time_qaic_session,
)

# Generated-QPC directory names under --work-dir (default build/mask_rcnn), following
# the qpc_* convention the other recipes and the functional harness's qpc_profile use.
BACKBONE_QPC_SUBDIR = "qpc_backbone_fpn"
ROI_HEADS_QPC_SUBDIR = "qpc_roi_heads"
from .preprocess import MASK_RCNN_IMAGE_SIZE, MaskRCNNImage
from .postprocess import PostprocessConfig, postprocess_detections
from .visualize import visualize_instances


class _MaskRcnnPredictor(Protocol):
    post_cfg: PostprocessConfig

    def predict(self, image_path: str, output_path: str | None) -> dict: ...

    def predict_benchmark(
        self,
        image_path: str,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 100,
        warmup: int = 1,
        runs: int = 10,
    ) -> PredictResult: ...


class _MaskRCNNRecipeMixin:
    LABELS_URL = "COCO_80"
    LABELS_TMP = ""
    post_cfg: PostprocessConfig = PostprocessConfig()

    def preprocess(self, image_path: str) -> dict:
        return {"image": MaskRCNNImage.load(image_path)}

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        post = postprocess_detections(
            proposals_xyxy=outputs["proposals"],
            cls_logits=outputs["cls_logits"],
            bbox_deltas=outputs["bbox_deltas"],
            mask_logits=outputs["mask_logits"],
            image_hw=MASK_RCNN_IMAGE_SIZE.hw,
            cfg=self.post_cfg,
        )
        self._instance_masks = post.masks
        labels = self._get_labels()
        instances = [
            {
                "label": labels[c] if labels and c < len(labels) else str(c),
                "score": float(s),
                "box_xyxy": [float(v) for v in post.boxes_xyxy[j]],
            }
            for j, (c, s) in enumerate(zip(post.classes.tolist(), post.scores.tolist()))
        ]
        return {"prediction": {"instances": instances, "num_instances": len(instances)}, "latency_ms": latency_ms}

    def visualize(self, image_path: str, prediction, output_path: str):
        visualize_instances(
            image_path,
            prediction,
            output_path,
            getattr(self, "_instance_masks", None),
            self.get_font,
        )


class MaskRCNNOnnxCPU(_MaskRCNNRecipeMixin, CPUPredictor):
    """Split-ONNX CPU reference runner for Mask R-CNN.

    Expected files in model_dir:
      - model_backbone_fpn.onnx
      - model_rpn_fp16ready.onnx or model_rpn.onnx
      - model_roi_heads_fp16ready.onnx or model_roi_heads.onnx

    Postprocess (decode+NMS) runs in Python.
    """

    def __init__(self, model_dir: str, providers: Optional[list[str]] = None):
        self.providers = providers or list(self.DEFAULT_PROVIDERS)
        self._warmed_up = False
        model_dir = os.path.abspath(model_dir)
        self.model_dir = model_dir

        backbone_fp16ready = os.path.join(model_dir, "model_backbone_fpn_fp16ready.onnx")
        rpn_fp16ready = os.path.join(model_dir, "model_rpn_fp16ready.onnx")
        roi_fp16ready = os.path.join(model_dir, "model_roi_heads_fp16ready.onnx")
        self.backbone_path = backbone_fp16ready if os.path.exists(backbone_fp16ready) else os.path.join(model_dir, "model_backbone_fpn.onnx")
        self.rpn_path = rpn_fp16ready if os.path.exists(rpn_fp16ready) else os.path.join(model_dir, "model_rpn.onnx")
        self.roi_path = roi_fp16ready if os.path.exists(roi_fp16ready) else os.path.join(model_dir, "model_roi_heads.onnx")

        missing = [p for p in [self.backbone_path, self.rpn_path, self.roi_path] if not os.path.exists(p)]
        if missing:
            rel_missing = [os.path.relpath(p, model_dir) for p in missing]
            raise FileNotFoundError(
                "Missing ONNX model files in MaskRCNNOnnxCPU model_dir. "
                f"Missing: {', '.join(rel_missing)}. "
                "Please export models first and try again."
            )

        self.sess_backbone = self._create_session(self.backbone_path)
        self.sess_rpn = self._create_session(self.rpn_path)
        self.sess_roi = self._create_session(self.roi_path)

    def _forward(self, model_inputs: dict) -> dict:
        image_nchw_fp32 = model_inputs["image"]
        p2, p3, p4, p5, p6 = self.sess_backbone.run(None, {"image": image_nchw_fp32})
        rpn_inputs = {i.name: t for i, t in zip(self.sess_rpn.get_inputs(), [p2, p3, p4, p5, p6])}
        (proposals,) = self.sess_rpn.run(None, rpn_inputs)
        roi_inputs = {i.name: t for i, t in zip(self.sess_roi.get_inputs(), [p2, p3, p4, p5, proposals])}
        cls_logits, bbox_deltas, mask_logits = self.sess_roi.run(None, roi_inputs)
        return {
            "proposals": proposals,
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "mask_logits": mask_logits,
        }

    def predict_benchmark(
        self,
        image_path: str,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 100,
        warmup: int = 1,
        runs: int = 10,
    ) -> PredictResult:
        """Existing benchmarking entry point with per-stage latency statistics."""
        lat: Dict[str, LatencyStats] = {}

        image_nchw_fp32 = MaskRCNNImage.load(image_path)

        (p2, p3, p4, p5, p6), lat_backbone = time_onnx_session(
            self.sess_backbone,
            {"image": image_nchw_fp32},
            warmup=warmup,
            runs=runs,
        )
        lat["backbone_fpn"] = lat_backbone

        rpn_inputs = {i.name: t for i, t in zip(self.sess_rpn.get_inputs(), [p2, p3, p4, p5, p6])}
        (proposals,), lat_rpn = time_onnx_session(self.sess_rpn, rpn_inputs, warmup=warmup, runs=runs)
        lat["rpn"] = lat_rpn

        roi_inputs = {i.name: t for i, t in zip(self.sess_roi.get_inputs(), [p2, p3, p4, p5, proposals])}
        (cls_logits, bbox_deltas, mask_logits), lat_roi = time_onnx_session(
            self.sess_roi, roi_inputs, warmup=warmup, runs=runs
        )
        lat["roi_heads"] = lat_roi

        pp_cfg = PostprocessConfig(
            num_classes=80,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk=topk,
        )
        post, lat["python_postprocess"] = time_mask_rcnn_postprocess(
            proposals,
            cls_logits,
            bbox_deltas,
            mask_logits,
            pp_cfg,
            warmup=warmup,
            runs=runs,
        )
        return PredictResult(post=post, latencies=lat)


class MaskRCNNQaic(_MaskRCNNRecipeMixin, QAICPredictor):
    """QAIC-based implementation.

    Backbone+FPN and ROI heads run on the accelerator via qaic.Session. RPN and
    postprocess run on CPU using ONNXRuntime and Python, respectively.

    Expected directory layout (example, the default export work dir):
      build/mask_rcnn/
        qpc_backbone_fpn/programqpc.bin
        qpc_roi_heads/programqpc.bin
        model_rpn_fp16ready.onnx or model_rpn.onnx
    """

    def __init__(self, model_dir: str, providers: Optional[list[str]] = None, device_id: Optional[int] = None):
        if qaic is None:
            raise ImportError("qaic module not available; install the QAIC SDK to use MaskRCNNQaic.")

        model_dir = os.path.abspath(model_dir)
        self.model_dir = model_dir
        self.providers = providers or ["QAicExecutionProvider"]
        self._warmed_up = False

        self.backbone_prog = os.path.join(model_dir, BACKBONE_QPC_SUBDIR, "programqpc.bin")
        self.roi_prog = os.path.join(model_dir, ROI_HEADS_QPC_SUBDIR, "programqpc.bin")

        rpn_fp16ready = os.path.join(model_dir, "model_rpn_fp16ready.onnx")
        self.rpn_onnx = rpn_fp16ready if os.path.exists(rpn_fp16ready) else os.path.join(model_dir, "model_rpn.onnx")

        required_paths = [self.backbone_prog, self.roi_prog, self.rpn_onnx]
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            rel_missing = [os.path.relpath(p, model_dir) for p in missing]
            raise FileNotFoundError(
                "Missing QPC model binary or ONNX model in MaskRCNNQaic model_dir. "
                f"Missing: {', '.join(rel_missing)}. "
                "Please export models first and try again."
            )

        session_kwargs = {"num_activations": 1, "set_size": 1, "oversubscription_name": "group1"}
        if device_id is not None:
            session_kwargs["dev_id"] = device_id
        self.q_sess_backbone = qaic.Session(self.backbone_prog, **session_kwargs)
        self.q_sess_roi = qaic.Session(self.roi_prog, **session_kwargs)

        self.sess_rpn = CPUPredictor(self.rpn_onnx)._session

    def _forward(self, model_inputs: dict) -> dict:
        backbone_feed = {"image": model_inputs["image"]}
        backbone_out = self.q_sess_backbone.run(backbone_feed)
        p2, p3, p4, p5, p6 = [backbone_out[k] for k in ("p2", "p3", "p4", "p5", "p6")]

        rpn_inputs = {i.name: t for i, t in zip(self.sess_rpn.get_inputs(), [p2, p3, p4, p5, p6])}
        (proposals,) = self.sess_rpn.run(None, rpn_inputs)

        roi_feed = {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "proposals": proposals}
        roi_out = self.q_sess_roi.run(roi_feed)
        cls_logits, bbox_deltas, mask_logits = [roi_out[k] for k in ("cls_logits", "bbox_deltas", "mask_logits")]

        return {
            "proposals": proposals,
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "mask_logits": mask_logits,
        }

    def predict_benchmark(
        self,
        image_path: str,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 100,
        warmup: int = 1,
        runs: int = 10,
    ) -> PredictResult:
        """Existing benchmarking entry point with per-stage latency statistics."""
        lat: Dict[str, LatencyStats] = {}

        backbone_feed = {"image": MaskRCNNImage.load(image_path)}
        backbone_out, lat_backbone = time_qaic_session(self.q_sess_backbone, backbone_feed, warmup=warmup, runs=runs)
        p2, p3, p4, p5, p6 = [backbone_out[k] for k in ("p2", "p3", "p4", "p5", "p6")]
        lat["backbone_fpn"] = lat_backbone

        rpn_inputs = {i.name: t for i, t in zip(self.sess_rpn.get_inputs(), [p2, p3, p4, p5, p6])}
        (proposals,), lat_rpn = time_onnx_session(self.sess_rpn, rpn_inputs, warmup=warmup, runs=runs)
        lat["rpn"] = lat_rpn
        assert proposals is not None

        roi_feed = {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "proposals": proposals}
        roi_out, lat_roi = time_qaic_session(self.q_sess_roi, roi_feed, warmup=warmup, runs=runs)
        cls_logits, bbox_deltas, mask_logits = [roi_out[k] for k in ("cls_logits", "bbox_deltas", "mask_logits")]
        lat["roi_heads"] = lat_roi

        pp_cfg = PostprocessConfig(
            num_classes=80,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk=topk,
        )
        post, lat["python_postprocess"] = time_mask_rcnn_postprocess(
            proposals,
            cls_logits,
            bbox_deltas,
            mask_logits,
            pp_cfg,
            warmup=warmup,
            runs=runs,
        )
        return PredictResult(post=post, latencies=lat)


class MaskRCNN:
    """Factory wrapper.

    Instantiate this class with a provider; the concrete backend (and its
    sessions) is created lazily on the first predict()/predict_benchmark() call,
    so constructing the factory never touches hardware or requires the generated
    artifacts to exist yet.

    Example:
        m = MaskRCNN("build/mask_rcnn", provider="CPUExecutionProvider")
        m = MaskRCNN("build/mask_rcnn", provider="QAicExecutionProvider")
    """

    def __init__(self, model_dir: str, provider: str = "QAicExecutionProvider", *, device_id: Optional[int] = None):
        provider = provider.strip()
        if provider not in ("CPUExecutionProvider", "QAicExecutionProvider"):
            raise ValueError(f"Unsupported provider: {provider}")
        self._model_dir = model_dir
        self._provider = provider
        self._device_id = device_id
        self._impl: Optional[_MaskRcnnPredictor] = None

    @classmethod
    def from_compiled(cls, qpc_dir: str, provider: str = "QAicExecutionProvider", **kwargs):
        return cls(qpc_dir, provider=provider, **kwargs)

    def _ensure_impl(self) -> _MaskRcnnPredictor:
        if self._impl is None:
            if self._provider == "CPUExecutionProvider":
                self._impl = MaskRCNNOnnxCPU(self._model_dir, providers=[self._provider])
            else:
                self._impl = MaskRCNNQaic(self._model_dir, providers=[self._provider], device_id=self._device_id)
        return self._impl

    def predict(
        self,
        image_path: str,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk: int = 100,
        output_path: Optional[str] = None,
    ) -> dict:
        """Canonical single-run result: {"prediction": {...instances...}, "latency_ms": float}.

        Draws the instances (masks/labels) to output_path when given, via the
        recipe's visualize(). Use predict_benchmark() for per-stage latency stats.
        """
        impl = self._ensure_impl()
        impl.post_cfg = PostprocessConfig(score_thresh=score_thresh, nms_thresh=nms_thresh, topk=topk)
        return impl.predict(image_path, output_path)

    def predict_benchmark(self, image_path: str, **kwargs) -> PredictResult:
        """Per-stage latency statistics (warmup/runs/score_thresh/nms_thresh/topk)."""
        return self._ensure_impl().predict_benchmark(image_path, **kwargs)
