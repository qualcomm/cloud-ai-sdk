# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Run the generated layout -> det -> batched-rec OCR pipeline on an image or PDF.

Standalone script layered on the installed ``qaic_models`` package — run it
by path (``python demo_pages.py --image document.pdf --model build/paddle_ocr``)
or as a module (``python -m qaic_models.vision.ocr.paddle_ocr.demo_pages``).

The multi-page pipeline: the :class:`PageScheduler` streams every page through
independent layout, det, and rec queues. On Cloud AI 100 it stripes layout
regions over independent detector QPC sessions on ``--det-devices``, then fans
recognition batches out over ``--rec-sessions`` concurrent QPC sessions spread
across ``--rec-devices``; the CPU reference uses one session per stage and
runs the batches sequentially. For the single-image API (one layout session,
one serial detector, four rec sessions), see :mod:`pipeline`; :mod:`demo` is
its CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from qaic_models.vision.ocr.paddle_ocr import postprocess, preprocess, visualize
from qaic_models.vision.ocr.paddle_ocr.model import (
    PaddleOCRLayoutOnnxCPU,
    PaddleOCRLayoutQaic,
    PaddleOCRDetOnnxCPU,
    PaddleOCRDetQaic,
    PaddleOCRRecOnnxCPU,
    PaddleOCRRecQaic,
    REC_BATCH,
)
from qaic_models.vision.ocr.paddle_ocr.workers import _WorkerPool


class TraceRecorder:
    """Emit timestamped JSON-lines for one profiled pipeline pass."""

    def __init__(self, stream=None):
        self._start_ns = time.perf_counter_ns()
        self._stream = stream if stream is not None else sys.stderr
        self._lock = threading.Lock()

    def record(self, event: str, *, page: int | None = None, **fields) -> None:
        record = {
            "time_ms": (time.perf_counter_ns() - self._start_ns) / 1_000_000.0,
            "event": event,
        }
        if page is not None:
            record["page"] = page
        record.update(fields)
        with self._lock:
            print(json.dumps(record, sort_keys=True), file=self._stream, flush=True)


@dataclass
class _PageState:
    page_index: int
    page_number: int
    image: object
    layout_result: tuple | None = None
    layout_regions: list = field(default_factory=list)
    ocr_regions: list = field(default_factory=list)
    placeholders: list = field(default_factory=list)
    det_results: list | None = None
    det_remaining: int = 0
    det_start: float | None = None
    det_end: float | None = None
    kept_pairs: list = field(default_factory=list)
    kept_regions: list = field(default_factory=list)
    crops: list = field(default_factory=list)
    rec_results: list | None = None
    rec_remaining: int = 0
    rec_start: float | None = None
    rec_end: float | None = None
    prep_ms: float = 0.0
    timing: dict | None = None


class LayoutWorker:
    """Owns the single layout session: its warm pass, the timed per-page run,
    and the layout QPC trace events."""

    def __init__(self, predictor, index=0):
        self.predictor = predictor
        self.index = index

    def warm(self, image):
        self.predictor.run_image(image)

    def run(self, image, page_number, trace=None):
        start = time.perf_counter()
        if trace is not None:
            trace.record("layout_start", page=page_number)
            trace.record(
                "qpc_start", page=page_number, stage="layout", qpc_index=self.index,
                device_id=getattr(self.predictor, "device_id", None),
            )
        try:
            prediction = self.predictor.run_image(image)
        finally:
            if trace is not None:
                trace.record(
                    "qpc_end", page=page_number, stage="layout", qpc_index=self.index,
                    device_id=getattr(self.predictor, "device_id", None),
                )
                trace.record("layout_end", page=page_number)
        return prediction, (time.perf_counter() - start) * 1000.0


class DetWorker:
    """Owns one detector session: it crops a layout region, runs the det QPC
    pass with its trace events, and returns the region's detections in
    reading order, offset into image coordinates."""

    def __init__(self, predictor, index):
        self.predictor = predictor
        self.index = index

    def warm(self, image):
        self.predictor.run_image(image)

    def run(self, image, region, *, region_index=None, page_number=None, trace=None):
        region_crop = preprocess.crop_box(image, region["box"])
        if region_crop is None:
            return []
        if trace is not None:
            trace.record(
                "qpc_start", page=page_number, stage="det", qpc_index=self.index,
                device_id=getattr(self.predictor, "device_id", None),
                region_index=region_index,
            )
        try:
            boxes, scores = self.predictor.run_image(region_crop)
        finally:
            if trace is not None:
                trace.record(
                    "qpc_end", page=page_number, stage="det", qpc_index=self.index,
                    device_id=getattr(self.predictor, "device_id", None),
                    region_index=region_index,
                )
        x0, y0 = region["box"][:2]
        return [
            ([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0], score, region)
            for box, score in postprocess.reading_order(boxes, scores)
        ]


class RecWorker:
    """Owns one recognizer session: the rec QPC trace events around one
    ``run_batch`` over a group of line crops."""

    def __init__(self, predictor, index):
        self.predictor = predictor
        self.index = index

    def warm(self, crops):
        self.predictor.run_batch(crops)

    def run(self, group, *, batch_index=None, page_number=None, trace=None):
        if trace is not None:
            trace.record(
                "qpc_start", page=page_number, stage="rec", qpc_index=self.index,
                device_id=getattr(self.predictor, "device_id", None),
                batch_index=batch_index, batch_size=len(group),
            )
        try:
            return self.predictor.run_batch(group)
        finally:
            if trace is not None:
                trace.record(
                    "qpc_end", page=page_number, stage="rec", qpc_index=self.index,
                    device_id=getattr(self.predictor, "device_id", None),
                    batch_index=batch_index, batch_size=len(group),
                )


class PageScheduler:
    """Stream pages through independent layout, det, and rec queues.

    Takes one :class:`LayoutWorker` and det/rec worker lists; it owns only
    the page state transitions and the ordered collection of worker
    futures. Worker selection and the per-session serial queues live in
    :class:`_WorkerPool`.
    """

    def __init__(self, layout, det_workers, rec_workers):
        self.layout = layout
        self.det_workers = det_workers
        self.rec_workers = rec_workers

    def run(self, pages, *, profile, trace=None):
        pending = {}
        results = {}
        states = {
            page_index: _PageState(page_index, page_number, image)
            for page_index, (page_number, image) in enumerate(pages)
        }
        if trace is not None:
            trace.record("scheduler_start")
        with ExitStack() as stack:
            layout_pool = stack.enter_context(_WorkerPool([self.layout]))
            det_pool = stack.enter_context(_WorkerPool(self.det_workers))
            rec_pool = stack.enter_context(_WorkerPool(self.rec_workers))
            for page_index, (page_number, image) in enumerate(pages):
                if trace is not None:
                    trace.record("layout_queued", page=page_number)
                future = layout_pool.submit(image, page_number, trace)
                pending[future] = ("layout", page_index, None)

            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                completed = sorted(completed, key=lambda future: pending[future][1])
                for future in completed:
                    kind, page_index, slot = pending.pop(future)
                    state = states[page_index]
                    if kind == "layout":
                        self._start_page(
                            state, future.result(), pending, det_pool, rec_pool,
                            results, trace, profile,
                        )
                    elif kind == "det":
                        state.det_results[slot] = future.result()
                        state.det_remaining -= 1
                        if state.det_remaining == 0:
                            self._start_recognition(state, pending, rec_pool, results, trace)
                    else:
                        state.rec_results[slot] = future.result()
                        state.rec_remaining -= 1
                        if state.rec_remaining == 0:
                            results[page_index] = self._finish_page(state, trace)
        if trace is not None:
            trace.record("scheduler_end")
        return [
            (page_number, image, results[page_index])
            for page_index, (page_number, image) in enumerate(pages)
        ]

    def _start_page(
        self, state, layout_result, pending, det_pool, rec_pool, results, trace, profile,
    ):
        state.layout_result = layout_result
        layout_pred, _ = layout_result
        state.layout_regions, state.ocr_regions, state.placeholders = postprocess.split_layout_regions(
            layout_pred,
        )
        state.det_results = [None] * len(state.ocr_regions)
        state.timing = {} if profile else None
        if trace is not None:
            trace.record("page_queued", page=state.page_number)
            trace.record("page_start", page=state.page_number)
            trace.record("det_start", page=state.page_number)
        state.det_start = time.perf_counter()
        if not state.ocr_regions:
            state.det_end = state.det_start
            if trace is not None:
                trace.record("det_end", page=state.page_number)
            self._start_recognition(state, pending, rec_pool, results, trace)
            return

        for region_index, region in enumerate(state.ocr_regions):
            future = det_pool.submit(
                state.image,
                region,
                region_index=region_index,
                page_number=state.page_number,
                trace=trace,
            )
            pending[future] = ("det", state.page_index, region_index)
            state.det_remaining += 1

    def _start_recognition(self, state, pending, rec_pool, results, trace):
        if state.det_end is None:
            state.det_end = time.perf_counter()
            if trace is not None:
                trace.record("det_end", page=state.page_number)

        detections = [
            item
            for region_results in state.det_results or []
            for item in region_results
        ]
        if trace is not None:
            trace.record("prep_start", page=state.page_number)
        prep_start = time.perf_counter()
        for box, score, region in detections:
            crop = preprocess.crop_box(state.image, box)
            if crop is None:
                continue
            state.kept_pairs.append((box, score))
            state.kept_regions.append(region)
            state.crops.append(crop)
        prep_end = time.perf_counter()
        if trace is not None:
            trace.record("prep_end", page=state.page_number)
        state.prep_ms = (prep_end - prep_start) * 1000.0

        groups = [
            state.crops[i : i + REC_BATCH]
            for i in range(0, len(state.crops), REC_BATCH)
        ]
        state.rec_results = [None] * len(groups)
        state.rec_remaining = len(groups)
        if not groups:
            state.rec_start = state.rec_end = time.perf_counter()
            results[state.page_index] = self._finish_page(state, trace)
            return

        if trace is not None:
            trace.record("rec_start", page=state.page_number)
        state.rec_start = time.perf_counter()
        for batch_index, group in enumerate(groups):
            future = rec_pool.submit(
                group,
                page_number=state.page_number,
                batch_index=batch_index,
                trace=trace,
            )
            pending[future] = ("rec", state.page_index, batch_index)

    @staticmethod
    def _finish_page(state, trace):
        state.rec_end = state.rec_end or time.perf_counter()
        if trace is not None:
            trace.record("rec_end", page=state.page_number)
        if state.timing is not None:
            state.timing.update(
                layout_ms=state.layout_result[1] if state.layout_result else 0.0,
                det_ms=(state.det_end - state.det_start) * 1000.0,
                prep_ms=state.prep_ms,
                rec_ms=(state.rec_end - state.rec_start) * 1000.0,
            )
        decodes = [
            decode
            for group_result in state.rec_results or []
            for decode in group_result
        ]
        prediction = postprocess.build_prediction(
            state.layout_regions,
            state.placeholders,
            state.kept_pairs,
            state.kept_regions,
            decodes,
        )
        if trace is not None:
            trace.record("page_end", page=state.page_number)
        return prediction, state.timing


def _page_output_path(out_path: str | None, page_number: int, multiple: bool) -> str | None:
    if out_path is None:
        return None
    if not multiple:
        return out_path
    path = Path(out_path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return str(path.with_name(f"{path.stem}_page_{page_number:04d}{path.suffix}"))
    path.mkdir(parents=True, exist_ok=True)
    return str(path / f"page_{page_number:04d}.jpg")


class PipelineResources:
    """Create, validate, and warm the layout/det/rec stage workers: one
    layout worker, one det worker per detector device, and rec workers
    spread evenly over the rec devices."""

    def __init__(
        self,
        model_dir,
        provider,
        *,
        layout_device_id=0,
        device_id=None,
        det_devices=(1, 2),
        rec_devices=(3,),
        rec_sessions=4,
    ):
        if provider == "CPUExecutionProvider":
            self.layout = LayoutWorker(PaddleOCRLayoutOnnxCPU(model_dir, providers=[provider]))
            self.dets = [DetWorker(PaddleOCRDetOnnxCPU(model_dir, providers=[provider]), 0)]
            self.recs = [RecWorker(PaddleOCRRecOnnxCPU(model_dir, providers=[provider]), 0)]
            return
        if provider != "QAicExecutionProvider":
            raise ValueError(f"Unsupported provider: {provider}")
        self.layout = LayoutWorker(PaddleOCRLayoutQaic(model_dir, device_id=layout_device_id))
        if device_id is not None:
            det_devices = (device_id,)
        if not det_devices:
            raise ValueError("det_devices must be non-empty")
        if len(set(det_devices)) != len(det_devices):
            raise ValueError(f"det_devices must not contain duplicates: {det_devices}")
        self.dets = [
            DetWorker(PaddleOCRDetQaic(model_dir, device_id=device), index)
            for index, device in enumerate(det_devices)
        ]
        if not rec_devices or rec_sessions < len(rec_devices):
            raise ValueError(
                f"rec_devices must be non-empty and rec_sessions ({rec_sessions}) >= "
                f"the number of devices ({len(rec_devices)})"
            )
        rec_device_list = [
            device
            for device in rec_devices
            for _ in range(rec_sessions // len(rec_devices))
        ]
        self.recs = [
            RecWorker(PaddleOCRRecQaic(model_dir, device_id=device), index)
            for index, device in enumerate(rec_device_list)
        ]

    def warm(self, img_bgr):
        """Warm every stage session with one untimed pass before the timed
        run.

        The first inference on a QPC session pays one-time device setup, so
        the timed pass must never be the first one. The rec sessions warm on
        a small corner crop (line content is irrelevant to the warmup).
        """
        self.layout.warm(img_bgr)
        for det in self.dets:
            det.warm(img_bgr)
        warmup_crop = preprocess.crop_box(
            img_bgr, [0, 0, min(img_bgr.shape[1], 100), min(img_bgr.shape[0], 30)]
        )
        for rec in self.recs:
            rec.warm([warmup_crop])


class DocumentPipeline:
    """Run the timed pipeline pass over the pages and format the document
    result: warm the sessions, stream the pages through the
    :class:`PageScheduler`, write the annotated outputs, and build the
    per-page prediction/timing payload."""

    def __init__(self, pages, resources, *, out_path=None, profile=False, pdf=False):
        self.pages = pages
        self.resources = resources
        self.out_path = out_path
        self.profile = profile
        self.pdf = pdf

    def run(self):
        resources = self.resources
        resources.warm(self.pages[0][1])
        trace = TraceRecorder() if self.profile else None
        if trace is not None:
            trace.record("pipeline_start")
        start = time.perf_counter()
        page_results = PageScheduler(
            resources.layout, resources.dets, resources.recs,
        ).run(self.pages, profile=self.profile, trace=trace)
        predictions = []
        timings = []
        output_paths = []
        for page_number, image, (prediction, timing) in page_results:
            predictions.append((page_number, prediction))
            if timing is not None:
                timings.append((page_number, timing))
            page_output = _page_output_path(
                self.out_path, page_number, self.pdf or len(self.pages) > 1,
            )
            if page_output:
                if trace is not None:
                    trace.record("output_start", page=page_number)
                visualize.visualize_rec(
                    image, prediction, page_output,
                    resources.dets[0].predictor.get_font,
                )
                if trace is not None:
                    trace.record("output_end", page=page_number)
                output_paths.append(page_output)

        latency_ms = (time.perf_counter() - start) * 1000.0
        if trace is not None:
            trace.record("pipeline_end")
        if len(predictions) == 1 and not self.pdf:
            result = {"prediction": predictions[0][1], "latency_ms": latency_ms}
            if self.profile:
                result["timing_ms"] = timings[0][1]
        else:
            result = {
                "prediction": {"pages": [{"page": number, **pred} for number, pred in predictions]},
                "latency_ms": latency_ms,
            }
            if self.profile:
                result["timing_ms"] = {
                    "pages": [{"page": number, **timing} for number, timing in timings],
                    "total_ms": latency_ms,
                }
        if output_paths:
            result["output_paths"] = output_paths
        return result


def run_pipeline(
    image_path: str,
    model_dir: str,
    provider: str = "QAicExecutionProvider",
    *,
    layout_device_id: int = 0,
    device_id: int | None = None,
    det_devices: tuple[int, ...] = (1, 2),
    rec_devices: tuple[int, ...] = (3,),
    rec_sessions: int = 4,
    out_path: str | None = None,
    profile: bool = False,
) -> dict:
    """Run the layout -> det -> crop -> rec pipeline on a document image.

    Returns ``{"prediction": {"texts": [...], "regions": [...]},
    "latency_ms": float}`` — the same result schema the CLI prints. With
    ``profile=True`` the result also carries ``"timing_ms"`` with the layout
    / det / crop+prep / rec split of the timed pass. One untimed warm pass
    per stage session runs first (the first QPC inference pays one-time
    device setup), then one timed pass. The default QAIC placement uses
    layout SoC 0, detector SoCs 1 and 2, and four 4-core rec sessions on
    SoC 3.
    """
    if provider not in {"CPUExecutionProvider", "QAicExecutionProvider"}:
        raise ValueError(f"Unsupported provider: {provider}")
    img_bgr = preprocess._load_bgr(image_path)
    resources = PipelineResources(
        model_dir, provider,
        layout_device_id=layout_device_id, device_id=device_id,
        det_devices=det_devices, rec_devices=rec_devices,
        rec_sessions=rec_sessions,
    )
    return DocumentPipeline(
        [(1, img_bgr)], resources,
        out_path=out_path, profile=profile, pdf=False,
    ).run()


def run_document(
    input_path: str,
    model_dir: str,
    provider: str = "QAicExecutionProvider",
    *,
    page: int | None = None,
    layout_device_id: int = 0,
    device_id: int | None = None,
    det_devices: tuple[int, ...] = (1, 2),
    rec_devices: tuple[int, ...] = (3,),
    rec_sessions: int = 4,
    out_path: str | None = None,
    profile: bool = False,
) -> dict:
    """Run OCR on an image or every page of a PDF."""
    if provider not in {"CPUExecutionProvider", "QAicExecutionProvider"}:
        raise ValueError(f"Unsupported provider: {provider}")
    pages = preprocess.load_bgr_pages(input_path, page=page)
    resources = PipelineResources(
        model_dir, provider,
        layout_device_id=layout_device_id, device_id=device_id,
        det_devices=det_devices, rec_devices=rec_devices,
        rec_sessions=rec_sessions,
    )
    return DocumentPipeline(
        pages, resources,
        out_path=out_path, profile=profile,
        pdf=str(input_path).lower().endswith(".pdf"),
    ).run()


def run_layout_document(
    input_path: str,
    model_dir: str,
    provider: str = "QAicExecutionProvider",
    *,
    page: int | None = None,
    layout_device_id: int = 0,
    out_path: str | None = None,
    profile: bool = False,
) -> dict:
    """Run layout-only inference on an image or PDF pages."""
    pages = preprocess.load_bgr_pages(input_path, page=page)
    if provider == "CPUExecutionProvider":
        layout = LayoutWorker(PaddleOCRLayoutOnnxCPU(model_dir, providers=[provider]))
    elif provider == "QAicExecutionProvider":
        layout = LayoutWorker(PaddleOCRLayoutQaic(model_dir, device_id=layout_device_id))
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    layout.warm(pages[0][1])
    start = time.perf_counter()
    predictions = []
    timings = []
    output_paths = []
    pdf = str(input_path).lower().endswith(".pdf")
    for page_number, image in pages:
        page_start = time.perf_counter()
        prediction = layout.run(image, page_number)[0]
        predictions.append((page_number, prediction))
        if profile:
            timings.append((page_number, (time.perf_counter() - page_start) * 1000.0))
        page_output = _page_output_path(out_path, page_number, pdf or len(pages) > 1)
        if page_output:
            visualize.visualize_layout(image, prediction, page_output, layout.predictor.get_font)
            output_paths.append(page_output)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if len(predictions) == 1 and not pdf:
        result = {"prediction": predictions[0][1], "latency_ms": latency_ms}
        if profile:
            result["timing_ms"] = {"layout_ms": timings[0][1]}
    else:
        result = {
            "prediction": {"pages": [{"page": number, **pred} for number, pred in predictions]},
            "latency_ms": latency_ms,
        }
        if profile:
            result["timing_ms"] = {
                "pages": [{"page": number, "layout_ms": elapsed} for number, elapsed in timings],
                "total_ms": latency_ms,
            }
    if output_paths:
        result["output_paths"] = output_paths
    return result


def _write_json_out(path: str, payload: dict) -> None:
    """Write the prediction payload the final stdout line carries to a file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    print(f"wrote: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the PP-DocLayoutV3 -> PP-OCRv6 OCR pipeline on an image or PDF "
        "(multi-page scheduler; see qaic_models.vision.ocr.paddle_ocr.demo for the single-image example)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Document image to scan for text")
    source.add_argument("--pdf", help="PDF document to scan for text")
    parser.add_argument("--model", required=True, help="Work directory produced by export.py")
    parser.add_argument(
        "--provider", default="QAicExecutionProvider",
        choices=["QAicExecutionProvider", "CPUExecutionProvider"],
        help="Run the compiled QPCs or the CPU ONNX reference",
    )
    parser.add_argument(
        "--layout", action="store_true",
        help="Run only PP-DocLayoutV3 and return document regions",
    )
    parser.add_argument("--layout-device", type=int, default=0, help="Device for the layout session (QAIC provider only)")
    parser.add_argument(
        "--det-devices", default="1,2",
        help="Comma-separated devices (SoCs) for concurrent det sessions (QAIC provider only)",
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Legacy single det device override; supersedes --det-devices",
    )
    parser.add_argument(
        "--rec-devices", default="3",
        help="Comma-separated devices (SoCs) for the rec sessions (QAIC provider only)",
    )
    parser.add_argument(
        "--rec-sessions", type=int, default=4,
        help="Concurrent rec QPC sessions, spread across the rec devices (QAIC provider only)",
    )
    parser.add_argument(
        "--page", type=int, default=None,
        help="One-based PDF page to process; default processes every page",
    )
    parser.add_argument("--out", default=None, help="Optional annotated JPEG output path")
    parser.add_argument("--json-out", default=None, help="Optional path to write the prediction JSON payload (same object as the final stdout line)")
    parser.add_argument(
        "--profile", action="store_true",
        help="Print stage timings and emit a timestamped execution trace",
    )
    args = parser.parse_args(argv)
    input_path = args.pdf if args.pdf is not None else args.image
    if input_path.lower().endswith(".pdf") and args.pdf is None:
        parser.error("--image takes a single image; pass the file with --pdf for PDF input")
    is_pdf = args.pdf is not None
    if args.page is not None and not is_pdf:
        parser.error("--page is only valid with --pdf")

    if args.layout:
        # An image is a single page, so the document runner covers both input
        # types (main() above already rejects --page for non-PDF input).
        result = run_layout_document(
            input_path, args.model, provider=args.provider, page=args.page,
            layout_device_id=args.layout_device, out_path=args.out, profile=args.profile,
        )
        print(f"latency_ms: {result['latency_ms']:.3f}")
        if "timing_ms" in result and "layout_ms" in result["timing_ms"]:
            print(f"layout_ms: {result['timing_ms']['layout_ms']:.1f}")
        if "pages" in result["prediction"]:
            print(f"pages: {len(result['prediction']['pages'])}")
            if "timing_ms" in result:
                for page_timing in result["timing_ms"]["pages"]:
                    print(f"page {page_timing['page']}: layout_ms: {page_timing['layout_ms']:.1f}")
        else:
            print(f"regions: {len(result['prediction']['regions'])}")
        output_paths = result.get("output_paths", [args.out] if args.out else [])
        for output_path in output_paths:
            print(f"wrote: {output_path}")
        if args.json_out:
            _write_json_out(args.json_out, result["prediction"])
        print(json.dumps(result["prediction"], ensure_ascii=False, sort_keys=True))
        return

    det_devices = tuple(int(d) for d in args.det_devices.split(",") if d.strip())
    rec_devices = tuple(int(d) for d in args.rec_devices.split(",") if d.strip())
    runner = run_document if is_pdf else run_pipeline
    runner_args = dict(
        provider=args.provider, layout_device_id=args.layout_device,
        device_id=args.device, det_devices=det_devices,
        rec_devices=rec_devices, rec_sessions=args.rec_sessions,
        out_path=args.out, profile=args.profile,
    )
    if is_pdf:
        runner_args["page"] = args.page
    result = runner(input_path, args.model, **runner_args)
    prediction = result["prediction"]
    # stdout layout: summary, then the detected text (one line per region,
    # top to bottom), then the full JSON prediction on the final line so
    # line-based consumers can still parse it.
    print(f"latency_ms: {result['latency_ms']:.3f}")
    if "timing_ms" in result and "pages" not in prediction:
        timing = result["timing_ms"]
        print(
            f"layout_ms: {timing['layout_ms']:.1f}  "
            f"det_ms: {timing['det_ms']:.1f}  "
            f"prep_ms: {timing['prep_ms']:.1f}  "
            f"rec_ms: {timing['rec_ms']:.1f}"
        )
    if "pages" in prediction:
        print(f"pages: {len(prediction['pages'])}")
        for page in prediction["pages"]:
            print(f"page {page['page']}: regions: {len(page['regions'])}  lines: {len(page['texts'])}")
            for entry in page["texts"]:
                print(entry["text"])
        if "timing_ms" in result:
            for page_timing in result["timing_ms"]["pages"]:
                print(
                    f"page {page_timing['page']}: layout_ms: {page_timing['layout_ms']:.1f}  "
                    f"det_ms: {page_timing['det_ms']:.1f}  prep_ms: {page_timing['prep_ms']:.1f}  "
                    f"rec_ms: {page_timing['rec_ms']:.1f}"
                )
    else:
        texts = prediction["texts"]
        print(f"regions: {len(prediction['regions'])}")
        print(f"lines: {len(texts)}")
        for entry in texts:
            print(entry["text"])
    output_paths = result.get("output_paths", [args.out] if args.out else [])
    for output_path in output_paths:
        print(f"wrote: {output_path}")
    if args.json_out:
        _write_json_out(args.json_out, prediction)
    if "pages" not in prediction and prediction["texts"]:
        print()
        print()
    print(json.dumps(result["prediction"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
