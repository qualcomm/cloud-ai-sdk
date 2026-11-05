#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""predict_base.py — shared predictor bases for Cloud AI 100 model recipes.

Standalone onboarding workspaces: copy this file to /workspace/<model>/ alongside predict.py.
Shared-package use (e.g. qaic_models.base.predict_base): import this one canonical copy from
every recipe instead of copying it per-model — QAICPredictor._resolve_qpc() anchors to the
subclass's own defining module, not to this file, so both deployment shapes resolve QPC
paths correctly with no further changes.

Use QAICPredictor for compiled QPC inference via qaic.Session. Use CPUPredictor for CPU
reference predictors; its default backend is ONNX Runtime CPU, but the public abstraction
is CPU reference prediction rather than a specific runtime.
"""

from __future__ import annotations

import argparse, base64, hashlib, json, os, sys, time, urllib.request
import numpy as np

from qaic_models.base.workspace import qaic_work_path


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class PredictorBase:
    """Common prediction lifecycle for packaged demo predictors."""

    LABELS_URL = ""   # subclass must set
    LABELS_TMP = ""   # subclass must set

    _FONT_CACHE: dict = {}
    _CJK_FONT_CACHE: dict = {}
    _FONT_URL     = "https://cdn.jsdelivr.net/npm/dejavu-fonts-ttf@2.37.3/ttf/DejaVuSans-Bold.ttf"
    _FONT_NAME    = "DejaVuSans-Bold.ttf"
    _CJK_FONT_URL = "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf"
    _CJK_FONT_NAME = "NotoSansSC-Regular.otf"


    @staticmethod
    def _has_cjk(text: str) -> bool:
        # U+3000–U+9FFF: CJK punctuation, Hiragana, Katakana, CJK Unified Ideographs
        # U+F900–U+FAFF: CJK Compatibility Ideographs
        # Copy these two ranges exactly. Do NOT add any other ranges — in particular,
        # U+0020–U+007E is ASCII and would make this return True for every English label.
        return any('　' <= c <= '鿿' or '豈' <= c <= '﫿' for c in text)

    @classmethod
    def get_font(cls, size: int, text: str = ""):
        """Return a font at `size`. Uses a CJK-capable font if text contains CJK characters."""
        from PIL import ImageFont
        if cls._has_cjk(text):
            cache, url = cls._CJK_FONT_CACHE, cls._CJK_FONT_URL
            tmp = str(qaic_work_path("fonts", cls._CJK_FONT_NAME))
            sys_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                tmp,
            ]
        else:
            cache, url = cls._FONT_CACHE, cls._FONT_URL
            tmp = str(qaic_work_path("fonts", cls._FONT_NAME))
            sys_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", tmp]
        if size in cache:
            return cache[size]
        for path in sys_paths:
            if os.path.exists(path):
                try:
                    f = ImageFont.truetype(path, size)
                    cache[size] = f
                    return f
                except Exception:
                    pass
        try:
            urllib.request.urlretrieve(url, tmp)
            f = ImageFont.truetype(tmp, size)
            cache[size] = f
            return f
        except Exception as e:
            sys.stderr.write(f"get_font: download failed ({e}), using default\n")
            return ImageFont.load_default()

    def _get_labels(self) -> list:
        """Return label list. LABELS_URL accepts sentinels 'COCO_80' or 'COCO_91',
        or a real URL (downloaded under ``QAIC_WORK_ROOT``). Returns [] on failure."""
        if self.LABELS_URL == "COCO_80":
            return COCO_80_LABELS
        if self.LABELS_URL == "COCO_91":
            return COCO_91_LABELS
        if not self.LABELS_URL or not self.LABELS_TMP:
            return []
        labels_path = self.LABELS_TMP
        if not os.path.isabs(labels_path):
            labels_path = str(qaic_work_path("labels", labels_path))
        try:
            if not os.path.exists(labels_path):
                urllib.request.urlretrieve(self.LABELS_URL, labels_path)
            with open(labels_path) as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            sys.stderr.write(f"_get_labels: failed ({e}), using numeric indices\n")
            return []

    @staticmethod
    def _nms(x1, y1, x2, y2, scores, iou_thresh: float, classes=None) -> np.ndarray:
        """Greedy NMS. Per-class if classes array provided, class-agnostic otherwise."""
        keep = []
        groups = [np.where(classes == c)[0] for c in np.unique(classes)] \
            if classes is not None else [np.arange(len(scores))]
        for idx in groups:
            idx = idx[np.argsort(scores[idx])[::-1]]
            while len(idx) > 0:
                i = idx[0]; keep.append(i)
                if len(idx) == 1: break
                rest = idx[1:]
                ix1 = np.maximum(x1[i], x1[rest]); iy1 = np.maximum(y1[i], y1[rest])
                ix2 = np.minimum(x2[i], x2[rest]); iy2 = np.minimum(y2[i], y2[rest])
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                union = (x2[i]-x1[i])*(y2[i]-y1[i]) + (x2[rest]-x1[rest])*(y2[rest]-y1[rest]) - inter
                idx = rest[inter / (union + 1e-6) < iou_thresh]
        return np.array(keep, dtype=np.intp)


    @staticmethod
    def _resolve_image(src: str) -> str:
        """Resolve --image value to a local file path.

        Accepts:
          - file path         (returned as-is)
          - http/https URL    (downloaded under QAIC_WORK_ROOT, cached by URL hash)
          - raw base64 string (decoded under QAIC_WORK_ROOT; Pillow detects format from magic bytes)
        """
        def _work_path(tag: str) -> str:
            key = hashlib.md5(tag.encode()).hexdigest()[:12]
            return str(qaic_work_path("inputs", f"qaic_img_{key}"))

        if src.startswith(("http://", "https://")):
            path = _work_path(src)
            if not os.path.exists(path):
                urllib.request.urlretrieve(src, path)
            return path

        if not os.path.exists(src):
            try:
                data = base64.b64decode(src + "==")
                path = _work_path(src[:64])
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(data)
                return path
            except Exception:
                pass

        return src

    @staticmethod
    def _resolve_audio(src: str) -> str:
        """Resolve --audio value to a local file path.

        Accepts:
          - file path         (returned as-is)
          - http/https URL    (downloaded under QAIC_WORK_ROOT, cached by URL hash)
          - raw base64 string (decoded under QAIC_WORK_ROOT; WAV/MP3/FLAC/OGG all work)
        """
        def _work_path(tag: str) -> str:
            key = hashlib.md5(tag.encode()).hexdigest()[:12]
            return str(qaic_work_path("inputs", f"qaic_audio_{key}"))

        if src.startswith(("http://", "https://")):
            path = _work_path(src)
            if not os.path.exists(path):
                urllib.request.urlretrieve(src, path)
            return path

        if not os.path.exists(src):
            try:
                data = base64.b64decode(src + "==")
                path = _work_path(src[:64])
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(data)
                return path
            except Exception:
                pass

        return src

    def parse_args(self):
        p = argparse.ArgumentParser()
        p.add_argument("--image",  default=None,
                       help="Image input: file path, http/https URL, or base64 string")
        p.add_argument("--output", default=None,
                       help="Path to output JPEG (image models; omit for text/audio models)")
        p.add_argument("--annotation", default=True, action=argparse.BooleanOptionalAction,
                       help="Write annotated output image (default: on; --no-annotation to disable)")
        p.add_argument("--text",  default=None, help="Input text string (text-input models)")
        p.add_argument("--audio", default=None,
                       help="Audio input: file path, http/https URL, or base64 string (WAV/MP3/FLAC/OGG)")
        args = p.parse_args()
        if args.text is None and args.image is None and args.audio is None:
            p.error("Provide --image (image model), --text (text model), or --audio (audio model)")
        return args

    def predict(self, image_path: str, output_path: str | None) -> dict:
        model_inputs = self.preprocess(image_path)
        if not self._warmed_up:
            for _ in range(3):
                self._forward(model_inputs)
            self._warmed_up = True
        t0 = time.perf_counter()
        model_outputs = self._forward(model_inputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = self.postprocess(model_outputs, latency_ms)
        if output_path is not None:
            self.visualize(image_path, result["prediction"], output_path)
        return result

    @staticmethod
    def _rasterize_pdf(pdf_path: str, dpi: int = 150, max_pages: int = 20) -> list:
        """Rasterize a PDF to a list of PIL Images (one per page, capped at max_pages).

        Requires pymupdf (fitz). Each page is rendered at `dpi` and returned as an
        RGB PIL Image. If the PDF has more than max_pages pages, only the first
        max_pages are processed and a warning is written to stderr.
        """
        import fitz  # pymupdf
        from PIL import Image
        doc = fitz.open(pdf_path)
        try:
            n = len(doc)
            if n > max_pages:
                sys.stderr.write(
                    f"_rasterize_pdf: {pdf_path} has {n} pages; processing first {max_pages} only\n"
                )
                n = max_pages
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pages = []
            for i in range(n):
                pix = doc[i].get_pixmap(matrix=mat, alpha=False)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        finally:
            doc.close()
        return pages

    def main(self):
        args = self.parse_args()
        if args.text is not None:
            print(json.dumps(self.predict_text(args.text)))
        elif args.audio is not None:
            audio_path = self._resolve_audio(args.audio)
            if not os.path.exists(audio_path):
                sys.stderr.write(f"Input not found: {args.audio}\n")
                sys.exit(1)
            print(json.dumps(self.predict_audio(audio_path)))
        elif args.image is not None and args.image.lower().endswith(".pdf"):
            pages = self._rasterize_pdf(args.image)
            results = []
            total_ms = 0.0
            for i, page_img in enumerate(pages):
                tmp = str(qaic_work_path("pdf-pages", f"qaic_pdf_page_{i}.jpg"))
                out = str(qaic_work_path("pdf-pages", f"qaic_pdf_page_{i}_out.jpg"))
                try:
                    page_img.save(tmp, "JPEG")
                    r = self.predict(tmp, out)
                    result_image = None
                    out_path = out if os.path.exists(out) else None
                    if out_path:
                        with open(out_path, "rb") as f:
                            result_image = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
                    if result_image is None:
                        # no annotated output — embed the plain page image
                        with open(tmp, "rb") as f:
                            result_image = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
                finally:
                    for p in (tmp, out):
                        if os.path.exists(p):
                            os.unlink(p)
                results.append({"page": i + 1, "prediction": r["prediction"],
                                 "latency_ms": r["latency_ms"], "result_image": result_image})
                total_ms += r.get("latency_ms") or 0.0
            print(json.dumps({"prediction": {"type": "pdf_pages", "pages": results}, "latency_ms": total_ms}))
        else:
            image_path = self._resolve_image(args.image)
            if not os.path.exists(image_path):
                sys.stderr.write(f"Input not found: {args.image}\n")
                sys.exit(1)
            output_path = args.output if args.annotation else None
            print(json.dumps(self.predict(image_path, output_path)))

    def predict_text(self, text: str) -> dict:
        """Override for text-input models (NLP, text classification).
        Receives the raw text string. Must return a dict with 'prediction' and 'latency_ms' keys.
        """
        raise NotImplementedError("predict_text: override this method for text-input models")

    def predict_audio(self, audio_path: str) -> dict:
        """Override for audio-input models (ASR, audio classification, etc.).
        Receives the local file path (WAV/MP3/FLAC/OGG).
        Must return a dict with 'prediction' and 'latency_ms' keys.
        """
        raise NotImplementedError("predict_audio: override this method for audio-input models")

    def preprocess(self, image_path: str) -> dict:
        raise NotImplementedError

    def postprocess(self, outputs: dict, latency_ms: float) -> dict:
        raise NotImplementedError

    def visualize(self, image_path: str, prediction, output_path: str):
        raise NotImplementedError


class QAICPredictor(PredictorBase):
    """Base class for Cloud AI 100 QPC inference via qaic.Session."""

    def _resolve_qpc(self, qpc_dir: str) -> str:
        """Return an absolute programqpc.bin path for qpc_dir.

        Anchors a relative qpc_dir against the directory of the module that DEFINES
        the concrete ModelPredictor subclass — not against predict_base.py's own
        file. In a standalone onboarding workspace, the subclass lives in predict.py,
        which sits next to the model's QPC directory, so this matches today's
        behavior exactly. When predict_base.py is imported once from a shared
        package location (e.g. qaic_models.base.predict_base) and reused by many
        recipes, each recipe's subclass is defined in its own model.py, which is
        co-located with that recipe's QPC directory — so the same anchoring logic
        resolves correctly there too, with no per-repo branching needed.
        Absolute qpc_dir values are returned as-is (never anchored).
        """
        if os.path.isabs(qpc_dir):
            return os.path.join(qpc_dir, "programqpc.bin")
        module = sys.modules.get(type(self).__module__)
        module_file = getattr(module, "__file__", None) if module else None
        base = os.path.dirname(os.path.abspath(module_file)) if module_file else os.getcwd()
        return os.path.join(base, qpc_dir, "programqpc.bin")

    def __init__(self, qpc_dir: str, device_id: int | None = None):
        import qaic  # noqa: PLC0415
        kwargs = {"model_path": self._resolve_qpc(qpc_dir), "num_activations": 1}
        if device_id is not None:
            kwargs["dev_id"] = device_id
        self._session = qaic.Session(**kwargs)
        self._warmed_up = False

    def _forward(self, model_inputs: dict) -> dict:
        """Run one QAIC inference pass; return raw outputs dict.

        Single-stage default: passes model_inputs directly to self._session.
        Multi-stage subclasses override to orchestrate named sessions
        (self.encoder, self.decoder, etc.).
        """
        contiguous = {k: np.ascontiguousarray(v) for k, v in model_inputs.items()}
        return self._session.run(contiguous)


class CPUPredictor(PredictorBase):
    """Base class for CPU reference predictors.

    The default backend is ONNX Runtime CPU. Subclasses may override
    _create_session() and _forward() for other CPU frameworks, but the public
    concept is CPU reference prediction, not ONNX Runtime specifically.
    """

    DEFAULT_PROVIDERS = ["CPUExecutionProvider"]

    def __init__(self, model_path: str, providers: list[str] | None = None):
        self.providers = providers or list(self.DEFAULT_PROVIDERS)
        self._session = self._create_session(model_path)
        self._warmed_up = False

    def _create_session(self, model_path: str):
        import onnxruntime as ort  # noqa: PLC0415
        return ort.InferenceSession(model_path, providers=self.providers)

    def _forward(self, model_inputs: dict):
        """Run one CPU reference inference pass; return raw outputs."""
        contiguous = {k: np.ascontiguousarray(v) for k, v in model_inputs.items()}
        return self._session.run(None, contiguous)


# ── COCO label lists ──────────────────────────────────────────────────────────

# COCO-80: 80-class dense list used by YOLO, SSD, EfficientDet, etc. (0-indexed).
COCO_80_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]

# COCO-91: sparse 91-entry list used by DETR and models trained on original COCO category IDs.
# Index = COCO category ID (1-based); index 0 is 'N/A' (background).
# Filter out label == 'N/A' when building detection results.
COCO_91_LABELS = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush',
]
