# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""PaddleOCR PP-OCRv6 input preparation (det letterbox + rec resize).

Both models share the input tensor name ``x`` (Paddle's default) and take NCHW
float32. The two stages use different, validated preparation:

* **det** — letterbox a document image to a fixed 640x640 canvas: aspect-preserving
  resize of the longest side to 640 (dims rounded to a multiple of 32, min 32),
  right+bottom white-255 pad, then PaddleOCR det normalization (BGR order):
  ``x/255`` then ``(x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]``. The
  letterbox metadata (source size, resized size, per-axis scale ratios) is returned so the
  postprocess can map canvas boxes back to original-image coordinates.

* **rec** — resize a pre-localized text line to height 48 (aspect preserved, width
  capped at 1536), right-pad white-255 to 1536, then PaddleOCR rec normalization
  ``(x/255 - 0.5) / 0.5``. The channel order is **cv2 BGR** to match the golden
  ``input.raw`` the QPC was compiled and validated against (the export's
  ``build_input_raw`` decodes with ``cv2.imread(..., IMREAD_COLOR)``).

Both are recipe-local: the base helpers use PIL/RGB and a different normalization,
so wrapping them would change the validated behavior.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Shared input tensor name (Paddle default) for both det and rec.
INPUT_NAME = "x"
LAYOUT_INPUT_NAME = "pixel_values"
LAYOUT_INPUT_SIZE = 704
# PP-DocLayoutV3's optional mask head produces a grid at one quarter of the
# input resolution.  The default layout export omits this output.
LAYOUT_MASK_SIZE = LAYOUT_INPUT_SIZE // 4

# ── Detection ─────────────────────────────────────────────────────────────────
DET_INPUT_H = 640
DET_INPUT_W = 640
DET_INPUT_SHAPE = (1, 3, DET_INPUT_H, DET_INPUT_W)
# PaddleOCR det normalization (BGR order) — identical to the export script so the
# demo input matches the exact tensor the QPC was validated against (input.raw).
_DET_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Recognition ───────────────────────────────────────────────────────────────
REC_INPUT_H = 48
REC_INPUT_W = 1536
REC_INPUT_SHAPE = (1, 3, REC_INPUT_H, REC_INPUT_W)
# PaddleOCR rec normalization: (x/255 - 0.5) / 0.5
REC_NORM_MEAN = 0.5
REC_NORM_STD = 0.5


def preprocess_layout_image(image_bgr: np.ndarray) -> np.ndarray:
    """Prepare the PP-DocLayoutV3 704x704 RGB stretch canvas."""
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = cv2.resize(rgb, (LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)


def _load_bgr(image_path: str) -> np.ndarray:
    """Decode an image with cv2 (BGR, HxW uint8). Raises FileNotFoundError if unreadable."""
    import cv2  # function-local: opencv-python-headless is an export/runtime extra.

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    return img


def load_bgr_pages(input_path: str, page: Optional[int] = None) -> list[Tuple[int, np.ndarray]]:
    """Load an image or render PDF pages as BGR arrays.

    ``page`` is one-based and only applies to PDFs. PDF rendering follows
    PaddleOCR's reference policy: render at 2x, but use 1x when either
    rendered dimension would exceed 2000 pixels.
    """
    path = str(input_path)
    if not path.lower().endswith(".pdf"):
        if page is not None:
            raise ValueError("--page is only valid for PDF input")
        return [(1, _load_bgr(path))]

    import cv2
    try:
        import pymupdf as pdf_api
    except ImportError:  # PyMuPDF < 1.24 exposed the module as ``fitz``.
        import fitz as pdf_api

    with pdf_api.open(path) as pdf:
        page_count = pdf.page_count
        if page is not None and not 1 <= page <= page_count:
            raise ValueError(f"PDF page must be between 1 and {page_count}, got {page}")
        page_indices = [page - 1] if page is not None else range(page_count)
        pages = []
        for page_index in page_indices:
            pdf_page = pdf[page_index]
            pixmap = pdf_page.get_pixmap(matrix=pdf_api.Matrix(2, 2), alpha=False)
            if pixmap.width > 2000 or pixmap.height > 2000:
                pixmap = pdf_page.get_pixmap(matrix=pdf_api.Matrix(1, 1), alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
            pages.append((page_index + 1, np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))))
    return pages


def letterbox_det(img_bgr: np.ndarray) -> Tuple[Dict[str, np.ndarray], dict]:
    """Letterbox a BGR document image to the fixed 640x640 det canvas.

    ``img_bgr`` is an HxW uint8 BGR array. Returns ``(inputs, meta)`` where
    ``inputs`` is ``{"x": NCHW float32 (1,3,640,640)}`` and ``meta`` carries the
    letterbox metadata (``src_h, src_w, resized_h, resized_w, ratio_h, ratio_w``) needed to
    undo the transform in the postprocess.
    """
    import cv2  # function-local

    src_h, src_w = img_bgr.shape[:2]
    limit_side = DET_INPUT_W
    ratio = limit_side / max(src_h, src_w)
    new_h = int(round(src_h * ratio))
    new_w = int(round(src_w * ratio))
    new_h = max(32, int(round(new_h / 32)) * 32)
    new_w = max(32, int(round(new_w / 32)) * 32)
    img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((DET_INPUT_H, DET_INPUT_W, 3), 255, dtype=np.uint8)
    canvas[:new_h, :new_w] = img
    x = canvas.astype(np.float32) * (1.0 / 255.0)
    x = (x - _DET_NORM_MEAN) / _DET_NORM_STD
    x = x.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    x = np.ascontiguousarray(x)
    meta = {
        "src_h": src_h,
        "src_w": src_w,
        "resized_h": new_h,
        "resized_w": new_w,
        # Rounding each resized dimension to a multiple of 32 can make these
        # differ from the nominal longest-side ratio.  Postprocess must restore
        # x and y with their actual, independent scales.
        "ratio_h": new_h / src_h,
        "ratio_w": new_w / src_w,
        # Retained for callers that only need the nominal resize policy.  It is
        # not valid for coordinate restoration after the 32-pixel rounding.
        "ratio": ratio,
    }
    return {INPUT_NAME: x}, meta


def preprocess_det_image(image_path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """Load (cv2 BGR) and letterbox a document image for the det stage.

    Returns ``(inputs, meta)`` — see :func:`letterbox_det`.
    """
    return letterbox_det(_load_bgr(image_path))


def crop_box(src_bgr: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """Crop a detected box (original-image coords) out of the source BGR image.

    ``box`` is ``[x0, y0, x1, y1]`` in original-image coordinates. Coordinates
    are rounded and clamped to the image bounds; returns the BGR uint8 crop, or
    ``None`` when the clamped box is empty.
    """
    src_h, src_w = src_bgr.shape[:2]
    x0 = max(0, min(int(round(box[0])), src_w))
    y0 = max(0, min(int(round(box[1])), src_h))
    x1 = max(0, min(int(round(box[2])), src_w))
    y1 = max(0, min(int(round(box[3])), src_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return src_bgr[y0:y1, x0:x1]


def preprocess_rec_image(image_bgr: np.ndarray) -> np.ndarray:
    """Prepare a BGR text-line crop for the rec stage.

    ``image_bgr`` is an HxW uint8 BGR array (a pre-localized text line). RecResizeImg:
    keep aspect ratio, resize to height 48 (cv2 INTER_LINEAR), cap width at 1536,
    right-pad white (255) to 1536. NormalizeImage: ``/255`` then ``(x - 0.5) / 0.5``,
    HWC -> NCHW. Returns a contiguous ``[1,3,48,1536]`` float32 array.
    """
    import cv2  # function-local

    h, w = image_bgr.shape[:2]
    ratio = w / h
    resize_w = int(math.ceil(ratio * REC_INPUT_H))
    if resize_w > REC_INPUT_W:
        resize_w = REC_INPUT_W
    resized = cv2.resize(image_bgr, (resize_w, REC_INPUT_H), interpolation=cv2.INTER_LINEAR)
    if resize_w < REC_INPUT_W:
        pad = np.full((REC_INPUT_H, REC_INPUT_W - resize_w, 3), 255, dtype=np.uint8)
        resized = np.concatenate([resized, pad], axis=1)
    imgf = resized.astype(np.float32) / 255.0
    imgf = (imgf - REC_NORM_MEAN) / REC_NORM_STD
    imgf = np.transpose(imgf, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(imgf)
