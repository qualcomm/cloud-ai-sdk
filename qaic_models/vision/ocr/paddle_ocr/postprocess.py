# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# ---------------------------------------------------------------------------
#
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

# Portions of this file — the DBPostProcess methods get_mini_boxes,
# box_score_fast, unclip, and boxes_from_bitmap — are adapted from the
# PaddleOCR project:
# (https://github.com/PaddlePaddle/PaddleOCR/blob/v3.7.0/ppocr/postprocess/db_postprocess.py)
#
# Modifications (Apache-2.0 section 4(b)):
#   - paddle dependency, polygons_from_bitmap, box_score_slow,
#     __call__, and DistillationDBPostProcess removed; the port needs
#     only numpy / opencv / shapely / pyclipper.
#   - Methods converted to static/classmethod style with type hints,
#     lazy imports, an empty-polygon guard in unclip, OpenCV 4
#     findContours return handling, and fixed 640x640-map cropping
#     handled by the postprocess_det wrapper instead of __call__.

"""PaddleOCR PP-OCRv6 output interpretation (det DBPostProcess + rec greedy CTC).

* **det** — the single output is a text-region probability map ``(1,1,640,640)``,
  values in [0,1] (sigmoid-activated; bimodal: background ~0, text ~1) — no further
  activation. The detector uses the PP-OCRv6 ``DBPostProcess`` geometry: threshold
  -> rotated min-area box -> score -> pyclipper unclip -> rotated min-area box.
  Single class: ``"text"``.
* **rec** — the deployable graph outputs the per-timestep greedy-CTC summary:
  ``ctc_token_ids`` (int32 ``(B,T)``) and ``ctc_max_probs`` (float32 ``(B,T)``) —
  the argmax class and its probability per timestep (already softmaxed; do not
  apply a second softmax). Greedy CTC: collapse consecutive repeat ids, drop
  blank (idx 0), map class idx (1-based) -> ``char_dict[idx-1]``.
  ``decode_ctc``/``postprocess_rec`` still interpret the raw full-probability
  matrix ``(B,T,18710)`` (export validation / goldens).

Both functions normalize the raw output from the CPU path (a list) and the QAIC path
(a dict keyed by tensor name) into one internal array before interpreting it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .preprocess import DET_INPUT_H, DET_INPUT_W

# Shared output tensor name (Paddle default) for both det and rec.
OUTPUT_NAME = "fetch_name_0"

# ── Detection ─────────────────────────────────────────────────────────────────
# Postprocess constants match the PP-OCRv6 medium det config
# (configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml PostProcess: DBPostProcess).
DET_THRESH = 0.2  # binarize the probability map (DB "thresh")
DET_BOX_THRESH = 0.45  # mean-prob score filter (DB "box_thresh")
DET_UNCLIP_RATIO = 1.4  # dilate each contour before box extraction (DB "unclip_ratio")
DET_MIN_BOX_SIZE = 3  # min box side (canvas px)
DET_MAX_CANDIDATES = 3000

# ── PP-DocLayoutV3 ───────────────────────────────────────────────────────────
LAYOUT_NUM_QUERIES = 300
LAYOUT_NUM_CLASSES = 25
LAYOUT_SCORE_THRESHOLD = 0.5
LAYOUT_LABELS = (
    "abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
    "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number",
    "header", "header_image", "image", "inline_formula", "number", "paragraph_title",
    "reference", "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote",
)

# Regions that are safe to send through the ordinary text detector/recognizer.
# Other layout classes remain in the result as placeholders until a
# table/chart/formula/image-specific handler is added.
LAYOUT_OCR_LABELS = frozenset({
    "abstract", "algorithm", "aside_text", "content", "doc_title",
    "figure_title", "footer", "footnote", "formula_number", "header",
    "number", "paragraph_title", "reference", "reference_content", "text",
    "vertical_text", "vision_footnote",
})


def _layout_sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float32), -80.0, 80.0)))


def _layout_heads(outputs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = outputs.values() if isinstance(outputs, dict) else outputs
    logits = boxes = order = None
    for value in values:
        array = np.asarray(value, dtype=np.float32)
        if array.size == LAYOUT_NUM_QUERIES * LAYOUT_NUM_CLASSES:
            logits = array.reshape(-1, LAYOUT_NUM_QUERIES, LAYOUT_NUM_CLASSES)
        elif array.size == LAYOUT_NUM_QUERIES * 4:
            boxes = array.reshape(-1, LAYOUT_NUM_QUERIES, 4)
        elif array.size == LAYOUT_NUM_QUERIES * LAYOUT_NUM_QUERIES:
            order = array.reshape(-1, LAYOUT_NUM_QUERIES, LAYOUT_NUM_QUERIES)
    if logits is None or boxes is None or order is None:
        raise ValueError("layout outputs must include logits, pred_boxes, and order_logits")
    if any(array.shape[0] != 1 for array in (logits, boxes, order)):
        raise ValueError("layout stage accepts one document per pass")
    return logits[0], boxes[0], order[0]


def postprocess_layout(outputs: Any, source_shape: Tuple[int, int], threshold: float = LAYOUT_SCORE_THRESHOLD) -> Dict[str, Any]:
    """Reduce PP-DocLayoutV3 heads to source-frame, reading-ordered regions."""
    src_h, src_w = source_shape
    logits, pred_boxes, order_logits = _layout_heads(outputs)
    scores = _layout_sigmoid(logits).reshape(-1)
    top = np.argpartition(scores, -LAYOUT_NUM_QUERIES)[-LAYOUT_NUM_QUERIES:]
    top = top[np.argsort(scores[top])[::-1]]
    order_scores = _layout_sigmoid(order_logits)
    votes = (order_scores * np.triu(np.ones_like(order_scores), 1)
             + (1.0 - order_scores.T) * np.tril(np.ones_like(order_scores), -1)).sum(axis=0)
    sequence = np.empty(LAYOUT_NUM_QUERIES, dtype=np.int32)
    sequence[np.argsort(votes)] = np.arange(LAYOUT_NUM_QUERIES)
    regions = []
    for index in top:
        score = float(scores[index])
        if score <= threshold:
            continue
        query, class_id = divmod(int(index), LAYOUT_NUM_CLASSES)
        cx, cy, width, height = (float(value) for value in pred_boxes[query])
        box = [max(0.0, min(float(src_w), (cx - width / 2) * src_w)),
               max(0.0, min(float(src_h), (cy - height / 2) * src_h)),
               max(0.0, min(float(src_w), (cx + width / 2) * src_w)),
               max(0.0, min(float(src_h), (cy + height / 2) * src_h))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        regions.append({"box": box, "label": LAYOUT_LABELS[class_id], "class_id": class_id,
                        "score": round(score, 4), "order": int(sequence[query]) + 1,
                        "_vote": float(votes[query])})
    regions.sort(key=lambda region: (region["order"], -region["_vote"]))
    for region in regions:
        region.pop("_vote")
    return {"regions": regions}


def layout_region_is_ocr_eligible(region: Dict[str, Any]) -> bool:
    """Return whether a layout region should enter the standard OCR path."""
    label = region.get("label")
    # Preserve compatibility with older synthetic callers that omitted labels.
    return not label or label in LAYOUT_OCR_LABELS


def layout_placeholder(region: Dict[str, Any]) -> Dict[str, Any]:
    """Build a text-shaped result entry for a non-OCR layout region."""
    box = [float(value) for value in region["box"][:4]]
    layout = {key: region[key] for key in ("box", "label", "class_id", "score", "order")}
    label = str(region.get("label") or "region")
    return {
        "text": f"[{label}]",
        "confidence": 0.0,
        "box": box + [0.0],
        "status": "layout_placeholder",
        "layout": layout,
    }

# ── Recognition ───────────────────────────────────────────────────────────────
REC_BLANK_IDX = 0  # CTC blank token index
REC_NUM_CLASSES = 18710  # 18708 dict chars + trailing space + blank
# Deployable rec graph outputs: the per-timestep greedy-CTC summary (argmax
# class id + its probability) instead of the full (B,T,18710) probability
# matrix — the same text and confidence from ~3 KB of output instead of
# ~28.7 MB per batch-2 pass (see export.CtcGreedySummaryTransform).
REC_TOKEN_IDS_NAME = "ctc_token_ids"
REC_MAX_PROBS_NAME = "ctc_max_probs"


def load_char_dict(path) -> List[str]:
    """Load the character dictionary (one char per line; last entry is a space)."""
    with open(path) as f:
        content = f.read()
    chars = content.split("\n")
    if chars and chars[-1] == "":
        chars.pop()
    return chars


def _first_output_array(outputs: Any) -> np.ndarray:
    """Normalize a raw model output (QAIC dict / CPU list / raw array) to a float32 array.

    The QAIC path returns a dict keyed by tensor name; the CPU path returns a list of
    one array. Raises ValueError if the expected output is missing.
    """
    if isinstance(outputs, dict):
        if OUTPUT_NAME not in outputs:
            raise ValueError(f"missing expected QAIC output {OUTPUT_NAME!r}; got {sorted(outputs)}")
        value = outputs[OUTPUT_NAME]
    elif isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise ValueError(f"expected exactly 1 output tensor, got {len(outputs)}")
        value = outputs[0]
    else:
        value = outputs
    return np.asarray(value, dtype=np.float32)


class DBPostProcess:
    """PP-OCRv6 DB postprocess adapted from PaddleOCR v3.7.0.

    PaddleOCR 3.7's published package exposes the PaddleX pipeline but does
    not package its legacy ``ppocr`` modules.  This narrow local port preserves
    the detector's reference geometry without making the recipe depend on the
    full Paddle/PaddleX runtime.  In particular, it retains the two
    ``minAreaRect`` passes and the vector ``pyclipper`` unclip; substituting a
    raster dilation changes the right edge of narrow text lines.
    """

    def __init__(self, thresh: float, box_thresh: float, unclip_ratio: float):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.min_size = DET_MIN_BOX_SIZE
        self.max_candidates = DET_MAX_CANDIDATES

    @staticmethod
    def get_mini_boxes(contour) -> Tuple[List[np.ndarray], float]:
        import cv2

        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda point: point[0])
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        return [points[index_1], points[index_2], points[index_3], points[index_4]], min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
        import cv2

        height, width = bitmap.shape[:2]
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, width - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, width - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, height - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, height - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return float(cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0])

    @staticmethod
    def unclip(box: np.ndarray, unclip_ratio: float):
        try:
            import pyclipper
            from shapely.geometry import Polygon
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR DB postprocess requires pyclipper and shapely; "
                "install the paddle-ocr or paddle-ocr-infer extra."
            ) from exc

        polygon = Polygon(box)
        if polygon.length == 0:
            return []
        distance = polygon.area * unclip_ratio / polygon.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return offset.Execute(distance)

    def boxes_from_bitmap(
        self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int
    ) -> Tuple[np.ndarray, List[float]]:
        """Return PaddleOCR's expanded quadrilaterals in destination coordinates."""
        import cv2

        height, width = bitmap.shape
        contours = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        boxes: List[np.ndarray] = []
        scores: List[float] = []
        for contour in contours[: self.max_candidates]:
            points, short_side = self.get_mini_boxes(contour)
            if short_side < self.min_size:
                continue
            points = np.asarray(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if score < self.box_thresh:
                continue
            expanded = self.unclip(points, self.unclip_ratio)
            if len(expanded) != 1:
                continue
            box, short_side = self.get_mini_boxes(np.asarray(expanded).reshape(-1, 1, 2))
            if short_side < self.min_size + 2:
                continue
            box = np.asarray(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.asarray(boxes, dtype="int32"), scores


def postprocess_det(
    outputs: Any,
    meta: dict,
    thresh: float = DET_THRESH,
    box_thresh: float = DET_BOX_THRESH,
    unclip_ratio: float = DET_UNCLIP_RATIO,
) -> Dict[str, Any]:
    """DBPostProcess for PP-OCRv6 text detection.

    ``outputs`` is the raw det model output (QAIC dict / CPU list / raw array);
    ``meta`` is the letterbox metadata from :func:`preprocess.letterbox_det`.

    This is a wrapper around the PP-OCRv6 ``DBPostProcess`` geometry.  It crops
    the fixed 640x640 QPC map to the actual resized image before extracting
    contours: the remaining right/bottom pixels are padding, not image
    coordinates.  That crop lets PaddleOCR's per-axis destination mapping undo
    the stride-rounded resize exactly.

    Returns ``{"boxes": [[x0,y0,x1,y1], ...], "labels": ["text", ...], "scores": [...]}``.
    """
    prob = _first_output_array(outputs).reshape(DET_INPUT_H, DET_INPUT_W)
    src_h, src_w = int(meta["src_h"]), int(meta["src_w"])
    resized_h, resized_w = int(meta["resized_h"]), int(meta["resized_w"])
    if not 0 < resized_h <= DET_INPUT_H or not 0 < resized_w <= DET_INPUT_W:
        raise ValueError(
            f"invalid detector resized dimensions {(resized_h, resized_w)} for "
            f"{DET_INPUT_H}x{DET_INPUT_W} canvas"
        )

    # DBPostProcess uses the map dimensions as the source frame for coordinate
    # restoration.  Supplying the padded 640x640 map here would make x scale as
    # 640/src_w instead of resized_w/src_w on non-square inputs.
    valid_prob = prob[:resized_h, :resized_w]
    db_postprocess = DBPostProcess(thresh, box_thresh, unclip_ratio)
    quads, scores = db_postprocess.boxes_from_bitmap(
        valid_prob,
        valid_prob > db_postprocess.thresh,
        src_w,
        src_h,
    )

    # The recipe's stable public schema is axis-aligned boxes.  The extrema of
    # PaddleOCR's expanded rotated quadrilateral conservatively cover each text
    # region and keep the existing crop/visualization contract intact.
    orig_boxes = [
        [
            float(quad[:, 0].min()),
            float(quad[:, 1].min()),
            float(quad[:, 0].max()),
            float(quad[:, 1].max()),
        ]
        for quad in quads
    ]
    order = np.argsort(scores)[::-1]
    orig_boxes = [orig_boxes[int(index)] for index in order]
    scores = [round(float(scores[int(index)]), 4) for index in order]
    labels = ["text"] * len(orig_boxes)
    return {"boxes": orig_boxes, "labels": labels, "scores": scores}


def decode_ctc(probs: np.ndarray, char_dict: Sequence[str]) -> Dict[str, Any]:
    """Greedy CTC decode of a ``(T, C)`` probability matrix.

    ``probs`` is the per-timestep class probabilities (already softmaxed).
    ``char_dict`` has ``C - 1`` entries (class idx 0 is the blank). Decode: argmax
    over classes at each timestep, collapse consecutive repeats, strip the blank
    (idx 0), map class idx (1-based) -> ``char_dict[idx-1]``. Confidence is the mean
    of the max-class probability over non-blank timesteps (0.0 when all blank).

    Returns ``{"text": str, "confidence": float}``.
    """
    T, C = probs.shape
    if C - 1 != len(char_dict):
        raise ValueError(f"class dim {C} does not match char dict of {len(char_dict)} (+ blank)")
    idx = probs.argmax(axis=1)
    text: List[str] = []
    prev = -1
    for t in range(T):
        i = int(idx[t])
        if i != prev and i != REC_BLANK_IDX:
            text.append(char_dict[i - 1])
        prev = i
    maxp = probs.max(axis=1)
    nonblank = idx != REC_BLANK_IDX
    confidence = float(maxp[nonblank].mean()) if nonblank.any() else 0.0
    return {"text": "".join(text), "confidence": round(confidence, 4)}


def decode_ctc_summary(token_ids: np.ndarray, max_probs: np.ndarray, char_dict: Sequence[str]) -> Dict[str, Any]:
    """Greedy CTC decode of a pre-reduced ``(T,)`` token-id / max-probability pair.

    The exact per-timestep reduction of :func:`decode_ctc`: ``token_ids[t]`` is
    the argmax class at timestep t and ``max_probs[t]`` its probability.
    Decode: collapse consecutive repeat ids, strip the blank (idx 0), map class
    idx (1-based) -> ``char_dict[idx-1]``. Confidence is the mean of
    ``max_probs`` over non-blank timesteps (0.0 when all blank).

    Returns ``{"text": str, "confidence": float}``.
    """
    token_ids = np.asarray(token_ids).ravel()
    max_probs = np.asarray(max_probs, dtype=np.float32).ravel()
    if token_ids.shape != max_probs.shape:
        raise ValueError(f"token ids {token_ids.shape} and max probs {max_probs.shape} must match")
    text: List[str] = []
    prev = -1
    for t in range(len(token_ids)):
        i = int(token_ids[t])
        if i != prev and i != REC_BLANK_IDX:
            text.append(char_dict[i - 1])
        prev = i
    nonblank = token_ids != REC_BLANK_IDX
    confidence = float(max_probs[nonblank].mean()) if nonblank.any() else 0.0
    return {"text": "".join(text), "confidence": round(confidence, 4)}


def postprocess_rec(outputs: Any, char_dict: Sequence[str]) -> Dict[str, Any]:
    """Normalize + decode the rec model output.

    ``outputs`` is the raw rec model output (QAIC dict / CPU list / raw array);
    ``char_dict`` is the 18,709-character dictionary (18,708 dict chars + trailing
    space). Validates the ``(1,T,18710)`` shape, converts float16 -> float32, and
    runs :func:`decode_ctc`. An all-blank decode yields ``text=""``,
    ``confidence=0.0``.

    Returns ``{"text": str, "confidence": float}``.
    """
    arr = _first_output_array(outputs)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"expected rec output shape (1,T,{REC_NUM_CLASSES}), got {arr.shape}")
    if arr.shape[2] != REC_NUM_CLASSES:
        raise ValueError(f"expected class dim {REC_NUM_CLASSES}, got {arr.shape[2]}")
    probs = arr.reshape(arr.shape[1], REC_NUM_CLASSES)
    return decode_ctc(probs, char_dict)


def _summary_output_arrays(outputs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize the rec summary outputs to ``(token_ids, max_probs)`` arrays.

    The QAIC path returns a dict keyed by tensor name; the CPU path returns
    the two arrays in graph output order. Raises ValueError if an expected
    output is missing.
    """
    if isinstance(outputs, dict):
        missing = [name for name in (REC_TOKEN_IDS_NAME, REC_MAX_PROBS_NAME) if name not in outputs]
        if missing:
            raise ValueError(f"missing expected QAIC outputs {missing}; got {sorted(outputs)}")
        token_ids, max_probs = outputs[REC_TOKEN_IDS_NAME], outputs[REC_MAX_PROBS_NAME]
    elif isinstance(outputs, (list, tuple)):
        if len(outputs) != 2:
            raise ValueError(f"expected exactly 2 summary outputs, got {len(outputs)}")
        token_ids, max_probs = outputs
    else:
        raise ValueError("expected a QAIC output dict or the ordered 2-output CPU list")
    return np.asarray(token_ids, dtype=np.int32), np.asarray(max_probs, dtype=np.float32)


def decode_rec_batch(outputs: Any, char_dict: Sequence[str]) -> List[Dict[str, Any]]:
    """Decode the rec model's batched summary output.

    ``outputs`` is the raw rec model output (QAIC dict / CPU list) for a batch
    of B text lines: ``ctc_token_ids`` (int32 ``(B,T)``) and ``ctc_max_probs``
    (float32 ``(B,T)``) (the pipeline feeds the rec throughput QPC in fixed
    batches, padding the last batch by repeating its final crop). Returns one
    ``{"text": str, "confidence": float}`` per batch row, in row order.
    """
    token_ids, max_probs = _summary_output_arrays(outputs)
    if token_ids.ndim != 2 or max_probs.shape != token_ids.shape:
        raise ValueError(f"expected matching rank-2 summary shapes (B,T), got {token_ids.shape} / {max_probs.shape}")
    return [decode_ctc_summary(token_ids[b], max_probs[b], char_dict) for b in range(token_ids.shape[0])]


def reading_order(
    boxes: Sequence[Sequence[float]], scores: Sequence[float]
) -> List[Tuple[Sequence[float], float]]:
    """Order detector output for emission: top to bottom, left to right within a line.

    ``boxes``/``scores`` are the det postprocess output (detector score order).
    Returns ``(box, score)`` pairs sorted by box top edge, then left edge — the
    order the line crops are prepared and decoded in.
    """
    return sorted(zip(boxes, scores), key=lambda pair: (pair[0][1], pair[0][0]))


def build_texts(
    ordered_pairs: Sequence[Tuple[Sequence[float], float]],
    decodes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pair reading-ordered det boxes with rec decodes into the pipeline schema.

    ``ordered_pairs`` is :func:`reading_order` output (the line crops were
    prepared and decoded in exactly that order) and ``decodes`` is one
    ``{"text": str, "confidence": float}`` per pair, in the same order. Returns
    ``[{"text", "confidence", "box": [x0, y0, x1, y1, score]}, ...]``.
    """
    if len(ordered_pairs) != len(decodes):
        raise ValueError(f"expected {len(ordered_pairs)} decodes, got {len(decodes)}")
    return [
        {
            "text": pred["text"],
            "confidence": pred["confidence"],
            "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)],
        }
        for (box, score), pred in zip(ordered_pairs, decodes)
    ]


def split_layout_regions(layout_prediction: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a layout prediction into (all regions, OCR-eligible regions, placeholders).

    The OCR-eligible regions are the ones the detector scans; the placeholders
    are the text-shaped entries the pipeline emits for the remaining (non-text)
    regions so the document's reading order survives the OCR pass.
    """
    layout_regions = layout_prediction["regions"]
    ocr_regions = [
        region for region in layout_regions if layout_region_is_ocr_eligible(region)
    ]
    placeholders = [
        layout_placeholder(region)
        for region in layout_regions
        if not layout_region_is_ocr_eligible(region)
    ]
    return layout_regions, ocr_regions, placeholders


def build_prediction(
    layout_regions: Sequence[Dict[str, Any]],
    placeholders: Sequence[Dict[str, Any]],
    kept_pairs: Sequence[Tuple[Sequence[float], float]],
    kept_regions: Sequence[Dict[str, Any]],
    decodes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the pipeline prediction from layout regions and line decodes.

    Pairs the reading-ordered line boxes with the rec decodes via
    :func:`build_texts`, tags each text with its parent layout region, appends
    the non-text layout ``placeholders``, and sorts everything into document
    reading order. Returns ``{"texts": [...], "regions": [...]}`` — the
    prediction both demo CLIs print.
    """
    texts = build_texts(kept_pairs, decodes)
    for text, region in zip(texts, kept_regions):
        text["layout"] = {
            key: region[key] for key in ("box", "label", "class_id", "score", "order")
        }
    texts.extend(placeholders)
    texts.sort(key=lambda entry: (
        int(entry.get("layout", {}).get("order", 1_000_000)),
        float(entry.get("box", [0.0, 0.0])[1]),
        float(entry.get("box", [0.0, 0.0])[0]),
    ))
    return {"texts": texts, "regions": layout_regions}
