# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Client for the generated Triton OCR models: layout -> det -> rec.

Runs the three-stage hand-off against a running Triton server (see the Triton
serving runbook in the README): call ``paddle_ocr_layout`` for the document
regions, crop and dispatch each region to ``paddle_ocr_det_qpc`` for its
text-line boxes, offset the boxes into image coordinates, sort them into
reading order, crop and prepare each line, then dispatch the whole line list
to ``paddle_ocr_rec`` without waiting for each response. The rec model's
dynamic batching pairs the in-flight lines into batch-2 QPC passes on the
server; this client has no knowledge of the model's batch size or instance
count. The only client-side knob is ``concurrency``, the tritonclient cap on
in-flight requests (its default of 1 would serialize the dispatch; >= 2 is
all the rec batcher needs).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from qaic_models.vision.ocr.paddle_ocr import postprocess, preprocess


def _json_element(element) -> dict:
    """Decode one STRING [1] tensor element (UTF-8 JSON blob)."""
    return json.loads(element.decode("utf-8") if isinstance(element, bytes) else element)


def _order_detections(detections):
    """Sort detector triples while preserving each parent layout region."""
    return sorted(detections, key=lambda item: (item[0][1], item[0][0]))


def _prepare_rec_request(line):
    """Return one recognizer request with Triton's singleton batch axis."""
    line = np.asarray(line, dtype=np.float32)
    if line.ndim == 3:
        line = line[np.newaxis, ...]
    if line.ndim != 4 or line.shape[0] != 1:
        raise ValueError(f"expected one rec input with a singleton batch axis, got {line.shape}")
    return np.ascontiguousarray(line)


def run_triton_pipeline(image_path: str, server_url: str = "localhost:8000", *, concurrency: int = 8) -> dict:
    """Run the layout -> det -> reading order -> rec Triton hand-off on a document image.

    Returns ``{"prediction": {"texts": [...], "regions": [...]}, "latency_ms": float}``
    — the same prediction schema as :mod:`pipeline` (each text carries its
    parent layout region). ``latency_ms`` covers the whole
    hand-off end to end, client side.
    """
    import cv2
    import tritonclient.http as triton_http
    from tritonclient.http import InferInput, InferRequestedOutput

    client = triton_http.InferenceServerClient(url=server_url, concurrency=concurrency)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"could not read BGR image from {image_path!r}")

    t0 = time.perf_counter()
    # 1. Layout: the raw BGR image in, the regions JSON out.
    src_tensor = InferInput("src", list(img_bgr.shape), "UINT8")
    src_tensor.set_data_from_numpy(img_bgr)
    layout = client.infer(
        model_name="paddle_ocr_layout",
        inputs=[src_tensor],
        outputs=[InferRequestedOutput("regions"), InferRequestedOutput("region_count")],
    )
    regions = _json_element(layout.as_numpy("regions")[0])["regions"]

    # 2. Detection: crop each layout region and dispatch all the region crops
    # without waiting. The boxes come back in region coordinates and are
    # offset into the original image below.
    region_items = []
    det_handles = []
    for region in regions:
        crop = preprocess.crop_box(img_bgr, region["box"])
        if crop is None:
            continue
        tensor = InferInput("src", list(crop.shape), "UINT8")
        tensor.set_data_from_numpy(crop)
        det_handles.append(
            client.async_infer(
                model_name="paddle_ocr_det_qpc",
                inputs=[tensor],
                outputs=[InferRequestedOutput("boxes"), InferRequestedOutput("box_count")],
            )
        )
        region_items.append(region)
    detections = []
    for region, handle in zip(region_items, det_handles):
        boxes = _json_element(handle.get_result().as_numpy("boxes")[0])["boxes"]
        x0, y0 = region["box"][:2]
        for box in boxes:  # [x0, y0, x1, y1, score], detector score order
            detections.append(([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0], box[4], region))

    # 3. Recognition: sort into reading order, crop and prepare each line,
    # then dispatch the whole list without waiting — the rec model's dynamic
    # batching pairs the in-flight lines into batch-2 QPC passes.
    ordered = _order_detections(detections)
    lines = []
    for box, score, region in ordered:
        crop = preprocess.crop_box(img_bgr, box)
        if crop is not None:
            lines.append((box, score, region, _prepare_rec_request(preprocess.preprocess_rec_image(crop))))

    rec_handles = []
    for _, _, _, line in lines:
        line_tensor = InferInput("x", list(line.shape), "FP32")
        line_tensor.set_data_from_numpy(line)
        rec_handles.append(
            client.async_infer(
                model_name="paddle_ocr_rec",
                inputs=[line_tensor],
                outputs=[InferRequestedOutput("text"), InferRequestedOutput("confidence")],
            )
        )
    results = [
        (rec.as_numpy("text")[0], float(rec.as_numpy("confidence")[0]))
        for rec in (handle.get_result() for handle in rec_handles)
    ]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    texts = postprocess.build_texts(
        [(box, score) for box, score, _, _ in lines],
        [{"text": t.decode("utf-8") if isinstance(t, bytes) else t, "confidence": conf} for t, conf in results],
    )
    # Each text carries its parent layout region (same prediction schema as
    # the public pipeline).
    for text, (_, _, region, _) in zip(texts, lines):
        text["layout"] = {key: region[key] for key in ("box", "label", "class_id", "score", "order")}
    return {"prediction": {"texts": texts, "regions": regions}, "latency_ms": latency_ms}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the generated Triton layout -> det -> rec OCR models on a document image.")
    parser.add_argument("--image", required=True, help="Document image to scan for text")
    parser.add_argument("--server-url", default="localhost:8000", help="Triton HTTP endpoint")
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Client-side cap on in-flight requests (the rec batcher needs only >= 2)",
    )
    args = parser.parse_args(argv)

    result = run_triton_pipeline(args.image, args.server_url, concurrency=args.concurrency)
    prediction = result["prediction"]
    # stdout layout: summary, then the decoded text (one line per region,
    # top to bottom), then the full JSON prediction on the final line so
    # line-based consumers can still parse it.
    print(f"latency_ms: {result['latency_ms']:.3f}")
    print(f"regions: {len(prediction['regions'])}")
    print(f"lines: {len(prediction['texts'])}")
    if prediction["texts"]:
        print()
        for entry in prediction["texts"]:
            print(entry["text"])
        print()
    print(json.dumps(prediction, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
