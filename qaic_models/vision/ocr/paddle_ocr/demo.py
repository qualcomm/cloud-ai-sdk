# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Run the generated layout -> det -> rec OCR pipeline on one document image.

The simple end-to-end example of the pipeline assembled here, downstream of
the stage predictors in :mod:`model`: one layout session finds the document
regions, two detector sessions scan the OCR-eligible regions, and the line
crops fan out over four recognizer sessions (the rec throughput QPC takes
two lines per batch). For PDF input, more sessions, and per-stage profiling,
see :mod:`demo_pages`.

On Cloud AI 100 the default placement is layout on SoC 0, the two detector
sessions on SoCs 1 and 2 (the 16-core det QPC fills a 16-NSP SoC), and four
4-core rec sessions on SoC 3 (they fill one 16-NSP SoC); the CPU reference
uses one session per stage and runs the passes sequentially.
"""

from __future__ import annotations

import argparse
import json
import os

from .pipeline import PaddleOCRPipeline as Pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the PP-DocLayoutV3 -> PP-OCRv6 OCR pipeline on a document image.")
    parser.add_argument("--image", required=True, help="Document image to scan for text")
    parser.add_argument("--model", required=True, help="Work directory produced by export.py")
    parser.add_argument(
        "--provider", default="QAicExecutionProvider",
        choices=["QAicExecutionProvider", "CPUExecutionProvider"],
        help="Run the compiled QPCs or the CPU ONNX reference",
    )
    parser.add_argument("--layout-device", type=int, default=0, help="Device for the layout session (QAIC provider only)")
    parser.add_argument(
        "--det-devices", default="1,2",
        help="Devices for the det sessions, one 16-core session each (QAIC provider only)",
    )
    parser.add_argument("--rec-device", type=int, default=3, help="Device for the rec sessions (QAIC provider only)")
    parser.add_argument(
        "--rec-sessions", type=int, default=4,
        help="Concurrent rec QPC sessions on the rec device (QAIC provider only; 4 fill a 16-NSP SoC)",
    )
    parser.add_argument("--out", default=None, help="Optional annotated JPEG output path")
    parser.add_argument("--json-out", default=None, help="Optional path to write the prediction JSON payload (same object as the final stdout line)")
    args = parser.parse_args(argv)
    if args.image.lower().endswith(".pdf"):
        parser.error("--image is a single image here; use qaic_models.vision.ocr.paddle_ocr.demo_pages for PDF input")

    pipeline = Pipeline(
        args.model,
        provider=args.provider,
        layout_device_id=args.layout_device,
        det_devices=tuple(int(d) for d in args.det_devices.split(",") if d.strip()),
        rec_device_id=args.rec_device,
        rec_sessions=args.rec_sessions,
    )
    result = pipeline.predict(args.image, output_path=args.out)
    prediction = result["prediction"]
    # stdout layout: summary, then the detected text (one line per region,
    # top to bottom), then the full JSON prediction on the final line so
    # line-based consumers can still parse it.
    print(f"latency_ms: {result['latency_ms']:.3f}")
    print(f"regions: {len(prediction['regions'])}")
    texts = prediction["texts"]
    print(f"lines: {len(texts)}")
    if args.out:
        print(f"wrote: {args.out}")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(prediction, f, ensure_ascii=False, sort_keys=True, indent=2)
        print(f"wrote: {args.json_out}")
    if texts:
        print()
        for entry in texts:
            print(entry["text"])
        print()
    print(json.dumps(result["prediction"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
