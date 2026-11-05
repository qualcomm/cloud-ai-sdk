####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import argparse


def run_ultralytics_classify_demo(model_cls, *, description: str, default_variant: str, supported_variants: tuple[str, ...]) -> None:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--image", required=True, help="Path to an RGB image")
    ap.add_argument("--model", required=True, help="Generated work directory, direct .onnx file, or QPC directory")
    ap.add_argument("--variant", default=default_variant, choices=supported_variants)
    ap.add_argument(
        "--provider",
        default="QAicExecutionProvider",
        choices=["QAicExecutionProvider", "CPUExecutionProvider"],
    )
    ap.add_argument("--device-id", type=int, default=None, help="QAIC device/QID (default: SDK selection)")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    model = model_cls(args.model, variant=args.variant, provider=args.provider, device_id=args.device_id)
    result = model.predict(args.image, args.out)
    prediction = result["prediction"]

    print(f"latency_ms: {result['latency_ms']:.3f}")
    top1 = prediction["top1"]
    print(f"top1: {top1['score']:.4f}\t{top1['label']}\t(class_id={top1['class_id']})")
    print("rank\tscore\tlabel\tclass_id")
    for rank, item in enumerate(prediction["topk"], start=1):
        print(f"{rank}\t{item['score']:.4f}\t{item['label']}\t{item['class_id']}")
    if args.out:
        print(f"wrote: {args.out}")
