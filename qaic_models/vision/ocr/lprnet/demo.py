####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import argparse
import json
import os

from .model import DEFAULT_VARIANT, SUPPORTED_VARIANTS, LPRNet


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LPRNet license plate recognition on a single image.")
    ap.add_argument("--image", required=True, help="Path to a cropped license plate image")
    ap.add_argument(
        "--model",
        required=True,
        help="Work directory produced by export.py (contains lprnet_fp16ready.onnx "
             "and/or qpc_latency/), or a direct .onnx file / QPC directory",
    )
    ap.add_argument("--device-id", type=int, default=None, help="QAIC device/QID (default: SDK selection)")
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=SUPPORTED_VARIANTS)
    ap.add_argument(
        "--provider",
        default="QAicExecutionProvider",
        choices=["QAicExecutionProvider", "CPUExecutionProvider"],
        help="QAicExecutionProvider runs the compiled QPC; CPUExecutionProvider runs the ONNX Runtime CPU reference",
    )
    ap.add_argument("--out", default=None, help="Optional output image path with the decoded plate drawn on it")
    ap.add_argument("--json-out", default=None, help="Optional path to write the prediction dict as JSON")
    args = ap.parse_args()

    model = LPRNet(args.model, variant=args.variant, provider=args.provider, device_id=args.device_id)
    result = model.predict(args.image, args.out)

    prediction = result["prediction"]
    print(f"latency_ms: {result['latency_ms']:.3f}")
    print(f"plate: {prediction['plate']}")
    print(f"confidence: {prediction['confidence']:.4f}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=2)
        print(f"wrote: {args.json_out}")

    if args.out:
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
