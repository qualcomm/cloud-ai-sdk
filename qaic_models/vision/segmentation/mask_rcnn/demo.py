####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import argparse
import json
import os

from .model import MaskRCNN


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Mask R-CNN split pipeline on an image.")
    ap.add_argument("--image", required=True, help="Path to an RGB image")
    ap.add_argument(
        "--model",
        required=True,
        help=(
            "Generated work directory (export --work-dir, default build/mask_rcnn). "
            "QAicExecutionProvider needs qpc_backbone_fpn/ + qpc_roi_heads/ + model_rpn*.onnx; "
            "CPUExecutionProvider needs the model_*fp16ready.onnx files."
        ),
    )
    ap.add_argument(
        "--provider",
        default="QAicExecutionProvider",
        choices=("QAicExecutionProvider", "CPUExecutionProvider"),
        help="Inference backend (default: QAicExecutionProvider)",
    )
    ap.add_argument("--device-id", type=int, default=None, help="Cloud AI 100 QID for the QAIC provider (default: auto-select)")
    ap.add_argument("--top", type=int, default=20, help="Max instances to print")
    ap.add_argument("--score-thresh", type=float, default=0.05)
    ap.add_argument("--nms-thresh", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="Optional output image path with drawn masks/labels")
    ap.add_argument("--json-out", default=None, help="Optional path to write the prediction dict as JSON")
    args = ap.parse_args()

    model = MaskRCNN(args.model, provider=args.provider, device_id=args.device_id)
    result = model.predict(
        args.image,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        topk=args.top,
        output_path=args.out,
    )

    prediction = result["prediction"]
    print(f"num detections (after postprocess): {prediction['num_instances']}")
    for inst in prediction["instances"][: args.top]:
        x1, y1, x2, y2 = inst["box_xyxy"]
        print(f"{inst['label']}\tscore={inst['score']:.4f}\tbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
    print(f"latency_ms: {result['latency_ms']:.3f}")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=2)
        print(f"wrote: {args.json_out}")

    if args.out:
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
