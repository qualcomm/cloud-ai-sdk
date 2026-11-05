# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Minimal Triton client for YOLO inference."""

import argparse
import json
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from postprocess import postprocess_yolo, COCO_CLASSES

# Color palette
BOX_COLORS = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]  # red, green, blue


def draw_detections(image, detections, output_path="output.jpg"):
    """Draw bounding boxes and labels on image."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # Scale font and line width based on image size
    orig_w, orig_h = img.size
    font_size = max(16, min(orig_w, orig_h) // 40)
    line_width = max(2, min(orig_w, orig_h) // 200)

    # Common Ubuntu font paths
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    font = ImageFont.load_default()
    font_found = False
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            font_found = True
            break
        except (OSError, IOError):
            continue
    if not font_found:
        print("Warning: No TrueType font found, using default (text may be very small)")

    # Assign colors to unique class_ids
    unique_classes = list(dict.fromkeys(det["class_id"] for det in detections))
    class_to_color = {cid: BOX_COLORS[i % len(BOX_COLORS)] for i, cid in enumerate(unique_classes)}

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        class_id = det["class_id"]
        label = f"{COCO_CLASSES.get(class_id, class_id)} {score:.2f}"

        color = class_to_color[class_id]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label with background
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Label background
        draw.rectangle([x1, y1 - text_h - 5, x1 + text_w + 5, y1], fill=color)
        # Label text
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)

    img.save(output_path)
    print(f"Output image saved to: {output_path}")


def run_triton_inference(triton_url, image_path, model_name="yolo", output_path="output.jpg"):
    """Run inference via Triton server."""
    triton_url = triton_url.rstrip('/')
    if not triton_url.startswith("http"):
        triton_url = f"http://{triton_url}"

    print(f"Loading image from: {image_path}")
    if image_path.startswith("http"):
        img = Image.open(urllib.request.urlopen(image_path))
    else:
        img = Image.open(image_path)

    # Keep original for output
    original_img = img.convert("RGB").copy()

    # Resize for inference
    img_resized = img.convert("RGB").resize((640, 640))
    img_array = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1).reshape(1, 3, 640, 640)
    img_array /= 255.0

    payload = {
        "inputs": [
            {
                "name": "images",
                "shape": list(img_array.shape),
                "datatype": "FP32",
                "data": img_array.tolist()
            }
        ]
    }

    req = urllib.request.Request(
        f"{triton_url}/v2/models/{model_name}/infer",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )

    print(f"Sending inference request to {triton_url}/v2/models/{model_name}/infer")
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode())
        output_data = result["outputs"][0]["data"]
        output_array = np.array(output_data, dtype=np.float32)
        detections = postprocess_yolo(output_array)

        # Scale boxes back to original image size
        orig_w, orig_h = original_img.size
        scale_x = orig_w / 640
        scale_y = orig_h / 640

        for det in detections:
            det["bbox"] = [
                det["bbox"][0] * scale_x,
                det["bbox"][1] * scale_y,
                det["bbox"][2] * scale_x,
                det["bbox"][3] * scale_y
            ]

        # Draw detections on original image
        draw_detections(original_img, detections, output_path)

        print(f"\n{'='*50}")
        print(f"Detected {len(detections)} objects:")
        print(f"{'='*50}")
        for i, det in enumerate(detections[:10]):
            class_name = COCO_CLASSES.get(det['class_id'], f"unknown({det['class_id']})")
            print(f"{i+1}. {class_name}: score={det['score']:.3f}")
            print(f"   Box: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, "
                  f"{det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
        if len(detections) > 10:
            print(f"... and {len(detections) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="YOLO Triton Inference Client")
    parser.add_argument("--endpoint", type=str, required=True,
                        help="Triton server URL (e.g., localhost:8000)")
    parser.add_argument("--image", type=str,
                        default="https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg",
                        help="Image path or URL")
    parser.add_argument("--model", type=str, default="yolo",
                        help="Model name in Triton (default: yolo)")
    parser.add_argument("--output", type=str, default="output.jpg",
                        help="Output image path (default: output.jpg)")
    args = parser.parse_args()

    run_triton_inference(args.endpoint, args.image, args.model, args.output)


if __name__ == "__main__":
    main()
