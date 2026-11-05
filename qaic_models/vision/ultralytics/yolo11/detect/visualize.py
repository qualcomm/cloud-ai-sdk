####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import os

from PIL import Image, ImageDraw

_BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 127, 0),
]


def visualize_detections(image_path: str, prediction: dict, output_path: str, font_getter) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    boxes = prediction.get("boxes", [])
    labels = prediction.get("labels", [])
    scores = prediction.get("scores", [])

    for j, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        color = _BOX_COLORS[j % len(_BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        text = f"{label}:{score:.2f}"
        font = font_getter(16, text)
        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
            draw.text((x1, y1), text, fill="white", font=font)
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.convert("RGB").save(output_path, "JPEG", quality=92)
