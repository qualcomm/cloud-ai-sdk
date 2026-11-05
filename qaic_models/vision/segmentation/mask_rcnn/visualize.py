####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw

from qaic_models.base.predict_base import hex_to_rgb


_MASK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 127, 0),
]


def _label_anchor(draw, label, box, image_size, font):
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    image_width, image_height = image_size
    x1, y1, _, _ = (float(value) for value in box)

    max_x = max(0, image_width - text_width - 4)
    label_x = min(max(0, int(round(x1))), max_x)
    label_y = int(round(y1)) - text_height - 4
    if label_y < 0:
        label_y = min(int(round(y1)) + 2, max(0, image_height - text_height - 2))
    return label_x, label_y


def visualize_instances(image_path: str, prediction: dict, output_path: str, masks, font_getter) -> None:
    img = Image.open(image_path).convert("RGB").resize((1216, 800), resample=Image.BILINEAR)
    base_arr = np.array(img).astype(np.float32)
    overlay = base_arr.copy()
    instances = prediction.get("instances", [])

    if masks is not None:
        for j, _inst in enumerate(instances):
            mask = masks[j] if j < len(masks) else None
            if mask is None:
                continue
            color = np.array(_MASK_COLORS[j % len(_MASK_COLORS)], dtype=np.float32)
            m = mask.astype(bool)
            overlay[m] = 0.5 * overlay[m] + 0.5 * color

    vis_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(vis_img)
    font = font_getter(16)

    for j, inst in enumerate(instances):
        label = f"{inst['label']}:{inst['score']:.2f}"
        r, g, b = hex_to_rgb(("#%02x%02x%02x" % _MASK_COLORS[j % len(_MASK_COLORS)]))
        box = inst.get("box_xyxy")
        if box:
            draw.rectangle([float(v) for v in box], outline=(r, g, b), width=2)
            lx, ly = _label_anchor(draw, label, box, vis_img.size, font)
        else:
            # Preserve a visible diagnostic label for malformed predictions that
            # do not provide the required box field.
            lx, ly = 4, 4 + j * 18
        try:
            bbox = draw.textbbox((lx + 2, ly + 1), label, font=font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(r, g, b))
            draw.text((lx + 2, ly + 1), label, fill="white", font=font)
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    vis_img.convert("RGB").save(output_path, "JPEG", quality=92)
