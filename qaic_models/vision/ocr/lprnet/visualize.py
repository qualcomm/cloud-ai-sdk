####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

"""LPRNet result rendering: overlay the decoded plate string on the input image."""

from __future__ import annotations

import os

OVERLAY_ALPHA = 200


def visualize_plate(image_path: str, prediction: dict, output_path: str, get_font) -> None:
    """Write the input image with the decoded plate string drawn in a bottom bar.

    `prediction` is the canonical LPRNet prediction dict
    ({"plate": str, "confidence": float, "characters": [str]}).
    `get_font` is PredictorBase.get_font (bound), which selects a CJK-capable
    font when the label contains CJK characters.
    """
    from PIL import Image, ImageDraw

    plate = prediction.get("plate", "")
    confidence = prediction.get("confidence", 0.0)
    label = f"{plate}  ({confidence * 100:.1f}%)"

    img = Image.open(image_path).convert("RGBA")
    w0, h0 = img.size
    # Upscale tiny plate images for readable overlay
    MIN_DIM = 256
    scale = max(1, int(max(MIN_DIM / w0, MIN_DIM / h0)))
    if scale > 1:
        new_size = (w0 * scale, h0 * scale)
        img = img.resize(new_size, Image.LANCZOS)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ov_draw = ImageDraw.Draw(overlay)
    short_side = min(img.width, img.height)
    font_size = max(28, short_side // 12)
    pad = max(12, short_side // 40)
    font = get_font(font_size, text=label)
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = font_size * len(label) // 2, font_size
    w, h = img.size
    ry0 = h - th - 2 * pad
    ov_draw.rectangle([0, ry0, tw + 2 * pad, h], fill=(0, 0, 0, OVERLAY_ALPHA))
    img = Image.alpha_composite(img, overlay)
    ImageDraw.Draw(img).text((pad, ry0 + pad), label, fill=(0, 255, 80), font=font)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    img.convert("RGB").save(output_path, "JPEG", quality=92)
