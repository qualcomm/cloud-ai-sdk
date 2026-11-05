# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""PaddleOCR PP-OCRv6 result rendering.

* ``visualize_det`` — overlay the detected text-region boxes (with per-box score
  chips) on the input image (det-only result schema).
* ``visualize_rec`` — overlay layout regions as translucent fills and detected
  text-line boxes on the input image (the pipeline result schema).
* ``visualize_layout`` — overlay layout regions as translucent fills with
  ``label: score`` chips (the layout-only result schema; the demo's
  ``--layout`` mode).

Both are recipe-local rendering; they write only fields present in the validated
prediction schemas and handle empty predictions.
"""

from __future__ import annotations

import os

from qaic_models.base.predict_base import hex_to_rgb

# Per-box outline palette (cycles when there are more boxes than colors).
COLORS = [
    "#FF3333", "#33FF33", "#3399FF", "#FF9900", "#CC33FF",
    "#FF66CC", "#00CCCC", "#FFFF33", "#FF6633", "#33CCFF",
]

DET_OVERLAY_ALPHA = 160
LAYOUT_FILL_ALPHA = 55


def _load_rgba_image(source):
    """Load a path or convert a BGR image array to RGBA for rendering."""
    from PIL import Image

    if isinstance(source, (str, os.PathLike)):
        return Image.open(source).convert("RGBA")
    import numpy as np

    image = np.asarray(source)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image array must have shape [H, W, 3] in BGR order")
    return Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]), mode="RGB").convert("RGBA")


def visualize_det(image_path: str, prediction: dict, output_path: str, get_font) -> None:
    """Write the input image with the detected text-region boxes drawn on it.

    ``prediction`` is the det schema ``{"boxes": [[x0,y0,x1,y1], ...], "labels": [...],
    "scores": [...]}``. Boxes are drawn in the input image's coordinate frame (the
    postprocess already undoes the letterbox).
    """
    from PIL import Image, ImageDraw

    img = _load_rgba_image(image_path)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)  # used for textbbox measurement only
    ov_draw = ImageDraw.Draw(overlay)
    short_side = min(img.width, img.height)
    font_size = max(24, short_side // 30)
    boxes = prediction.get("boxes", [])
    # For small-object detectors the image-proportional font can dwarf tiny boxes;
    # cap to 60% of the smallest box height so labels fit the objects they annotate.
    if boxes:
        min_box_h = min(max(1, b[3] - b[1]) for b in boxes)
        font_size = max(12, min(font_size, int(min_box_h * 0.6)))
    box_width = max(2, min(4, short_side // 200))
    pad = max(3, font_size // 4)
    labels = prediction.get("labels", [])
    font = get_font(font_size, text=" ".join(labels))
    labels_to_draw = []
    for i, (box, label, score) in enumerate(
        zip(boxes, labels, prediction.get("scores", []))
    ):
        r, g, b = hex_to_rgb(COLORS[i % len(COLORS)])
        x1, y1, x2, y2 = box
        ov_draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, DET_OVERLAY_ALPHA), width=box_width)
        text = f"{label}: {score:.2f}"
        lx, ly = x1 + pad, y1 + box_width // 2
        try:
            bbox = draw.textbbox((lx, ly), text, font=font)
            ov_draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(r, g, b, DET_OVERLAY_ALPHA))
        except Exception:
            pass
        labels_to_draw.append((lx, ly, text))
    img = Image.alpha_composite(img, overlay)
    final_draw = ImageDraw.Draw(img)
    for lx, ly, text in labels_to_draw:
        final_draw.text((lx, ly), text, fill="white", font=font)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    img.convert("RGB").save(output_path, "JPEG", quality=92)


def visualize_rec(image_path: str, prediction: dict, output_path: str, get_font) -> None:
    """Write the input image with layout fills and text-line boxes drawn on it.

    Layout regions are drawn first as semi-transparent solid fills. Each text
    entry's ``box`` (``[x0, y0, x1, y1, score]`` in original-image coordinates)
    is then drawn as an outline on top.
    """
    from PIL import Image, ImageDraw

    img = _load_rgba_image(image_path)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    short_side = min(img.width, img.height)
    box_width = max(2, min(4, short_side // 200))
    for index, region in enumerate(prediction.get("regions", [])):
        box = region.get("box")
        if not box or len(box) < 4:
            continue
        class_id = int(region.get("class_id", index))
        r, g, b = hex_to_rgb(COLORS[class_id % len(COLORS)])
        x0, y0, x1, y1 = (int(round(value)) for value in box[:4])
        ov_draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, LAYOUT_FILL_ALPHA))
    for i, entry in enumerate(prediction.get("texts", [])):
        box = entry.get("box")
        if not box or len(box) < 4:
            continue
        r, g, b = hex_to_rgb(COLORS[i % len(COLORS)])
        x0, y0, x1, y1 = (int(round(v)) for v in box[:4])
        ov_draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, DET_OVERLAY_ALPHA), width=box_width)
    img = Image.alpha_composite(img, overlay)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    img.convert("RGB").save(output_path, "JPEG", quality=92)


def visualize_layout(image_path: str, prediction: dict, output_path: str, get_font) -> None:
    """Write the input image with layout regions drawn on it (layout-only mode).

    det and rec did not run, so each region is drawn as a semi-transparent
    solid fill in its class color with a ``label: score`` text chip in the
    same color.
    """
    from PIL import Image, ImageDraw

    img = _load_rgba_image(image_path)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)  # used for textbbox measurement only
    ov_draw = ImageDraw.Draw(overlay)
    short_side = min(img.width, img.height)
    # Valid layout regions: class, label, score, and the int-rounded box.
    regions = []
    for index, region in enumerate(prediction.get("regions", [])):
        box = region.get("box")
        if not box or len(box) < 4:
            continue
        x0, y0, x1, y1 = (int(round(value)) for value in box[:4])
        regions.append(
            {
                "class_id": int(region.get("class_id", index)),
                "label": region.get("label", ""),
                "score": float(region.get("score", 0.0)),
                "box": (x0, y0, x1, y1),
            }
        )
    for region in regions:
        r, g, b = hex_to_rgb(COLORS[region["class_id"] % len(COLORS)])
        x0, y0, x1, y1 = region["box"]
        ov_draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, LAYOUT_FILL_ALPHA))
    font = None
    labels_to_draw = []
    if regions:
        font_size = max(24, short_side // 30)
        # Cap to 60% of the smallest region height so labels fit (the same
        # idiom as visualize_det's box cap).
        min_region_h = min(max(1, region["box"][3] - region["box"][1]) for region in regions)
        font_size = max(12, min(font_size, int(min_region_h * 0.6)))
        font = get_font(font_size, text=" ".join(region["label"] for region in regions))
        pad = max(3, font_size // 4)
        for region in regions:
            r, g, b = hex_to_rgb(COLORS[region["class_id"] % len(COLORS)])
            x0, y0 = region["box"][:2]
            text = f"{region['label']}: {region['score']:.2f}"
            lx, ly = x0 + pad, y0 + pad
            # Solid chip in a 50% darkened class color: white text stays
            # readable over the pale document background, even for the
            # palette's bright entries.
            chip = tuple(int(c * 0.5) for c in (r, g, b)) + (255,)
            try:
                bbox = draw.textbbox((lx, ly), text, font=font)
                ov_draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=chip)
            except Exception:
                pass
            labels_to_draw.append((lx, ly, text))
    img = Image.alpha_composite(img, overlay)
    if labels_to_draw:
        final_draw = ImageDraw.Draw(img)
        for lx, ly, text in labels_to_draw:
            final_draw.text((lx, ly), text, fill="white", font=font)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    img.convert("RGB").save(output_path, "JPEG", quality=92)
