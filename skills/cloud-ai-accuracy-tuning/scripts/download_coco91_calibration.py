# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Download 1000 COCO val2017 images for PGQ/AIMET calibration — RCNN / COCO-91 preprocessing.

Preprocessing: resize shortest edge to `--short-size` (default 800), cap longest edge
at `--max-size` (default 1333), ImageNet mean/std normalization, NCHW float32.
Matches torchvision Faster R-CNN / Mask R-CNN / DETR input convention. Batching is
single-image (batch=1) because RCNN-family models use variable-size inputs.

Output layout:
  <out_dir>/
    images/          preprocessed float32 .raw files (NCHW, variable spatial dims)
    sizes.json       per-image (H, W) so callers can reshape correctly
    batch_inputs.json  qaic-compile -aic-batch-json-input compatible index

Usage:
  # Primary: HuggingFace detection-datasets/coco (requires HF token)
  HF_TOKEN=hf_xxx python download_coco91_calibration.py

  # Fallback (no auth): COCO val2017 zip from cocodataset.org (~6.3 GB)
  python download_coco91_calibration.py --source coco_val

  # Fixed-square input (some RCNN deployments use 1024×1024)
  python download_coco91_calibration.py --source coco_val --square 1024

  # Point at a local COCO val2017 images directory (already on disk)
  python download_coco91_calibration.py --source local --coco-dir /data/coco/images/val2017

Run inside the container:
  /opt/aimet-env/bin/python download_coco91_calibration.py [flags]
"""
import argparse
import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# Requests verifies HTTPS certificates by default. For TLS-inspecting proxies,
# set REQUESTS_CA_BUNDLE to the proxy CA bundle.
_GET = requests.get

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"


# ── Preprocessing ─────────────────────────────────────────────────────────────
def resize_keep_aspect(img: Image.Image, short_size: int, max_size: int) -> Image.Image:
    """Scale so shortest edge = short_size; cap longest edge at max_size."""
    w, h = img.size
    short, long = min(w, h), max(w, h)
    scale = short_size / short
    if long * scale > max_size:
        scale = max_size / long
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BILINEAR)


def preprocess_rcnn(img: Image.Image, short_size: int, max_size: int) -> np.ndarray:
    """Resize → ImageNet normalize → NCHW float32."""
    img = img.convert("RGB")
    img = resize_keep_aspect(img, short_size, max_size)
    arr = np.array(img, dtype=np.float32) / 255.0          # (H,W,3) 0-1
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD              # normalize
    return arr.transpose(2, 0, 1)[np.newaxis]               # (1,3,H,W)


def preprocess_square(img: Image.Image, size: int) -> np.ndarray:
    """Resize to fixed square, ImageNet normalize, NCHW float32."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis]               # (1,3,H,W)


def save_raw(arr: np.ndarray, path: Path):
    arr.astype(np.float32).tofile(path)


# ── Source: HuggingFace detection-datasets/coco ───────────────────────────────
def download_hf(out_dir: Path, n: int, input_name: str, short_size: int,
                max_size: int, square: int | None):
    token = os.environ.get("HF_TOKEN", "")
    try:
        import datasets as hf_datasets
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        import datasets as hf_datasets

    print("Streaming detection-datasets/coco validation split from HuggingFace...")
    ds = hf_datasets.load_dataset(
        "detection-datasets/coco",
        split="val",
        streaming=True,
        trust_remote_code=True,
        token=token or True,
    )
    paths, sizes = [], []
    for i, sample in enumerate(ds):
        if i >= n:
            break
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        arr = preprocess_square(img, square) if square else preprocess_rcnn(img, short_size, max_size)
        p = out_dir / "images" / f"{i:05d}.raw"
        save_raw(arr, p)
        paths.append(str(p))
        sizes.append({"h": arr.shape[2], "w": arr.shape[3]})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")
    return paths, sizes


# ── Source: cocodataset.org val2017 zip (no auth, ~6.3 GB) ────────────────────
def download_coco_val(out_dir: Path, n: int, input_name: str, short_size: int,
                      max_size: int, square: int | None):
    zip_path = out_dir / "val2017.zip"
    if not zip_path.exists():
        print(f"Downloading COCO val2017 (~6.3 GB) → {zip_path}")
        print("  This may take several minutes on a slow connection.")
        r = _GET(COCO_VAL_URL, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                pct = done * 100 // total if total else 0
                print(f"\r  {pct}%", end="", flush=True)
        print()

    print("Extracting and preprocessing...")
    paths, sizes = [], []
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(
            m for m in zf.namelist()
            if m.lower().endswith((".jpg", ".jpeg")) and not m.startswith("__")
        )
        members = members[:n]
        for i, m in enumerate(members):
            with zf.open(m) as f:
                img = Image.open(BytesIO(f.read()))
                arr = preprocess_square(img, square) if square else preprocess_rcnn(img, short_size, max_size)
            p = out_dir / "images" / f"{i:05d}.raw"
            save_raw(arr, p)
            paths.append(str(p))
            sizes.append({"h": arr.shape[2], "w": arr.shape[3]})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(members)}")
    return paths, sizes


# ── Source: local COCO val2017 directory ─────────────────────────────────────
def download_local(out_dir: Path, n: int, input_name: str, short_size: int,
                   max_size: int, square: int | None, coco_dir: str):
    val_path = Path(coco_dir)
    jpegs = sorted(val_path.rglob("*.jpg")) + sorted(val_path.rglob("*.jpeg"))
    jpegs = jpegs[:n]
    if not jpegs:
        raise RuntimeError(f"No JPEG images found under {coco_dir}")
    print(f"Using {len(jpegs)} images from {coco_dir}")
    paths, sizes = [], []
    for i, jp in enumerate(jpegs):
        img = Image.open(jp)
        arr = preprocess_square(img, square) if square else preprocess_rcnn(img, short_size, max_size)
        p = out_dir / "images" / f"{i:05d}.raw"
        save_raw(arr, p)
        paths.append(str(p))
        sizes.append({"h": arr.shape[2], "w": arr.shape[3]})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jpegs)}")
    return paths, sizes


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "coco_val", "local"], default="hf",
                    help="hf=HuggingFace (needs HF_TOKEN), "
                         "coco_val=cocodataset.org zip (~6.3 GB), local=local dir")
    ap.add_argument("--coco-dir", default="/data/coco/images/val2017",
                    help="Path to COCO val2017 images directory (--source local)")
    ap.add_argument("--out-dir", default="./calibration_data/coco91")
    ap.add_argument("--n", type=int, default=1000, help="Number of samples")
    ap.add_argument("--short-size", type=int, default=800,
                    help="Resize shortest edge to this (variable-size mode)")
    ap.add_argument("--max-size", type=int, default=1333,
                    help="Cap longest edge at this (variable-size mode)")
    ap.add_argument("--square", type=int, default=None,
                    help="If set, resize to this fixed square instead of variable-size "
                         "(e.g. 1024 for fixed-input RCNN deployments)")
    ap.add_argument("--input-name", default="input",
                    help="ONNX input tensor name used in batch_inputs.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        out_dir=out_dir, n=args.n, input_name=args.input_name,
        short_size=args.short_size, max_size=args.max_size, square=args.square,
    )
    if args.source == "hf":
        paths, sizes = download_hf(**kwargs)
    elif args.source == "coco_val":
        paths, sizes = download_coco_val(**kwargs)
    else:
        paths, sizes = download_local(**kwargs, coco_dir=args.coco_dir)

    # Write per-image sizes (needed for variable-size reshape)
    sizes_path = out_dir / "sizes.json"
    with open(sizes_path, "w") as f:
        json.dump(sizes, f, indent=2)

    # Write qaic-compile / qaic-runner batch_inputs.json (IO-files format)
    # coco91 uses variable spatial dims — shape per entry is read from sizes.json at runtime.
    # The {"inputs": [...]} format causes json.exception.type_error.305 — do NOT use it.
    batch = {"IO-files": [
        [{"path": p, "io-direction": "in", "data-type": "float32", "map-to": args.input_name}]
        for p in paths
    ]}
    idx = out_dir / "batch_inputs.json"
    with open(idx, "w") as f:
        json.dump(batch, f, indent=2)

    unique_shapes = {(s["h"], s["w"]) for s in sizes}
    print(f"\nDone: {len(paths)} samples → {out_dir}")
    print(f"  batch_inputs.json: {idx}")
    print(f"  sizes.json:        {sizes_path}")
    if len(unique_shapes) > 1:
        print(f"  Note: {len(unique_shapes)} unique spatial shapes "
              f"(variable-size inputs); AIMET calibration requires reshape per sample")
    print()
    print("Use with AIMET (RCNN/torchvision input convention: ImageNet normalized):")
    print("  import numpy as np, json")
    print(f"  sizes = json.load(open('{sizes_path}'))")
    print(f"  raws  = sorted(Path('{out_dir}/images').glob('*.raw'))[:1000]")
    print(f"  calibration_inputs = [")
    print(f"      {{'{args.input_name}': np.fromfile(str(p), dtype=np.float32)")
    print(f"                          .reshape(1, 3, s['h'], s['w'])}}")
    print(f"      for p, s in zip(raws, sizes)]")
    print()
    print("For fixed-square models, use --square 1024 to produce uniform (1,3,1024,1024) tensors.")
    print()
    print("Use with qaic-compile PGQ (fixed-size only):")
    print(f"  qaic-compile -m=model.onnx \\")
    print(f"    -aic-batch-json-input={idx} \\")
    print(f"    -dump-profile=pgq_profile.yaml \\")
    print(f"    -quantization-calibration=KLMinimization \\")
    print(f"    -quantization-schema-activations=symmetric_with_uint8")


if __name__ == "__main__":
    main()
