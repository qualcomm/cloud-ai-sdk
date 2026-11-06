# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Download 1000 COCO val2017 images for PGQ/AIMET calibration — YOLO / COCO-80 preprocessing.

Preprocessing: letterbox resize to square (default 640×640), normalize 0→1 (/255),
NCHW float32. Matches YOLOv5/v8/v9/v10/DETR-family input convention.

Output layout:
  <out_dir>/
    images/          preprocessed float32 .raw files (NCHW)
    batch_inputs.json  qaic-compile -aic-batch-json-input compatible index

Usage:
  # Primary: HuggingFace detection-datasets/coco (requires HF token)
  HF_TOKEN=hf_xxx python download_coco80_calibration.py

  # Fallback (no auth): COCO val2017 zip from cocodataset.org (~6.3 GB)
  python download_coco80_calibration.py --source coco_val

  # Point at a local COCO val2017 images directory (already on disk)
  python download_coco80_calibration.py --source local --coco-dir /data/coco/images/val2017

Run inside the container:
  /opt/aimet-env/bin/python download_coco80_calibration.py [flags]
"""
import argparse
import json
import os
import sys
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# Requests verifies HTTPS certificates by default. For TLS-inspecting proxies,
# set REQUESTS_CA_BUNDLE to the proxy CA bundle.
_GET = requests.get

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"


# ── Preprocessing ─────────────────────────────────────────────────────────────
def letterbox(img: Image.Image, size: int = 640, pad_value: int = 114) -> np.ndarray:
    """Letterbox resize to square, normalize /255, return NCHW float32."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
    paste_x, paste_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(img, (paste_x, paste_y))
    arr = np.array(canvas, dtype=np.float32) / 255.0   # (H,W,3) 0-1
    return arr.transpose(2, 0, 1)[np.newaxis]           # (1,3,H,W)


def save_raw(arr: np.ndarray, path: Path):
    arr.astype(np.float32).tofile(path)


# ── Source: HuggingFace detection-datasets/coco ───────────────────────────────
def download_hf(out_dir: Path, n: int, input_name: str, size: int):
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
    paths = []
    for i, sample in enumerate(ds):
        if i >= n:
            break
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        arr = letterbox(img, size)
        p = out_dir / "images" / f"{i:05d}.raw"
        save_raw(arr, p)
        paths.append(str(p))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")
    return paths


# ── Source: cocodataset.org val2017 zip (no auth, ~6.3 GB) ────────────────────
def download_coco_val(out_dir: Path, n: int, input_name: str, size: int):
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
    paths = []
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(
            m for m in zf.namelist()
            if m.lower().endswith((".jpg", ".jpeg")) and not m.startswith("__")
        )
        members = members[:n]
        for i, m in enumerate(members):
            with zf.open(m) as f:
                img = Image.open(BytesIO(f.read()))
                arr = letterbox(img, size)
            p = out_dir / "images" / f"{i:05d}.raw"
            save_raw(arr, p)
            paths.append(str(p))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(members)}")
    return paths


# ── Source: local COCO val2017 directory ─────────────────────────────────────
def download_local(out_dir: Path, n: int, input_name: str, size: int, coco_dir: str):
    val_path = Path(coco_dir)
    jpegs = sorted(val_path.rglob("*.jpg")) + sorted(val_path.rglob("*.jpeg"))
    jpegs = jpegs[:n]
    if not jpegs:
        raise RuntimeError(f"No JPEG images found under {coco_dir}")
    print(f"Using {len(jpegs)} images from {coco_dir}")
    paths = []
    for i, jp in enumerate(jpegs):
        img = Image.open(jp)
        arr = letterbox(img, size)
        p = out_dir / "images" / f"{i:05d}.raw"
        save_raw(arr, p)
        paths.append(str(p))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jpegs)}")
    return paths


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "coco_val", "local"], default="hf",
                    help="hf=HuggingFace (needs HF_TOKEN), "
                         "coco_val=cocodataset.org zip (~6.3 GB), local=local dir")
    ap.add_argument("--coco-dir", default="/data/coco/images/val2017",
                    help="Path to COCO val2017 images directory (--source local)")
    ap.add_argument("--out-dir", default="./calibration_data/coco80")
    ap.add_argument("--n", type=int, default=1000, help="Number of samples")
    ap.add_argument("--size", type=int, default=640, help="Square input size (pixels)")
    ap.add_argument("--input-name", default="images",
                    help="ONNX input tensor name used in batch_inputs.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        paths = download_hf(out_dir, args.n, args.input_name, args.size)
    elif args.source == "coco_val":
        paths = download_coco_val(out_dir, args.n, args.input_name, args.size)
    else:
        paths = download_local(out_dir, args.n, args.input_name, args.size, args.coco_dir)

    # Write qaic-compile / qaic-runner batch_inputs.json (IO-files format)
    # This format works with both qaic-compile -aic-batch-json-input and qaic-runner.
    # The {"inputs": [...]} format causes json.exception.type_error.305 — do NOT use it.
    batch = {"IO-files": [
        [{"path": p, "io-direction": "in", "data-type": "float32", "map-to": args.input_name}]
        for p in paths
    ]}
    idx = out_dir / "batch_inputs.json"
    with open(idx, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"\nDone: {len(paths)} samples → {out_dir}")
    print(f"  batch_inputs.json: {idx}")
    print()
    print("Use with AIMET (YOLO input convention: 0-1, no per-channel normalization):")
    print("  import numpy as np")
    print(f"  calibration_inputs = [[{{'{args.input_name}': np.fromfile(p, dtype=np.float32)")
    print(f"                          .reshape(1,3,{args.size},{args.size})}}]")
    print(f"                        for p in sorted(Path('{out_dir}/images').glob('*.raw'))[:1000]]")
    print()
    print("Use with qaic-compile PGQ:")
    print(f"  qaic-compile -m=model.onnx \\")
    print(f"    -aic-batch-json-input={idx} \\")
    print(f"    -dump-profile=pgq_profile.yaml \\")
    print(f"    -quantization-calibration=KLMinimization \\")
    print(f"    -quantization-schema-activations=symmetric_with_uint8")


if __name__ == "__main__":
    main()
