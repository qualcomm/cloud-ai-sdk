# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Download 1000 images for PGQ/AIMET calibration — ImageNet preprocessing (224×224).

Sources (all permissive-licensed):
  pass        PASS dataset, CC BY 4.0, no people, direct URL list from Oxford [DEFAULT]
  open_images Open Images v7 validation, CC BY 4.0, direct URL list (requires Google Storage access)
  hf          HuggingFace imagenet-1k (requires HF_TOKEN + dataset access approval)
  local       Local ImageNet val directory

Output layout:
  <out_dir>/
    images/          preprocessed float32 .raw files (NCHW, ImageNet-normalized)
    batch_inputs.json  qaic-compile -aic-batch-json-input compatible index

Usage:
  # Default: Open Images v7 validation (CC BY 4.0, no auth required)
  python download_imagenet_calibration.py

  # PASS dataset (CC BY 4.0, no people, Zenodo download)
  python download_imagenet_calibration.py --source pass

  # HuggingFace imagenet-1k (requires HF_TOKEN)
  HF_TOKEN=hf_xxx python download_imagenet_calibration.py --source hf

  # Local ImageNet val directory
  python download_imagenet_calibration.py --source local --imagenet-dir /data/imagenet/val

  # Calibration/test split: 500 calib + 500 test from 1000 total
  python download_imagenet_calibration.py --n 1000
  # Then slice batch_inputs.json[0:500] → calib_inputs.json
  #           batch_inputs.json[500:]   → test_inputs.json

Run inside the container:
  /opt/aimet-env/bin/python download_imagenet_calibration.py [flags]
"""
import argparse
import csv
import json
import os
import sys
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# Requests verifies HTTPS certificates by default. For TLS-inspecting proxies,
# set REQUESTS_CA_BUNDLE to the proxy CA bundle.
_GET = requests.get

# ── Preprocessing ────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OPEN_IMAGES_VAL_CSV = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-boxable.csv"
)


def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """Resize → center-crop → normalize → NCHW float32."""
    scale = size * 256 // 224
    img = img.convert("RGB")
    w, h = img.size
    short = min(w, h)
    img = img.resize((int(w * scale / short), int(h * scale / short)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    arr = np.array(img, dtype=np.float32) / 255.0          # (H,W,3) 0-1
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD              # normalize
    return arr.transpose(2, 0, 1)[np.newaxis]               # (1,3,H,W)


def save_raw(arr: np.ndarray, path: Path):
    arr.astype(np.float32).tofile(path)


def _hf_stream(dataset_name: str, split: str, token: str | None = None):
    try:
        import datasets as hf_datasets
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        import datasets as hf_datasets
    return hf_datasets.load_dataset(
        dataset_name, split=split, streaming=True,
        trust_remote_code=True, token=token or True,
    )


# ── Source: PASS via URL list (CC BY 4.0, no auth) ──────────────────────────
PASS_URLS_TXT = "https://thor.robots.ox.ac.uk/pass/pass_urls.txt"


def download_pass(out_dir: Path, n: int, input_name: str, size: int):
    """Download PASS images from the official URL list — CC BY 4.0, no people, no auth."""
    print(f"Streaming PASS URL list (1.4 M entries) from {PASS_URLS_TXT} ...")
    r = _GET(PASS_URLS_TXT, stream=True, timeout=60)
    r.raise_for_status()

    paths, failed = [], 0
    for raw_line in r.iter_lines():
        if len(paths) >= n:
            break
        url = raw_line.decode("utf-8", errors="ignore").strip()
        if not url:
            continue
        try:
            resp = _GET(url, timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            arr = preprocess(img, size)
            i = len(paths)
            p = out_dir / "images" / f"{i:05d}.raw"
            save_raw(arr, p)
            paths.append(str(p))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n}")
        except Exception:
            failed += 1

    if failed:
        print(f"  ({failed} images skipped due to download errors)")
    return paths


# ── Source: Open Images v7 validation (CC BY 4.0, no auth) ───────────────────
def download_open_images(out_dir: Path, n: int, input_name: str, size: int):
    """Download Open Images v7 validation images via direct URL list."""
    print(f"Fetching Open Images v7 validation URL list → {OPEN_IMAGES_VAL_CSV}")
    r = _GET(OPEN_IMAGES_VAL_CSV, timeout=60)
    r.raise_for_status()
    rows = list(csv.DictReader(StringIO(r.text)))
    # fetch a buffer of extra rows to absorb failures
    fetch_rows = rows[: n * 2]
    print(f"  {len(rows)} total validation images; fetching first {len(fetch_rows)} URLs")

    paths, failed = [], 0
    for row in fetch_rows:
        if len(paths) >= n:
            break
        i = len(paths)
        url = row["OriginalURL"]
        try:
            resp = _GET(url, timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            arr = preprocess(img, size)
            p = out_dir / "images" / f"{i:05d}.raw"
            save_raw(arr, p)
            paths.append(str(p))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{n}")
        except Exception:
            failed += 1

    if failed:
        print(f"  ({failed} images skipped due to download errors)")
    if len(paths) < n:
        print(f"  WARNING: only collected {len(paths)}/{n} images; "
              f"increase fetch buffer or retry with --source pass")
    return paths


# ── Source: HuggingFace imagenet-1k (requires HF_TOKEN) ──────────────────────
def download_hf(out_dir: Path, n: int, input_name: str, size: int):
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("WARNING: HF_TOKEN not set; imagenet-1k requires dataset access approval.")
    print("Streaming imagenet-1k validation split from HuggingFace...")
    ds = _hf_stream("imagenet-1k", split="validation", token=token)
    paths = []
    for i, sample in enumerate(ds):
        if i >= n:
            break
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        arr = preprocess(img, size)
        p = out_dir / "images" / f"{i:05d}.raw"
        save_raw(arr, p)
        paths.append(str(p))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")
    return paths


# ── Source: local ImageNet val directory ─────────────────────────────────────
def download_local(out_dir: Path, n: int, input_name: str, size: int, val_dir: str):
    val_path = Path(val_dir)
    jpegs = sorted(val_path.rglob("*.JPEG")) + sorted(val_path.rglob("*.jpg"))
    jpegs = jpegs[:n]
    if not jpegs:
        raise RuntimeError(f"No JPEG images found under {val_dir}")
    print(f"Using {len(jpegs)} images from {val_dir}")
    paths = []
    for i, jp in enumerate(jpegs):
        img = Image.open(jp)
        arr = preprocess(img, size)
        p = out_dir / "images" / f"{i:05d}.raw"
        save_raw(arr, p)
        paths.append(str(p))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jpegs)}")
    return paths


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pass", "open_images", "hf", "local"],
                    default="pass",
                    help="pass=PASS CC-BY-4.0 no-auth Oxford URL list (default), "
                         "open_images=Open Images v7 CC-BY-4.0 (requires Google Storage access), "
                         "hf=HuggingFace imagenet-1k (needs HF_TOKEN), "
                         "local=local val dir")
    ap.add_argument("--imagenet-dir", default="/data/imagenet/val",
                    help="Path to ImageNet val directory (--source local)")
    ap.add_argument("--out-dir", default="./calibration_data/imagenet")
    ap.add_argument("--n", type=int, default=1000, help="Number of samples")
    ap.add_argument("--size", type=int, default=224, help="Crop size (pixels)")
    ap.add_argument("--input-name", default="pixel_values",
                    help="ONNX input tensor name used in batch_inputs.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    if args.source == "pass":
        paths = download_pass(out_dir, args.n, args.input_name, args.size)
    elif args.source == "open_images":
        paths = download_open_images(out_dir, args.n, args.input_name, args.size)
    elif args.source == "hf":
        paths = download_hf(out_dir, args.n, args.input_name, args.size)
    else:
        paths = download_local(out_dir, args.n, args.input_name, args.size, args.imagenet_dir)

    # Write qaic-compile / qaic-runner batch_inputs.json (IO-files format)
    # This format works with both:
    #   qaic-compile -aic-batch-json-input (dump-profile and load-profile passes)
    #   qaic-runner --aic-batch-json-input (inference / output capture)
    # Note: the {"inputs": [...]} format causes json.exception.type_error.305 — do NOT use it.
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
    print("Calibration/test split (500 calib + 500 test from 1000 total):")
    print("  python -c \"")
    print("  import json")
    print("  b = json.load(open('batch_inputs.json'))")
    print("  calib = {'IO-files': b['IO-files'][:500]}")
    print("  test  = {'IO-files': b['IO-files'][500:]}")
    print("  json.dump(calib, open('calib_inputs.json','w'))")
    print("  json.dump(test,  open('test_inputs.json','w'))\"")
    print()
    print("Use with AIMET:")
    print("  import numpy as np")
    print(f"  calibration_inputs = [{{'{args.input_name}': np.fromfile(p, dtype=np.float32)")
    print(f"                         .reshape(1,3,{args.size},{args.size})}}")
    print(f"                        for p in sorted(Path('{out_dir}/images').glob('*.raw'))[:1000]]")
    print()
    print("Use with qaic-compile PGQ:")
    print(f"  qaic-compile -m=model.onnx \\")
    print(f"    -aic-batch-json-input=calib_inputs.json \\")
    print(f"    -dump-profile=pgq_profile.yaml \\")
    print(f"    -quantization-calibration=KLMinimization \\")
    print(f"    -quantization-schema-activations=symmetric_with_uint8")


if __name__ == "__main__":
    main()
