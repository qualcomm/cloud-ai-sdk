####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import argparse
from pathlib import Path

from qaic_models.vision.ultralytics.common.export_base import (
    classify_artifact_paths,
    download_default_sample_image,
    export_ultralytics_classify_artifacts,
    generate_ultralytics_classify_triton_repo,
    qaic_compile_with_batch_symbol,
)

from .model import SUPPORTED_VARIANTS, _require_variant
from .postprocess import NUM_CLASSES, OUTPUT_NAME

INPUT_NAME = "images"
DEFAULT_SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
DEFAULT_SAMPLE_IMAGE_NAME = "bus.jpg"
LATENCY_QPC_SUBDIR = "qpc_latency"
THROUGHPUT_QPC_SUBDIR = "qpc_throughput"


def _download_default_sample_image(work_dir: Path) -> Path:
    return download_default_sample_image(work_dir, DEFAULT_SAMPLE_IMAGE_URL, DEFAULT_SAMPLE_IMAGE_NAME)


def _artifact_paths(work_dir: Path, variant: str) -> dict[str, Path]:
    return classify_artifact_paths(work_dir, variant, LATENCY_QPC_SUBDIR, THROUGHPUT_QPC_SUBDIR)


def export_onnx(weights: str, variant: str, work_dir: Path, sample_image: str) -> dict[str, Path]:
    return export_ultralytics_classify_artifacts(
        weights=weights,
        variant=variant,
        work_dir=work_dir,
        sample_image=sample_image,
        latency_qpc_subdir=LATENCY_QPC_SUBDIR,
        throughput_qpc_subdir=THROUGHPUT_QPC_SUBDIR,
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        num_classes=NUM_CLASSES,
    )


def compile_latency(paths: dict[str, Path], work_dir: Path, cores: int = 16) -> None:
    qaic_compile_with_batch_symbol(str(paths["final_onnx"].name), cores, paths["qpc_latency_dir"], cwd=work_dir, batch_size=1)


def compile_throughput(paths: dict[str, Path], work_dir: Path, cores: int = 4, mos: int = 1) -> None:
    qaic_compile_with_batch_symbol(str(paths["final_onnx"].name), cores, paths["qpc_throughput_dir"], cwd=work_dir, batch_size=4, mos=mos)


def generate_triton_repo(paths: dict[str, Path], repo_dir: Path, *, triton_device_id: int | None = None) -> dict[str, Path]:
    return generate_ultralytics_classify_triton_repo(
        paths,
        repo_dir,
        model_prefix="yolo11_classify",
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        num_classes=NUM_CLASSES,
        triton_device_id=triton_device_id,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export and compile YOLO11 classify artifacts.")
    parser.add_argument("--variant", default="yolo11s-cls", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--weights", default=None, help="Path/name of the .pt checkpoint (defaults to <variant>.pt)")
    parser.add_argument("--work-dir", required=True, help="Directory to write generated files into")
    parser.add_argument("--image", default=None, help=f"Sample image (default: download {DEFAULT_SAMPLE_IMAGE_NAME})")
    parser.add_argument("--compile-latency", action="store_true")
    parser.add_argument("--compile-throughput", action="store_true")
    parser.add_argument("--triton-model-repo", default=None, help="Optional Triton model repository output directory")
    parser.add_argument("--triton-device-id", type=int, default=None, help="QAIC QID for the generated Triton model")
    args = parser.parse_args(argv)

    variant = _require_variant(args.variant)
    weights = args.weights or f"{variant}.pt"
    work_dir = Path(args.work_dir).resolve()
    sample_image = args.image or str(_download_default_sample_image(work_dir))
    paths = export_onnx(weights, variant, work_dir, sample_image)
    print(f"[export] wrote {paths['final_onnx']}")
    if args.compile_latency:
        compile_latency(paths, work_dir)
        print(f"[export] compiled latency QPC to {paths['qpc_latency_dir']}")
    if args.compile_throughput:
        compile_throughput(paths, work_dir)
        print(f"[export] compiled throughput QPC to {paths['qpc_throughput_dir']}")
    if args.triton_model_repo:
        generated = generate_triton_repo(paths, Path(args.triton_model_repo).resolve(), triton_device_id=args.triton_device_id) or {}
        print(f"[export] wrote Triton model repository entries to {Path(args.triton_model_repo).resolve()}")
        for name, path in generated.items():
            print(f"[export] Triton {name}: {path}")


if __name__ == "__main__":
    main()
