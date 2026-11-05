####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from qaic_models.base.compile import qaic_compile
from qaic_models.vision.ultralytics.common.export_base import (
    generate_ultralytics_raw_triton_repo,
    prepare_ultralytics_weights_for_export,
    run_ultralytics_detection_transform_pipeline,
)
from qaic_models.vision.ultralytics.common.preprocess import ULTRALYTICS_DETECT_IMAGE_SIZE, load_ultralytics_letterbox_image

from .model import SUPPORTED_VARIANTS, _require_variant
from .postprocess import OUTPUT_CHANNELS, OUTPUT_NAME, OUTPUT_SLOTS

INPUT_NAME = "images"
DEFAULT_SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
DEFAULT_SAMPLE_IMAGE_NAME = "bus.jpg"


def _download_default_sample_image(work_dir: Path) -> Path:
    image_path = work_dir / DEFAULT_SAMPLE_IMAGE_NAME
    if not image_path.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DEFAULT_SAMPLE_IMAGE_URL, image_path)
    return image_path


def _artifact_paths(work_dir: Path, variant: str) -> dict[str, Path]:
    return {
        "input_raw": work_dir / "input.raw",
        "framework_output": work_dir / "framework_output_0.raw",
        "cpu_output": work_dir / "cpu_output.raw",
        "raw_onnx": work_dir / f"{variant}.onnx",
        "final_onnx": work_dir / f"{variant}_fp16ready.onnx",
        "qpc_latency_dir": work_dir / "qpc_latency",
        "qpc_throughput_dir": work_dir / "qpc_throughput",
    }


def validate_onnx_output_contract(onnx_path: Path, expected_output_name: str = OUTPUT_NAME) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    outputs = list(model.graph.output)
    names = [output.name for output in outputs]
    if names != [expected_output_name]:
        raise ValueError(f"expected ONNX outputs {[expected_output_name]}, got {names}")
    dims = outputs[0].type.tensor_type.shape.dim
    dim_values = [dim.dim_value if dim.dim_value else dim.dim_param for dim in dims]
    if len(dim_values) != 3:
        raise ValueError(f"expected rank 3 output [1, {OUTPUT_SLOTS}, {OUTPUT_CHANNELS}], got {dim_values}")
    if isinstance(dim_values[1], int) and dim_values[1] != OUTPUT_SLOTS:
        raise ValueError(f"expected output slot dimension {OUTPUT_SLOTS}, got {dim_values[1]}")
    if isinstance(dim_values[2], int) and dim_values[2] != OUTPUT_CHANNELS:
        raise ValueError(f"expected output channel dimension {OUTPUT_CHANNELS}, got {dim_values[2]}")


def export_onnx(weights: str, variant: str, work_dir: Path, sample_image: str, target_batch: int = 1) -> dict[str, Path]:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(work_dir, variant)

    image, _letterbox = load_ultralytics_letterbox_image(sample_image, size=ULTRALYTICS_DETECT_IMAGE_SIZE)
    if target_batch != 1:
        image = np.ascontiguousarray(np.repeat(image, target_batch, axis=0))
    image.tofile(paths["input_raw"])

    from ultralytics import YOLO

    weights_for_export = prepare_ultralytics_weights_for_export(weights, work_dir)

    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        yolo = YOLO(weights_for_export)
        exported = yolo.export(
            format="onnx",
            simplify=False,
            dynamic=False,
            opset=17,
            imgsz=ULTRALYTICS_DETECT_IMAGE_SIZE.width,
            batch=target_batch,
            nms=False,
        )
        exported_path = Path(exported)
        if exported_path.resolve() != paths["raw_onnx"].resolve():
            shutil.move(str(exported_path), paths["raw_onnx"])
    finally:
        os.chdir(cwd)

    validate_onnx_output_contract(paths["raw_onnx"])
    run_ultralytics_detection_transform_pipeline(paths["raw_onnx"], paths["final_onnx"])
    validate_onnx_output_contract(paths["final_onnx"])
    _write_cpu_golden(paths["final_onnx"], image, paths["cpu_output"])
    return paths


def _write_cpu_golden(onnx_path: Path, image: np.ndarray, output_path: Path) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (output0,) = sess.run(None, {INPUT_NAME: image})
    np.ascontiguousarray(output0).tofile(output_path)


def compile_latency(paths: dict[str, Path], work_dir: Path, cores: int = 16, mos: int = 1) -> None:
    qaic_compile(str(paths["final_onnx"].name), cores, paths["qpc_latency_dir"], cwd=work_dir, mos=mos)


def compile_throughput(paths: dict[str, Path], work_dir: Path, cores: int = 4) -> None:
    qaic_compile(str(paths["final_onnx"].name), cores, paths["qpc_throughput_dir"], cwd=work_dir)


def generate_triton_repo(
    paths: dict[str, Path],
    repo_dir: Path,
    *,
    target_batch: int = 1,
    triton_device_id: int | None = None,
    triton_protobuf=None,
) -> dict[str, Path]:
    # --target-batch flows into the ONNX/QPC shapes, so the served shapes in
    # the pbtxt must follow it — a hardcoded [1, ...] config would lie about
    # what a batch-N QPC serves.
    return generate_ultralytics_raw_triton_repo(
        paths,
        repo_dir,
        model_prefix="yolo26_detect",
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        input_shape=[target_batch, 3, 640, 640],
        output_shape=[target_batch, OUTPUT_SLOTS, OUTPUT_CHANNELS],
        triton_device_id=triton_device_id,
        triton_protobuf=triton_protobuf,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and compile YOLO26 detect artifacts.")
    parser.add_argument("--variant", default="yolo26m", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--weights", default=None, help="Path/name of the .pt checkpoint (defaults to <variant>.pt)")
    parser.add_argument("--work-dir", required=True, help="Directory to write generated files into")
    parser.add_argument("--image", default=None, help=f"Sample image (default: download {DEFAULT_SAMPLE_IMAGE_NAME})")
    parser.add_argument("--target-batch", type=int, default=1, help="Static export batch size; batch 1 is recommended")
    parser.add_argument("--compile-latency", action="store_true")
    parser.add_argument("--compile-throughput", action="store_true")
    parser.add_argument("--triton-model-repo", default=None, help="Optional Triton model repository output directory")
    parser.add_argument("--triton-device-id", type=int, default=None, help="QAIC QID for the generated Triton model")
    args = parser.parse_args()

    variant = _require_variant(args.variant)
    weights = args.weights or f"{variant}.pt"
    work_dir = Path(args.work_dir).resolve()
    sample_image = args.image or str(_download_default_sample_image(work_dir))
    paths = export_onnx(weights, variant, work_dir, sample_image, target_batch=args.target_batch)
    print(f"[export] wrote {paths['final_onnx']}")
    if args.compile_latency:
        compile_latency(paths, work_dir)
        print(f"[export] compiled latency QPC to {paths['qpc_latency_dir']}")
    if args.compile_throughput:
        compile_throughput(paths, work_dir)
        print(f"[export] compiled throughput QPC to {paths['qpc_throughput_dir']}")
    if args.triton_model_repo:
        generated = generate_triton_repo(
            paths,
            Path(args.triton_model_repo).resolve(),
            target_batch=args.target_batch,
            triton_device_id=args.triton_device_id,
        )
        print(f"[export] wrote Triton model repository entries to {Path(args.triton_model_repo).resolve()}")
        for name, path in generated.items():
            print(f"[export] Triton {name}: {path}")


if __name__ == "__main__":
    main()
