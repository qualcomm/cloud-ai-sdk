####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
"""Export/compile CLI for YOLO11 detect. Generates ONNX and QPC artifacts into a
caller-supplied --work-dir; never reads or writes committed model binaries.

Usage:
    python -m qaic_models.vision.ultralytics.yolo11.detect.export \\
        --variant yolo11m --work-dir build/yolo11 --image sample.jpg \\
        --compile-latency --compile-throughput

If --image is omitted, the Ultralytics repo's bundled sample image (zidane.jpg)
is downloaded into --work-dir and used instead.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

from qaic_models.vision.ultralytics.common.export_base import (
    export_ultralytics_onnx,
    generate_ultralytics_raw_triton_repo,
    qaic_compile_with_batch_symbol,
    run_fp16_transform_pipeline,
)

from qaic_models.vision.ultralytics.common.preprocess import (
    ULTRALYTICS_DETECT_IMAGE_SIZE,
    load_ultralytics_letterbox_image,
)

from .model import SUPPORTED_VARIANTS, _require_variant
from .postprocess import OUTPUT_CHANNELS, OUTPUT_NAME

INPUT_NAME = "images"

DEFAULT_SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"
DEFAULT_SAMPLE_IMAGE_NAME = "zidane.jpg"


def _download_default_sample_image(work_dir: Path) -> Path:
    """Download the Ultralytics repo's bundled sample image into work_dir if not already present."""
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


def validate_onnx_output_contract(
    onnx_path: Path,
    *,
    expected_output_names: Sequence[str] = (OUTPUT_NAME,),
    expected_output_channels: int = OUTPUT_CHANNELS,
) -> None:
    """Verify the exported deployable ONNX exposes only the validated outputs.

    This catches exports that accidentally include training, debug, intermediate,
    or auxiliary outputs before those artifacts reach CPU golden generation or
    qaic-compile.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    graph_outputs = list(model.graph.output)
    output_names = [output.name for output in graph_outputs]
    expected_names = list(expected_output_names)
    if output_names != expected_names:
        raise ValueError(f"expected ONNX outputs {expected_names}, got {output_names}")

    for output in graph_outputs:
        dims = output.type.tensor_type.shape.dim
        dim_values = [dim.dim_value if dim.dim_value else dim.dim_param for dim in dims]
        if len(dim_values) != 3:
            raise ValueError(f"expected {output.name!r} to be rank 3 [batch, channels, anchors], got {dim_values}")
        channel_dim = dim_values[1]
        if isinstance(channel_dim, int) and channel_dim != expected_output_channels:
            raise ValueError(
                f"expected {output.name!r} channel dimension {expected_output_channels}, got {channel_dim}"
            )


def export_onnx(weights: str, variant: str, work_dir: Path, sample_image: str) -> dict[str, Path]:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(work_dir, variant)

    image, _letterbox = load_ultralytics_letterbox_image(sample_image, size=ULTRALYTICS_DETECT_IMAGE_SIZE)
    image.tofile(paths["input_raw"])

    export_ultralytics_onnx(
        weights,
        paths["raw_onnx"],
        input_size=ULTRALYTICS_DETECT_IMAGE_SIZE.wh[0],
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
    )
    validate_onnx_output_contract(paths["raw_onnx"])
    run_fp16_transform_pipeline(paths["raw_onnx"], paths["final_onnx"])
    validate_onnx_output_contract(paths["final_onnx"])

    _write_cpu_golden(paths["final_onnx"], image, paths["cpu_output"])
    return paths


def _write_cpu_golden(onnx_path: Path, image: np.ndarray, output_path: Path) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (output0,) = sess.run(None, {INPUT_NAME: image})
    np.ascontiguousarray(output0).tofile(output_path)


def compile_latency(paths: dict[str, Path], work_dir: Path, cores: int = 16, mos: int = 1) -> None:
    qaic_compile_with_batch_symbol(
        str(paths["final_onnx"].name),
        cores,
        paths["qpc_latency_dir"],
        cwd=work_dir,
        batch_size=1,
        mos=mos,
    )


def compile_throughput(paths: dict[str, Path], work_dir: Path, cores: int = 4) -> None:
    qaic_compile_with_batch_symbol(
        str(paths["final_onnx"].name),
        cores,
        paths["qpc_throughput_dir"],
        cwd=work_dir,
        batch_size=1,
    )


def generate_triton_repo(paths: dict[str, Path], repo_dir: Path, *, triton_device_id: int | None = None, triton_protobuf=None) -> dict[str, Path]:
    return generate_ultralytics_raw_triton_repo(
        paths,
        repo_dir,
        model_prefix="yolo11_detect",
        input_name=INPUT_NAME,
        output_name=OUTPUT_NAME,
        input_shape=[1, 3, 640, 640],
        output_shape=[1, OUTPUT_CHANNELS, 8400],
        triton_device_id=triton_device_id,
        triton_protobuf=triton_protobuf,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="yolo11m", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--weights", default=None, help="Path/name of the .pt checkpoint (defaults to <variant>.pt)")
    parser.add_argument("--work-dir", required=True, help="Directory to write all generated artifacts into")
    parser.add_argument(
        "--image",
        default=None,
        help=f"Sample image used to build input.raw and CPU golden "
             f"(default: download {DEFAULT_SAMPLE_IMAGE_NAME} from the Ultralytics repo into --work-dir)",
    )
    parser.add_argument("--compile-latency", action="store_true", help="Compile the latency-tuned QPC (16 cores, -mos=1)")
    parser.add_argument("--compile-throughput", action="store_true", help="Compile the throughput-tuned QPC (4 cores)")
    parser.add_argument("--triton-model-repo", default=None, help="Optional Triton model repository output directory")
    parser.add_argument("--triton-device-id", type=int, default=None, help="QAIC QID for the generated Triton model")
    args = parser.parse_args()

    variant = _require_variant(args.variant)
    weights = args.weights or f"{variant}.pt"
    # Resolve to an absolute path: qaic_compile_with_batch_symbol() runs qaic-compile
    # with cwd=work_dir while also passing -aic-binary-dir=<work_dir>/qpc_*, and a
    # relative work_dir would make that flag resolve relative to itself.
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
        generated = generate_triton_repo(paths, Path(args.triton_model_repo).resolve(), triton_device_id=args.triton_device_id)
        print(f"[export] wrote Triton model repository entries to {Path(args.triton_model_repo).resolve()}")
        for name, path in generated.items():
            print(f"[export] Triton {name}: {path}")


if __name__ == "__main__":
    main()
