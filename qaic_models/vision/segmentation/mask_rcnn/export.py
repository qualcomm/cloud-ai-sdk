####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from importlib import resources  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2 -- repo requires Python 3.10/3.12; the py3.7-compat rule does not apply.
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

import qaic_models.vision.segmentation.mask_rcnn as mrcnn_pkg
from qaic_models.vision.segmentation.mask_rcnn.model import BACKBONE_QPC_SUBDIR, ROI_HEADS_QPC_SUBDIR
from qaic_models.base.compile import qaic_compile_parallel
from qaic_models.base.onnx_transforms import ClipUndefinedInputsTransform, FP16ClipTransform, OnnxSimTransform, run_pipeline
from qaic_models.base.preprocess import ImageSize
from qaic_models.base.triton import TritonModelConfig, TritonModelEnsemble
from qaic_models.vision.segmentation.mask_rcnn.preprocess import MaskRCNNImage


DEFAULT_WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

# Split-model tensor names, shared between ONNX export and golden generation so the
# names always agree. ROI heads consume only p2..p5 (see roi_feed below), not p6.
FPN_OUTPUTS = ["p2", "p3", "p4", "p5", "p6"]
ROI_HEADS_OUTPUTS = ["cls_logits", "bbox_deltas", "mask_logits"]


@dataclass
class MaskRCNNExportArtifacts:
    work_dir: Path

    def __post_init__(self):
        self.input_raw = self.work_dir / "input.raw"

        self.raw_backbone = self.work_dir / "model_backbone_fpn.onnx"
        self.raw_rpn = self.work_dir / "model_rpn.onnx"
        self.raw_roi_heads = self.work_dir / "model_roi_heads.onnx"

        self.final_backbone = self.work_dir / "model_backbone_fpn_fp16ready.onnx"
        self.final_rpn = self.work_dir / "model_rpn_fp16ready.onnx"
        self.final_roi_heads = self.work_dir / "model_roi_heads_fp16ready.onnx"

        self.backbone_qpc_dir = self.work_dir / BACKBONE_QPC_SUBDIR
        self.roi_heads_qpc_dir = self.work_dir / ROI_HEADS_QPC_SUBDIR

        self.backbone_qpc = self.backbone_qpc_dir / "programqpc.bin"
        self.roi_heads_qpc = self.roi_heads_qpc_dir / "programqpc.bin"


def run(cmd: list[str], cwd: str | None = None, env: Optional[dict[str, str]] = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _setup_detectron2(work_dir: Path, args: argparse.Namespace) -> Path:
    if args.detectron2_src:
        d2_dir = Path(args.detectron2_src).expanduser().resolve()
        if not (d2_dir / "detectron2").is_dir():
            raise FileNotFoundError(f"--detectron2-src is not a Detectron2 checkout (no detectron2/ package dir): {d2_dir}")
        print(f"[export] Using prepared detectron2 checkout at {d2_dir}")
    else:
        d2_dir = work_dir / "detectron2"
        if d2_dir.exists():
            print(f"[export] Using existing detectron2 at {d2_dir}")
        else:
            run(["git", "clone", args.detectron2_repo, str(d2_dir)])
            run(["git", "checkout", args.detectron2_commit], cwd=str(d2_dir))
            run(["pip3", "install", "-e", ".", "--no-build-isolation"], cwd=str(d2_dir))

    d2_str = str(d2_dir)
    if d2_str not in sys.path:
        sys.path.insert(0, d2_str)
    return d2_dir


def _resolve_sample_image(work_dir: Path, args: argparse.Namespace) -> Path:
    if args.image:
        test_image = Path(args.image).expanduser().resolve()
        if not test_image.is_file():
            raise FileNotFoundError(f"--image sample input not found: {test_image}")
        return test_image
    return _download_sample_image(work_dir)


def _download_sample_image(work_dir: Path) -> Path:
    test_image = work_dir / "000000000139.jpg"
    if not test_image.exists():
        run([
            "wget",
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "-q",
            "-O",
            str(test_image),
        ])
    return test_image


def _write_raw_outputs(prefix: Path, outputs) -> None:
    for idx, output in enumerate(outputs if isinstance(outputs, (tuple, list)) else (outputs,)):
        arr = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
        np.ascontiguousarray(arr).tofile(f"{prefix}_{idx}.raw")


def _write_ort_outputs(prefix: Path, onnx_path: Path, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, feed)
    for idx, output in enumerate(outputs):
        np.ascontiguousarray(output).tofile(f"{prefix}_{idx}.raw")
    return outputs


def _run_onnx_pipeline(raw_path: Path, final_path: Path) -> None:
    run_pipeline(
        str(raw_path),
        str(final_path),
        [
            ClipUndefinedInputsTransform,
            OnnxSimTransform,
            FP16ClipTransform,
        ],
    )


def _print_op_counts(path: Path) -> None:
    try:
        import onnx
        from collections import Counter
    except Exception as exc:
        print(f"[export] Could not import ONNX to print op counts for {path}: {exc}")
        return
    model = onnx.load(str(path))
    counts = Counter(node.op_type for node in model.graph.node)
    print(f"[export] ONNX op counts for {path.name}:")
    for op, count in sorted(counts.items()):
        print(f"  {op}: {count}")


def _export_split_onnx(artifacts: MaskRCNNExportArtifacts, args: argparse.Namespace, d2_dir: Path) -> None:
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    from qaic_models.vision.segmentation.mask_rcnn.export_config import build_export_cfg
    from qaic_models.vision.segmentation.mask_rcnn.export_wrappers import (
        BackboneFPNExportWrapper,
        ROIHeadsExportWrapper,
        RPNExportWrapper,
    )
    from qaic_models.vision.segmentation.mask_rcnn.pytorch_transforms import (
        apply_mask_rcnn_export_overlays,
        undo_mask_rcnn_export_overlays,
    )

    if not os.path.isabs(args.config_file):
        args.config_file = str(d2_dir / args.config_file)

    cfg = build_export_cfg(args)
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    test_image = _resolve_sample_image(artifacts.work_dir, args)
    image_size = ImageSize(width=args.input_width, height=args.input_height)
    image_nchw_np = MaskRCNNImage.load(str(test_image), size=image_size)
    np.ascontiguousarray(image_nchw_np).tofile(artifacts.input_raw)
    image = torch.from_numpy(image_nchw_np)

    transforms = []
    try:
        torch_model, transforms, _ = apply_mask_rcnn_export_overlays(torch_model)

        backbone = BackboneFPNExportWrapper(torch_model, args.input_height, args.input_width)
        rpn = RPNExportWrapper(torch_model, args.input_height, args.input_width)
        roi_heads = ROIHeadsExportWrapper(torch_model, args.input_height, args.input_width)

        # --- framework goldens ---
        with torch.no_grad():
            p2, p3, p4, p5, p6 = backbone(image)
            _write_raw_outputs(artifacts.work_dir / "framework_output_backbone_fpn", (p2, p3, p4, p5, p6))
            proposals = rpn(p2, p3, p4, p5, p6)
            _write_raw_outputs(artifacts.work_dir / "framework_output_rpn", (proposals,))
            cls_logits, bbox_deltas, mask_logits = roi_heads(p2, p3, p4, p5, proposals)
            _write_raw_outputs(artifacts.work_dir / "framework_output_roi_heads", (cls_logits, bbox_deltas, mask_logits))

        # --- raw ONNX export ---
        raw_backbone = artifacts.raw_backbone
        torch.onnx.export(
            backbone,
            (image,),
            str(raw_backbone),
            opset_version=args.opset,
            input_names=["image"],
            output_names=FPN_OUTPUTS,
        )

        raw_rpn = artifacts.raw_rpn
        torch.onnx.export(
            rpn,
            (p2, p3, p4, p5, p6),
            str(raw_rpn),
            opset_version=args.opset,
            input_names=FPN_OUTPUTS,
            output_names=["proposals"],
        )

        raw_roi = artifacts.raw_roi_heads
        torch.onnx.export(
            roi_heads,
            (p2, p3, p4, p5, proposals),
            str(raw_roi),
            opset_version=args.opset,
            input_names=["p2", "p3", "p4", "p5", "proposals"],
            output_names=ROI_HEADS_OUTPUTS,
        )
    finally:
        undo_mask_rcnn_export_overlays(transforms)

    # --- fp16 transforms ---
    _run_onnx_pipeline(raw_backbone, artifacts.final_backbone)
    _run_onnx_pipeline(raw_rpn, artifacts.final_rpn)
    _run_onnx_pipeline(raw_roi, artifacts.final_roi_heads)

    _print_op_counts(artifacts.final_backbone)
    _print_op_counts(artifacts.final_rpn)
    _print_op_counts(artifacts.final_roi_heads)

    _write_cpu_goldens(artifacts, image_nchw_np)
    print("[export] Wrote split ONNX models and final fp16ready artifacts to:", artifacts.work_dir)


def _write_cpu_goldens(artifacts: MaskRCNNExportArtifacts, image_nchw_np: np.ndarray) -> None:
    """Run ONNX Runtime on the final fp16ready ONNX models and write CPU goldens."""
    backbone_outputs = _write_ort_outputs(artifacts.work_dir / "cpu_output_backbone_fpn", artifacts.final_backbone, {"image": image_nchw_np})
    rpn_feed = {name: tensor for name, tensor in zip(FPN_OUTPUTS, backbone_outputs)}
    (rpn_proposals,) = _write_ort_outputs(artifacts.work_dir / "cpu_output_rpn", artifacts.final_rpn, rpn_feed)
    roi_feed = {name: tensor for name, tensor in zip(["p2", "p3", "p4", "p5"], backbone_outputs[:4])}
    roi_feed["proposals"] = rpn_proposals
    _write_ort_outputs(artifacts.work_dir / "cpu_output_roi_heads", artifacts.final_roi_heads, roi_feed)


def export_onnx(artifacts: MaskRCNNExportArtifacts, args: argparse.Namespace) -> None:
    """Run Detectron2-based ONNX export pipeline without modifying Detectron2 source."""
    d2_dir = _setup_detectron2(artifacts.work_dir, args)
    _export_split_onnx(artifacts, args, d2_dir)


def export_qaic(artifacts: MaskRCNNExportArtifacts, args: argparse.Namespace) -> None:
    """Compile QAIC-bound ONNX models and run qaic-runner smoke tests."""
    print("[export] Compiling ONNX models for QAIC...")
    backbone_kwargs = {
        "model": artifacts.final_backbone.name,
        "cores": min(12, args.max_cores),
        "mos": 1,
        "ols": 4,
        "multicast_weights": True,
        "aic_binary_dir": artifacts.backbone_qpc_dir,
        "cwd": artifacts.work_dir,
    }
    roi_kwargs = {
        "model": artifacts.final_roi_heads.name,
        "cores": min(4, args.max_cores),
        "aic_binary_dir": artifacts.roi_heads_qpc_dir,
        "cwd": artifacts.work_dir,
    }

    # The two QPCs share no inputs or targets; qaic-compile writes only into
    # its own binary dir, so they compile concurrently. The ExitStack keeps
    # the quantize-path's temp yaml alive until both compiles finish.
    with ExitStack() as stack:
        if args.quantize:
            fp16_nodes_path = stack.enter_context(
                resources.as_file(resources.files(mrcnn_pkg) / "fp16_nodes_backbone_fpn.yaml")
            )
            backbone_kwargs["quantize"] = True
            backbone_kwargs["node_precision_info"] = fp16_nodes_path
        qaic_compile_parallel([backbone_kwargs, roi_kwargs])

    print("[export] Running qaic-runner tests...")
    run(["/opt/qti-aic/exec/qaic-runner", "-t", f"./{artifacts.backbone_qpc_dir.name}"], cwd=str(artifacts.work_dir))
    run(["/opt/qti-aic/exec/qaic-runner", "-t", f"./{artifacts.roi_heads_qpc_dir.name}"], cwd=str(artifacts.work_dir))
    print("[export] QAIC binaries and test runs complete.")


def _write_triton_model(config: TritonModelConfig, repo_root: Path) -> None:
    model_root = repo_root / config.name
    print(f"[triton] Populating {model_root} ({config.backend})")
    config.tofile(model_root)


_POSTPROCESS_STUB = (
    "# Generated by qaic_models Mask R-CNN export. Triton loads this as the python\n"
    "# backend model; the implementation lives in the installed qaic_models package.\n"
    "from qaic_models.vision.segmentation.mask_rcnn.triton_postprocess import TritonPythonModel\n"
)


def export_triton(artifacts: MaskRCNNExportArtifacts, args: argparse.Namespace) -> None:
    """Generate the ensemble model repository: only the models the ensemble serves.

    The ensemble runs backbone and ROI heads on the accelerator (qaic backend) and
    keeps the RPN on CPU (onnxruntime) for its memory footprint, so no onnxruntime
    variants of the other stages are emitted.
    """
    repo_root = Path(args.triton_model_repo)
    count = args.triton_model_count
    rpn_onnx = TritonModelConfig(
        model_path=artifacts.final_rpn,
        name="mask_rcnn_rpn_onnx",
        backend="onnxruntime",
        count=count,
    )
    backbone_qaic = TritonModelConfig(
        model_path=artifacts.backbone_qpc,
        name="mask_rcnn_backbone_fpn_qaic",
        backend="qaic",
        count=count,
        device_id=args.triton_device_id,
    )
    roi_heads_qaic = TritonModelConfig(
        model_path=artifacts.roi_heads_qpc,
        name="mask_rcnn_roi_heads_qaic",
        backend="qaic",
        count=count,
        device_id=args.triton_device_id,
    )
    postprocess = TritonModelConfig(
        model_source=_POSTPROCESS_STUB,
        name="mask_rcnn_postprocess",
        backend="python",
        count=count,
        inputs=[
            {"name": "cls_logits", "data_type": "TYPE_FP32", "dims": [200, 81]},
            {"name": "bbox_deltas", "data_type": "TYPE_FP32", "dims": [200, 320]},
            {"name": "mask_logits", "data_type": "TYPE_FP32", "dims": [200, 80, 28, 28]},
            {"name": "proposals", "data_type": "TYPE_FP32", "dims": [200, 4]},
        ],
        outputs=[
            {"name": "cls_results", "data_type": "TYPE_FP32", "dims": [-1, 2]},
            {"name": "cls_counts", "data_type": "TYPE_INT32", "dims": [1]},
            {"name": "boxes", "data_type": "TYPE_FP32", "dims": [-1, 4]},
        ],
        parameters={"topk": "20"},
    )

    for config in (rpn_onnx, backbone_qaic, roi_heads_qaic, postprocess):
        _write_triton_model(config, repo_root)

    ensemble = TritonModelEnsemble(
        name="mask_rcnn",
        models=[backbone_qaic, rpn_onnx, roi_heads_qaic, postprocess],
        max_batch_size=0,
    )
    ensemble_root = repo_root / ensemble.name
    print(f"[triton] Populating ensemble {ensemble_root}")
    ensemble.tofile(ensemble_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Detectron2 Mask R-CNN model to ONNX and prepare QAIC artifacts.")
    parser.add_argument("--work-dir", default="build/mask_rcnn", help="Working directory to clone detectron2 and write generated artifacts (default: build/mask_rcnn)")
    parser.add_argument("--compile", action="store_true", help="Compile the backbone and ROI heads QPC binaries after the ONNX export (requires the Cloud AI SDK and device access)")
    parser.add_argument("--quantize", action="store_true", help="If set, export mixed-precision Int8 quantized model binaries instead of pure FP16")
    parser.add_argument("--image", default=None, help="Sample input image for input.raw and goldens (default: download 000000000139.jpg into the work dir)")
    parser.add_argument("--detectron2-repo", default="https://github.com/facebookresearch/detectron2.git", help="Git URL for Detectron2 repository")
    parser.add_argument("--detectron2-commit", default="48b598b4f61fbb24182a69b521b2a0ba3252b842", help="Detectron2 commit hash to checkout")
    parser.add_argument("--detectron2-src", default=None, help="Prepared Detectron2 checkout to import instead of cloning into the work dir; the environment must have already installed it (e.g. pip3 install -e <checkout> --no-build-isolation)")
    parser.add_argument("--config-file", default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", help="Detectron2 config file, relative to detectron2 root unless absolute")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Detectron2 weights URI or path")
    parser.add_argument("--input-height", type=int, default=800, help="Fixed export/runtime input height")
    parser.add_argument("--input-width", type=int, default=1216, help="Fixed export/runtime input width")
    parser.add_argument("--proposal-count", type=int, default=200, help="Fixed RPN proposal count")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument("--triton-model-repo", type=str, default=None, help="Generate the Triton ensemble model repository at the given path (run after the stages are compiled)")
    parser.add_argument("--triton-model-count", type=int, default=1, help="Number of instances per Triton model")
    parser.add_argument("--triton-device-id", type=int, default=None, help="QAIC QID for generated Triton models")
    parser.add_argument("--max-cores", type=int, default=16, help="Maximum number of NSP cores to use for model compilation")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    artifacts = MaskRCNNExportArtifacts(work_dir)
    export_onnx(artifacts, args)
    if args.compile:
        export_qaic(artifacts, args)
    if args.triton_model_repo:
        export_triton(artifacts, args)


if __name__ == "__main__":
    main()
