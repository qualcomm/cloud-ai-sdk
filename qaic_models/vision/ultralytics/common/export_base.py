####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
"""Shared Ultralytics export scaffolding: PyTorch load, ONNX export, and QAIC compile
with a dynamic batch-size symbol. Family task export.py CLIs (yolo11/detect/export.py,
future segment/classify/...) call these functions instead of reimplementing them.

Keep heavy imports (torch, ultralytics, onnx, onnxruntime) inside functions so importing
this module — or any recipe model.py that imports from the same package — stays light.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from qaic_models.base.compile import build_compile_cmd
from qaic_models.base.triton import TritonModelConfig, TritonModelEnsemble
from qaic_models.base.onnx_transforms import (
    ClipUndefinedInputsTransform,
    FP16ClipTransform,
    GlobalAveragePoolFusionTransform,
    OnnxSimTransform,
    run_pipeline,
)


def prepare_ultralytics_weights_for_export(weights: str, work_dir: Path) -> str:
    """Copy local weights into work_dir so Ultralytics export writes beside work_dir."""
    weights_path = Path(weights)
    if not weights_path.exists():
        if weights_path.is_absolute() or weights_path.parent != Path("."):
            raise FileNotFoundError(f"weights file not found: {weights}")
        return weights
    weights_for_export = work_dir / weights_path.name
    if weights_for_export.resolve() != weights_path.resolve():
        shutil.copy2(weights_path, weights_for_export)
    return weights_for_export.name


def export_ultralytics_onnx(
    weights: str,
    output_path: Path,
    *,
    input_size: int,
    input_names: list[str],
    output_names: list[str],
    opset: int = 16,
) -> None:
    """Load an Ultralytics `.pt` checkpoint and export it to ONNX with a
    symbolic batch axis on every input/output listed.

    Mirrors ultralytics.engine.exporter.Exporter's pre-export module patching:
    in eval/training mode, Detect.forward() returns (y, preds) — a tuple of the
    concatenated prediction tensor plus per-scale intermediates used for loss
    computation — which torch.onnx.export would trace as extra graph outputs.
    Setting `m.export = True` on every Detect-family submodule makes forward()
    return just `y`, matching the single output0 tensor validated in onboarding.
    """
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect

    model = YOLO(weights).model
    model.eval()
    model.float()
    model = model.fuse()
    for m in model.modules():
        if isinstance(m, Detect):  # includes Detect subclasses: Segment, Pose, OBB, etc.
            m.dynamic = False
            m.export = True
            m.format = "onnx"

    dummy_input = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    dynamic_axes = {name: {0: "batch_size"} for name in (*input_names, *output_names)}
    export_kwargs = {
        "opset_version": opset,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        **export_kwargs,
    )


def export_ultralytics_classify_onnx(
    weights: str,
    output_path: Path,
    sample_input,
    *,
    input_names: list[str],
    output_names: list[str],
    opset: int = 18,
) -> None:
    """Load an Ultralytics classification checkpoint and export a single probs output."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Classify

    model = YOLO(weights).model
    model.eval()
    for module in model.modules():
        if isinstance(module, Classify):
            module.export = True

    dynamic_axes = {name: {0: "batch_size"} for name in (*input_names, *output_names)}
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        torch.from_numpy(sample_input),
        str(output_path),
        **export_kwargs,
    )


def run_fp16_transform_pipeline(raw_onnx: Path, final_onnx: Path) -> None:
    """Run the shared OnnxSim + FP16-clip pipeline used by YOLO11 detect."""
    run_pipeline(str(raw_onnx), str(final_onnx), [OnnxSimTransform, FP16ClipTransform])


def run_ultralytics_detection_transform_pipeline(raw_onnx: Path, final_onnx: Path) -> None:
    run_pipeline(str(raw_onnx), str(final_onnx), [ClipUndefinedInputsTransform, OnnxSimTransform, FP16ClipTransform])


def run_ultralytics_classify_transform_pipeline(raw_onnx: Path, final_onnx: Path) -> None:
    run_pipeline(
        str(raw_onnx),
        str(final_onnx),
        [ClipUndefinedInputsTransform, OnnxSimTransform, GlobalAveragePoolFusionTransform, FP16ClipTransform],
    )


def download_default_sample_image(work_dir: Path, url: str, name: str) -> Path:
    image_path = work_dir / name
    if not image_path.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, image_path)
    return image_path


def classify_artifact_paths(work_dir: Path, variant: str, latency_qpc_subdir: str, throughput_qpc_subdir: str) -> dict[str, Path]:
    return {
        "input_raw": work_dir / "input.raw",
        "cpu_output": work_dir / "cpu_output.raw",
        "raw_onnx": work_dir / f"{variant}.onnx",
        "final_onnx": work_dir / f"{variant}_fp16ready.onnx",
        "qpc_latency_dir": work_dir / latency_qpc_subdir,
        "qpc_throughput_dir": work_dir / throughput_qpc_subdir,
    }


def validate_classify_onnx_output_contract(
    onnx_path: Path,
    *,
    output_name: str,
    num_classes: int,
    expected_output_names: Sequence[str] | None = None,
) -> None:
    import onnx

    expected_names = list(expected_output_names or (output_name,))
    model = onnx.load(str(onnx_path))
    outputs = list(model.graph.output)
    names = [output.name for output in outputs]
    if names != expected_names:
        raise ValueError(f"expected ONNX outputs {expected_names}, got {names}")
    dims = outputs[0].type.tensor_type.shape.dim
    dim_values = [dim.dim_value if dim.dim_value else dim.dim_param for dim in dims]
    if len(dim_values) != 2:
        raise ValueError(f"expected rank 2 output [batch, {num_classes}], got {dim_values}")
    if isinstance(dim_values[1], int) and dim_values[1] != num_classes:
        raise ValueError(f"expected class dimension {num_classes}, got {dim_values[1]}")


def write_classify_cpu_golden(onnx_path: Path, image: np.ndarray, output_path: Path, input_name: str) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (probs,) = sess.run(None, {input_name: image})
    np.ascontiguousarray(probs).tofile(output_path)


def export_ultralytics_classify_artifacts(
    *,
    weights: str,
    variant: str,
    work_dir: Path,
    sample_image: str,
    latency_qpc_subdir: str,
    throughput_qpc_subdir: str,
    input_name: str,
    output_name: str,
    num_classes: int,
) -> dict[str, Path]:
    from qaic_models.vision.ultralytics.common.preprocess import load_ultralytics_classify_image

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = classify_artifact_paths(work_dir, variant, latency_qpc_subdir, throughput_qpc_subdir)
    image, _info = load_ultralytics_classify_image(sample_image)
    image.tofile(paths["input_raw"])
    weights_for_export = prepare_ultralytics_weights_for_export(weights, work_dir)
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        export_ultralytics_classify_onnx(weights_for_export, paths["raw_onnx"], image, input_names=[input_name], output_names=[output_name])
    finally:
        os.chdir(cwd)
    validate_classify_onnx_output_contract(paths["raw_onnx"], output_name=output_name, num_classes=num_classes)
    run_ultralytics_classify_transform_pipeline(paths["raw_onnx"], paths["final_onnx"])
    validate_classify_onnx_output_contract(paths["final_onnx"], output_name=output_name, num_classes=num_classes)
    write_classify_cpu_golden(paths["final_onnx"], image, paths["cpu_output"], input_name)
    return paths


def _require_qpc_program(qpc_dir: Path) -> Path:
    programqpc = Path(qpc_dir) / "programqpc.bin"
    if not programqpc.is_file():
        raise FileNotFoundError(f"Triton QPC model requires {programqpc}")
    return programqpc


def _classify_topk_model_source() -> str:
    return '''\
import json
import numpy as np

from qaic_models.vision.ultralytics.common.postprocess import postprocess_classification_probs


class TritonPythonModel:
    def initialize(self, args):
        import triton_python_backend_utils as pb_utils

        config = json.loads(args["model_config"])
        parameters = config.get("parameters", {})
        self.topk = int(parameters.get("topk", {}).get("string_value", "5"))
        self.labels = None
        self.pb_utils = pb_utils

    def execute(self, requests):
        responses = []
        for request in requests:
            probs = self.pb_utils.get_input_tensor_by_name(request, "probs").as_numpy()
            prediction = postprocess_classification_probs(probs, self.labels, topk=self.topk)
            topk = prediction["topk"]
            classes = np.asarray([item["class_id"] for item in topk], dtype=np.int32)[None, :]
            scores = np.asarray([item["score"] for item in topk], dtype=np.float32)[None, :]
            labels = np.asarray([item["label"].encode("utf-8") for item in topk], dtype=object)[None, :]
            responses.append(
                self.pb_utils.InferenceResponse(
                    output_tensors=[
                        self.pb_utils.Tensor("topk_classes", classes),
                        self.pb_utils.Tensor("topk_scores", scores),
                        self.pb_utils.Tensor("topk_labels", labels),
                    ]
                )
            )
        return responses
'''


def _onnx_graph_dims(onnx_path: Path) -> tuple[list[int], list[int]]:
    """Input/output dims as the ONNX graph declares them, symbolic axes as -1.

    The onnxruntime backend rejects a concrete config dim on an axis the model
    declares symbolically (dynamic_axes batch export → string dim_param), so
    the ONNX model's config must mirror the graph's own dynamism: static axes
    keep their value, symbolic axes become -1.
    """
    import onnx

    model = onnx.load(str(onnx_path))

    def _dims(tensor) -> list[int]:
        return [d.dim_value if d.dim_value else -1 for d in tensor.type.tensor_type.shape.dim]

    return _dims(model.graph.input[0]), _dims(model.graph.output[0])


def generate_ultralytics_raw_triton_repo(
    paths: dict[str, Path],
    repo_dir: Path,
    *,
    model_prefix: str,
    input_name: str,
    output_name: str,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    triton_device_id: int | None = None,
    triton_protobuf=None,
) -> dict[str, Path]:
    repo_dir = Path(repo_dir).resolve()
    final_onnx = paths["final_onnx"]
    programqpc = _require_qpc_program(paths["qpc_latency_dir"])
    onnx_input_dims, onnx_output_dims = _onnx_graph_dims(final_onnx)
    # The QPC is compiled with a fixed batch, so its config keeps the concrete
    # compiled shapes; only the ONNX config tracks the graph's dynamic axes.
    onnx_cfg = TritonModelConfig(
        model_path=final_onnx,
        name=f"{model_prefix}_onnx",
        backend="onnxruntime",
        inputs=[{"name": input_name, "data_type": "TYPE_FP32", "dims": onnx_input_dims}],
        outputs=[{"name": output_name, "data_type": "TYPE_FP32", "dims": onnx_output_dims}],
    )
    qpc_cfg = TritonModelConfig(
        model_path=programqpc,
        name=f"{model_prefix}_qpc",
        backend="qaic",
        inputs=[{"name": input_name, "data_type": "TYPE_FP32", "dims": list(input_shape)}],
        outputs=[{"name": output_name, "data_type": "TYPE_FP32", "dims": list(output_shape)}],
        device_id=triton_device_id,
    )
    onnx_root = repo_dir / onnx_cfg.name
    qpc_root = repo_dir / qpc_cfg.name
    onnx_cfg.tofile(onnx_root, triton_protobuf=triton_protobuf)
    qpc_cfg.tofile(qpc_root, triton_protobuf=triton_protobuf)
    return {"onnx_model": onnx_root, "qpc_model": qpc_root}


def generate_ultralytics_classify_triton_repo(
    paths: dict[str, Path],
    repo_dir: Path,
    *,
    model_prefix: str,
    input_name: str,
    output_name: str,
    num_classes: int,
    topk: int = 5,
    triton_device_id: int | None = None,
    triton_protobuf=None,
) -> dict[str, Path]:
    generated = generate_ultralytics_raw_triton_repo(
        paths,
        repo_dir,
        model_prefix=model_prefix,
        input_name=input_name,
        output_name=output_name,
        input_shape=[1, 3, 224, 224],
        output_shape=[1, num_classes],
        triton_device_id=triton_device_id,
        triton_protobuf=triton_protobuf,
    )
    repo_dir = Path(repo_dir).resolve()
    topk_cfg = TritonModelConfig(
        name=f"{model_prefix}_topk",
        backend="python",
        model_source=_classify_topk_model_source(),
        inputs=[{"name": output_name, "data_type": "TYPE_FP32", "dims": [1, num_classes]}],
            outputs=[
            {"name": "topk_classes", "data_type": "TYPE_INT32", "dims": [1, topk]},
            {"name": "topk_scores", "data_type": "TYPE_FP32", "dims": [1, topk]},
            {"name": "topk_labels", "data_type": "TYPE_STRING", "dims": [1, topk]},
        ],
        parameters={"topk": str(topk)},
    )
    topk_root = repo_dir / topk_cfg.name
    topk_cfg.tofile(topk_root, triton_protobuf=triton_protobuf)
    onnx_ensemble_root = repo_dir / f"{model_prefix}_onnx_ensemble"
    TritonModelEnsemble(f"{model_prefix}_onnx_ensemble", [
        TritonModelConfig(
            model_path=paths["final_onnx"],
            name=f"{model_prefix}_onnx",
            backend="onnxruntime",
            inputs=[{"name": input_name, "data_type": "TYPE_FP32", "dims": [1, 3, 224, 224]}],
            outputs=[{"name": output_name, "data_type": "TYPE_FP32", "dims": [1, num_classes]}],
            device_id=triton_device_id,
        ),
        topk_cfg,
    ]).tofile(onnx_ensemble_root, triton_protobuf=triton_protobuf)
    qpc_ensemble_root = repo_dir / f"{model_prefix}_qpc_ensemble"
    TritonModelEnsemble(f"{model_prefix}_qpc_ensemble", [
        TritonModelConfig(
            model_path=_require_qpc_program(paths["qpc_latency_dir"]),
            name=f"{model_prefix}_qpc",
            backend="qaic",
            inputs=[{"name": input_name, "data_type": "TYPE_FP32", "dims": [1, 3, 224, 224]}],
            outputs=[{"name": output_name, "data_type": "TYPE_FP32", "dims": [1, num_classes]}],
        ),
        topk_cfg,
    ]).tofile(qpc_ensemble_root, triton_protobuf=triton_protobuf)
    generated.update({
        "postprocess_model": topk_root,
        "onnx_ensemble_model": onnx_ensemble_root,
        "qpc_ensemble_model": qpc_ensemble_root,
    })
    return generated


def qaic_compile_with_batch_symbol(
    model: str,
    cores: int,
    aic_binary_dir,
    *,
    cwd,
    batch_size: int = 1,
    mos: Optional[int] = None,
) -> None:
    """Compile with a fixed batch-size symbol bound to the model's dynamic batch
    axis. qaic_models.base.compile.build_compile_cmd() has no extension point for
    -onnx-define-symbol, so the flag is appended here rather than duplicating the
    rest of build_compile_cmd()'s flag-building logic."""
    aic_binary_dir = Path(aic_binary_dir)
    if aic_binary_dir.exists():
        subprocess.run(["rm", "-rf", str(aic_binary_dir)], check=True)
    cmd = build_compile_cmd(model, cores, aic_binary_dir, mos=mos)
    cmd.append(f"-onnx-define-symbol=batch_size,{batch_size}")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)
