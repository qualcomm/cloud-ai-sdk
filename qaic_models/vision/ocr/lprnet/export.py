####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
"""Export/compile CLI for LPRNet. Generates ONNX and QPC artifacts into a
caller-supplied --work-dir; never reads or writes committed model binaries.

Usage:
    python -m qaic_models.vision.ocr.lprnet.export \\
        --work-dir build/lprnet --image plate.jpg

Both QPC builds (latency and throughput) are always compiled. If --image is
omitted, the first sorted image of the source repository's data/test/
directory is used (京PL3N67.jpg — the validated golden plate). If --repo is
omitted, the LPRNet_Pytorch source repository is cloned into --work-dir/repo.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Tuple

from qaic_models.base.compile import qaic_compile
from qaic_models.base.onnx_transforms import (
    BaseOnnxTransform,
    FP16ClipTransform,
    OnnxSimTransform,
    ReduceMeanAxesInputToAttrTransform,
    run_pipeline,
)
from qaic_models.base.triton import TritonModelConfig, TritonModelEnsemble

from .model import FINAL_ONNX_NAME, LATENCY_QPC_SUBDIR, SUPPORTED_VARIANTS, THROUGHPUT_QPC_SUBDIR, _require_variant
from .postprocess import NUM_CLASSES, NUM_TIMESTEPS, OUTPUT_NAME
from .preprocess import INPUT_NAME, INPUT_SHAPE, preprocess_plate_image

# Authoritative export opset from the onboarded export_lprnet_onnx.py.
OPSET = 16
LPR_MAX_LEN = 8
DROPOUT_RATE = 0.0

REPO_URL = "https://github.com/sirius-ai/LPRNet_Pytorch"
DEFAULT_WEIGHTS_REL = "weights/Final_LPRNet_model.pth"
# The default golden sample comes from the source repository's test set (the
# recipe ships no images); the first sorted entry is 京PL3N67.jpg, the plate
# the golden was validated against.
SAMPLE_IMAGE_DIR = ("data", "test")


# ── MaxPool3d surgery (recipe-local PyTorch module swap) ─────────────────────
# The Cloud AI 100 compiler only handles 2D pooling. LPRNet's nn.MaxPool3d with
# kernel[0]=1 (no depth pooling) is mathematically equivalent to MaxPool2d plus a
# channel slice, so we swap the module before export. This is a PyTorch-level
# transform, not an ONNX transform, and has no second canonical consumer, so it
# stays recipe-local. The class is built inside a function so the torch import
# stays out of the module top level (runtime installs stay lightweight).


def _make_maxpool3d_decomposed():
    """Return an nn.Module class replacing nn.MaxPool3d(kernel=(1,kH,kW), stride=(s0,sH,sW)).

    Decomposes to MaxPool2d(kernel=(kH,kW), stride=(sH,sW)) + channel slice
    [::s0]. Exports to ONNX MaxPool (2D) + Slice, both supported by Cloud AI 100.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class MaxPool3dDecomposed(nn.Module):
        def __init__(self, kernel_size, stride):
            super().__init__()
            self.kH, self.kW = kernel_size[1], kernel_size[2]
            self.sH, self.sW = stride[1], stride[2]
            self.s0 = stride[0]

        def forward(self, x):
            x = F.max_pool2d(x, kernel_size=(self.kH, self.kW), stride=(self.sH, self.sW))
            if self.s0 > 1:
                x = x[:, :: self.s0, :, :]
            return x

    return MaxPool3dDecomposed


def _replace_maxpool3d(module) -> None:
    import torch.nn as nn

    MaxPool3dDecomposed = _make_maxpool3d_decomposed()
    for name, child in module.named_children():
        if isinstance(child, nn.MaxPool3d):
            k = child.kernel_size
            s = child.stride
            setattr(module, name, MaxPool3dDecomposed(kernel_size=k, stride=s))
            print(f"  Replaced MaxPool3d -> MaxPool3dDecomposed: kernel={k} stride={s}")
        else:
            _replace_maxpool3d(child)


# ── Batch-coupled normalization fix (recipe-local ONNX transform) ────────────
# LPRNet's global_context block (model/LPRNet.py forward) normalizes each
# feature map with a *manual* batch-norm computed at inference time:
#
#     f_pow  = torch.pow(f, 2)
#     f_mean = torch.mean(f_pow)          # global mean over ALL dims
#     f      = torch.div(f, f_mean)
#
# ``torch.mean`` with no ``dim`` reduces over the whole tensor, **including the
# batch axis**, to a single scalar that then scales every row. A row's output
# therefore depends on the aggregate statistics of the *entire batch*. This is
# a latent bug that is invisible at batch 1 (where the global mean equals the
# per-image mean) but corrupts batched inference:
#   - distinct images shift each other's logits (measured ~2.7 in fp32, 246
#     with zero-padding), and
#   - tiling one image to fill a fixed QPC batch overflows FP16 for ~19% of
#     images (the QAIC FP16 compile of the global-over-batch reduce hits ±inf),
#     producing NaN/inf logits and a garbled decode.
#
# The fix makes the normalization per-image (reduce over the spatial/channel
# dims only, keeping the batch axis). At batch 1 this is bit-identical to the
# original, so the validated single-image accuracy is unchanged; at batch > 1
# it removes the cross-row coupling entirely, making the throughput QPC correct
# for distinct images and safe to tile.
#
# In the exported ONNX this is the four ``ReduceMean`` nodes fed by the ``Pow``
# nodes with **no axes attribute and no axes input** (a global reduce). The
# fifth ``ReduceMean`` (axes=[2], the ``torch.mean(x, dim=2)`` head) is a
# legitimate spatial reduce and is left untouched.


class PerImageReduceMeanTransform(BaseOnnxTransform):
    """Rewrite global (all-axes) ReduceMean nodes to per-image (axes=[1,2,3]).

    Targets only the batch-coupled normalization reduces: ``ReduceMean`` nodes
    whose input is a rank-4 ``[batch, C, H, W]`` tensor and which carry no axes
    attribute and no axes input (i.e. a reduce over every axis, including the
    batch axis). Those are rewritten to reduce over the channel/spatial axes
    with ``keepdims=1`` so the per-image mean keeps the ``[batch, 1, 1, 1]``
    shape and broadcasts correctly against the feature map. Any other
    ``ReduceMean`` (e.g. the head's ``axes=[2]``) is left unchanged.
    """

    @classmethod
    def apply(cls, model: Any) -> Tuple[Any, bool]:
        from onnx import AttributeProto, shape_inference

        # Map each tensor to its rank. The raw PyTorch export carries no
        # value_info (only the graph input/output shapes), so the rank map is
        # nearly empty; run shape inference to populate the internal tensor
        # ranks. If inference is unsupported on this graph, fall back to a
        # producer-op heuristic: the backbone conv/pool/elementwise ops all
        # emit rank-4 [batch, C, H, W] tensors.
        ranks: dict[str, int] = {}
        for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
            try:
                ranks[vi.name] = len(vi.type.tensor_type.shape.dim)
            except Exception:  # noqa: SIM105  # non-tensor or missing shape
                pass
        try:
            inferred = shape_inference.infer_shapes(model, strict_mode=False)
            for vi in inferred.graph.value_info:
                try:
                    ranks.setdefault(vi.name, len(vi.type.tensor_type.shape.dim))
                except Exception:  # noqa: SIM105
                    pass
        except Exception:  # noqa: SIM105  # shape inference unsupported on this graph
            rank4_ops = {"Conv", "ConvTranspose", "MaxPool", "AveragePool", "Pow", "Relu", "BatchNormalization", "Add", "Mul", "Sub"}
            for node in model.graph.node:
                if node.op_type in rank4_ops:
                    for out in node.output:
                        ranks.setdefault(out, 4)

        fixed = 0
        for node in model.graph.node:
            if node.op_type != "ReduceMean" or len(node.input) < 1:
                continue
            has_axes_attr = any(a.name == "axes" for a in node.attribute)
            has_axes_input = len(node.input) >= 2 and node.input[1] != ""
            if has_axes_attr or has_axes_input:
                continue  # not a global reduce
            if ranks.get(node.input[0]) != 4:
                continue  # not a [batch, C, H, W] feature map
            for attr in list(node.attribute):
                if attr.name == "keepdims":
                    attr.i = 1
            if not any(a.name == "keepdims" for a in node.attribute):
                nk = node.attribute.add()
                nk.name = "keepdims"
                nk.type = AttributeProto.INT
                nk.i = 1
            na = node.attribute.add()
            na.name = "axes"
            na.type = AttributeProto.INTS
            na.ints.extend([1, 2, 3])
            fixed += 1
        return model, fixed > 0


def _ensure_repo(repo_arg: str | None, work_dir: Path) -> Path:
    """Return the path to a LPRNet_Pytorch checkout, cloning it if necessary."""
    if repo_arg:
        repo = Path(repo_arg).resolve()
        if not (repo / "model" / "LPRNet.py").is_file():
            raise FileNotFoundError(f"--repo {repo!r} does not contain model/LPRNet.py")
        return repo

    repo = work_dir / "repo"
    if (repo / "model" / "LPRNet.py").is_file():
        return repo
    print(f"[export] cloning {REPO_URL} -> {repo}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    return repo


def _build_model(repo: Path):
    """Build the LPRNet model, load weights, and apply MaxPool3d surgery.

    The architecture is imported from the cloned source repo (never inlined).
    The character set comes from the recipe's own validated CHARS in
    postprocess.py (identical to the repo's data/load_data.py CHARS), so the
    repo's data package — which pulls torch.utils.data at import — is not
    needed.
    """
    import torch

    from .postprocess import CHARS

    sys.path.insert(0, str(repo))
    try:
        from model.LPRNet import build_lprnet  # noqa: PLC0415
    finally:
        sys.path.pop(0)

    class_num = len(CHARS)
    model = build_lprnet(lpr_max_len=LPR_MAX_LEN, phase=False, class_num=class_num, dropout_rate=DROPOUT_RATE)
    weights = repo / DEFAULT_WEIGHTS_REL
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found at {weights!r}; pass --repo to a full LPRNet_Pytorch checkout")
    model.load_state_dict(torch.load(str(weights), map_location="cpu"))
    model.eval()

    print("[export] applying MaxPool3d surgery...")
    _replace_maxpool3d(model)
    model.eval()
    return model, class_num


def _artifact_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "input_raw": work_dir / "input.raw",
        "framework_output": work_dir / "framework_output_0.raw",
        "cpu_output": work_dir / "cpu_output.raw",
        "raw_onnx": work_dir / "lprnet.onnx",
        "final_onnx": work_dir / FINAL_ONNX_NAME,
        "qpc_latency_dir": work_dir / LATENCY_QPC_SUBDIR,
        "qpc_throughput_dir": work_dir / THROUGHPUT_QPC_SUBDIR,
    }


def validate_onnx_output_contract(onnx_path: Path) -> None:
    """Verify the exported deployable ONNX exposes exactly the validated output.

    Catches exports that accidentally include training, debug, intermediate, or
    auxiliary outputs before those artifacts reach CPU golden generation or
    qaic-compile.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    graph_outputs = list(model.graph.output)
    output_names = [output.name for output in graph_outputs]
    if output_names != [OUTPUT_NAME]:
        raise ValueError(f"expected ONNX outputs {[OUTPUT_NAME]}, got {output_names}")

    for output in graph_outputs:
        dims = output.type.tensor_type.shape.dim
        dim_values = [dim.dim_value if dim.dim_value else dim.dim_param for dim in dims]
        if len(dim_values) != 3:
            raise ValueError(f"expected {output.name!r} to be rank 3 [batch, classes, timesteps], got {dim_values}")
        # batch=1, classes=68, timesteps=18 (validated); allow symbolic batch.
        if isinstance(dim_values[1], int) and dim_values[1] != NUM_CLASSES:
            raise ValueError(f"expected {output.name!r} classes dim {NUM_CLASSES}, got {dim_values[1]}")
        if isinstance(dim_values[2], int) and dim_values[2] != NUM_TIMESTEPS:
            raise ValueError(f"expected {output.name!r} timesteps dim {NUM_TIMESTEPS}, got {dim_values[2]}")


def export_onnx(repo: Path, work_dir: Path, sample_image: str) -> dict[str, Path]:
    import torch

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(work_dir)

    model, _class_num = _build_model(repo)

    # Build the representative input from the sample image (validated pipeline).
    img = preprocess_plate_image(sample_image)
    img.tofile(paths["input_raw"])
    inp_tensor = torch.from_numpy(img)

    # Framework golden output.
    with torch.no_grad():
        fw_out = model(inp_tensor)
    fw_outputs = [fw_out] if not isinstance(fw_out, (tuple, list)) else list(fw_out)
    if len(fw_outputs) != 1:
        raise ValueError(f"expected exactly one model output, got {len(fw_outputs)}")
    fw_outputs[0].numpy().tofile(paths["framework_output"])

    # Export to ONNX. The batch axis is always a named symbol (never baked
    # concrete into the graph); the compile step binds it with
    # -onnx-define-symbol=batch_size,<N>.
    onnx_path = str(paths["raw_onnx"])
    torch.onnx.export(
        model,
        inp_tensor,
        onnx_path,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch_size"}, OUTPUT_NAME: {0: "batch_size"}},
    )
    print(f"[export] exported raw ONNX to {onnx_path}")
    validate_onnx_output_contract(paths["raw_onnx"])

    # Transform pipeline: batch-coupled-norm fix -> ReduceMean axes fix ->
    # onnxsim -> FP16 clip. The per-image fix runs first, on the raw exported
    # structure, so its rank-based detection sees the original [batch,C,H,W]
    # shapes before onnxsim rewrites the graph.
    run_pipeline(
        str(paths["raw_onnx"]),
        str(paths["final_onnx"]),
        [PerImageReduceMeanTransform, ReduceMeanAxesInputToAttrTransform, OnnxSimTransform, FP16ClipTransform],
    )
    validate_onnx_output_contract(paths["final_onnx"])

    # CPU golden from the final deployable artifact.
    _write_cpu_golden(paths["final_onnx"], img, paths["cpu_output"])
    return paths


def _write_cpu_golden(onnx_path: Path, image, output_path: Path) -> None:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (logits,) = sess.run(None, {INPUT_NAME: image})
    np.ascontiguousarray(logits).tofile(output_path)


# QPC build targets. The latency QPC is tuned for single-request latency
# (16 cores, batch 1); the throughput QPC is tuned for batched serving
# (4 cores, batch 8). Both are always compiled. The Triton lprnet_qpc model
# wraps the throughput build, so its serving config (max_batch_size, instance
# count, dynamic batching) is derived from the same constants that drive
# compilation.
LATENCY_CORES = 16
LATENCY_BATCH = 1
THROUGHPUT_CORES = 4
THROUGHPUT_BATCH = 8
# Triton instance count for the throughput QPC. Each python instance owns its
# own qaic.Session (num_activations=1), so N instances give an effective
# in-flight batch of N * QPC batch and reserve N * QPC-cores NSPs. The
# throughput QPC is compiled for 4 cores; 4 instances x 4 cores = 16 NSPs =
# one full SoC (16 NSP cores), the canonical (cores=4, instances=4)
# device-filling throughput point.
THROUGHPUT_INSTANCES = 4
# Batch-accumulation window (microseconds) for the throughput model's
# dynamic batching.
DYNAMIC_BATCHING_MAX_QUEUE_DELAY_US = 5000
PREFERRED_BATCH_SIZES = (1, 2, 4, 8)


def compile_latency(paths: dict[str, Path], work_dir: Path, cores: int = LATENCY_CORES, batch: int = LATENCY_BATCH, mos: int = 1) -> None:
    """Compile the latency-tuned QPC (16 cores, batch 1, -mos=1)."""
    qaic_compile(
        str(paths["final_onnx"].name),
        cores,
        paths["qpc_latency_dir"],
        cwd=work_dir,
        mos=mos,
        extra_args=(f"-onnx-define-symbol=batch_size,{batch}",),
    )


def compile_throughput(paths: dict[str, Path], work_dir: Path, cores: int = THROUGHPUT_CORES, batch: int = THROUGHPUT_BATCH, mos: int = 1) -> None:
    """Compile the throughput-tuned QPC (4 cores, batch 8, -mos=1)."""
    qaic_compile(
        str(paths["final_onnx"].name),
        cores,
        paths["qpc_throughput_dir"],
        cwd=work_dir,
        mos=mos,
        extra_args=(f"-onnx-define-symbol=batch_size,{batch}",),
    )


_LPRNET_POSTPROCESS_MODEL_SOURCE = """\
import numpy as np

from qaic_models.vision.ocr.lprnet.postprocess import decode_ctc


class TritonPythonModel:
    def initialize(self, args):
        import triton_python_backend_utils as pb_utils

        self.pb_utils = pb_utils

    def execute(self, requests):
        responses = []
        for request in requests:
            logits = self.pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            # logits shape [1, 68, 18]
            pred = decode_ctc(logits)
            plate = pred["plate"]
            confidence = pred["confidence"]
            # plate is a single STRING [1] tensor holding the full plate;
            # confidence/num_chars are scalar [1] tensors.
            plate_arr = np.array([plate], dtype=object)
            conf_arr = np.asarray([confidence], dtype=np.float32)
            num_chars_arr = np.asarray([len(plate)], dtype=np.int32)
            responses.append(
                self.pb_utils.InferenceResponse(
                    output_tensors=[
                        self.pb_utils.Tensor("plate", plate_arr),
                        self.pb_utils.Tensor("confidence", conf_arr),
                        self.pb_utils.Tensor("num_chars", num_chars_arr),
                    ]
                )
            )
        return responses
"""


def generate_triton_repo(
    paths: dict[str, Path],
    repo_root: Path,
    *,
    triton_device_id: int | None = None,
    triton_protobuf=None,
) -> dict[str, Path]:
    """Generate a Triton model repository for LPRNet.

    Expects ``paths`` as returned by :func:`export_onnx` with a ``final_onnx``
    entry and the compiled ``qpc_throughput_dir`` QPC (both QPC builds are
    always produced by a full export run). Creates four models under
    ``repo_root``:

    * ``lprnet_onnx`` – ONNX Runtime backend (dynamic batch).
    * ``lprnet_postprocess`` – Python backend that decodes CTC logits.
    * ``lprnet_onnx_ensemble`` – ``lprnet_onnx`` → postprocess.
    * ``lprnet_qpc`` – Python backend wrapping the throughput QPC. Triton
      dynamic batching coalesces up to the QPC's fixed batch; the model fills
      empty slots by cycling the real inputs (zero-padding is unsafe: the QPC
      has cross-batch interaction), runs the QPC, and decodes each plate.
      Four instances fill one SoC (effective in-flight batch 32).
    """
    repo_root = Path(repo_root).resolve()
    programqpc = paths["qpc_throughput_dir"] / "programqpc.bin"
    if not programqpc.is_file():
        raise FileNotFoundError(
            "Triton QPC model requires the generated throughput QPC "
            "(qpc_throughput/programqpc.bin); it is produced by a full export run."
        )

    # The deployable ONNX keeps a symbolic batch axis (the dual-compile
    # contract: one ONNX compiles to the batch-1 latency and batch-8
    # throughput QPCs), so its raw model declares a dynamic batch dim (-1).
    onnx_input_spec = {"name": INPUT_NAME, "data_type": "TYPE_FP32", "dims": [-1, *INPUT_SHAPE[1:]]}
    onnx_logits_spec = {"name": OUTPUT_NAME, "data_type": "TYPE_FP32", "dims": [-1, NUM_CLASSES, NUM_TIMESTEPS]}

    raw_onnx_cfg = TritonModelConfig(
        model_path=paths["final_onnx"],
        name="lprnet_onnx",
        backend="onnxruntime",
        inputs=[onnx_input_spec],
        outputs=[onnx_logits_spec],
    )
    raw_onnx_cfg.tofile(repo_root / raw_onnx_cfg.name, triton_protobuf=triton_protobuf)

    # Post-process model. plate is a single STRING [1] tensor holding the full
    # decoded plate; confidence/num_chars are scalar [1] tensors.
    logits_spec = {"name": OUTPUT_NAME, "data_type": "TYPE_FP32", "dims": [1, NUM_CLASSES, NUM_TIMESTEPS]}
    post_cfg = TritonModelConfig(
        name="lprnet_postprocess",
        backend="python",
        model_source=_LPRNET_POSTPROCESS_MODEL_SOURCE,
        inputs=[logits_spec],
        outputs=[
            {"name": "plate", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "confidence", "data_type": "TYPE_FP32", "dims": [1]},
            {"name": "num_chars", "data_type": "TYPE_INT32", "dims": [1]},
        ],
    )
    post_cfg.tofile(repo_root / post_cfg.name, triton_protobuf=triton_protobuf)

    # lprnet_qpc: a python model that owns the QPC. With max_batch_size > 0 the
    # base config strips the leading batch dim from each spec, so the input is
    # declared per-item [3, 24, 94] and each output per-item [1]. The model
    # (triton_model.py) is copied into the repo and the QPC binary is placed
    # next to it; the model discovers the QPC's fixed batch at initialization.
    qpc_model_root = repo_root / "lprnet_qpc"
    qpc_cfg = TritonModelConfig(
        model_path=Path(__file__).resolve().parent / "triton_model.py",
        name="lprnet_qpc",
        backend="python",
        max_batch_size=THROUGHPUT_BATCH,
        count=THROUGHPUT_INSTANCES,
        inputs=[{"name": INPUT_NAME, "data_type": "TYPE_FP32", "dims": list(INPUT_SHAPE)}],
        outputs=[
            {"name": "plate", "data_type": "TYPE_STRING", "dims": [1, 1]},
            {"name": "confidence", "data_type": "TYPE_FP32", "dims": [1, 1]},
            {"name": "num_chars", "data_type": "TYPE_INT32", "dims": [1, 1]},
        ],
        dynamic_batching={
            "preferred_batch_size": list(PREFERRED_BATCH_SIZES),
            "max_queue_delay_microseconds": DYNAMIC_BATCHING_MAX_QUEUE_DELAY_US,
        },
        parameters=(
            {"device_id": str(triton_device_id)}
            if triton_device_id is not None else None
        ),
    )
    qpc_cfg.tofile(qpc_model_root, triton_protobuf=triton_protobuf)
    # The python backend only writes model.py; place the QPC binary beside it
    # where triton_model.py looks for it (lprnet_qpc/1/programqpc.bin).
    qpc_artifact_dir = qpc_model_root / str(qpc_cfg.model_version)
    qpc_artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(programqpc, qpc_artifact_dir / "programqpc.bin")

    # Ensemble: ONNX raw model -> postprocess. The ensemble needs its own
    # batch-1 copy of the onnx member spec: the ensemble's top-level IO is
    # derived from its members, and the postprocess member is concrete batch-1
    # (it decodes one plate: [1, 68, 18] -> plate STRING [1]). Reusing the
    # dynamic-batch raw spec would have the ensemble advertise [-1, 3, 24, 94]
    # ingress that dies mid-graph at the queue handoff for batch>1; the
    # concrete copy rejects an oversized request at the boundary instead. The
    # standalone raw onnx model keeps its dynamic batch dim.
    ens_onnx_cfg = TritonModelConfig(
        model_path=raw_onnx_cfg.model_path,
        name=raw_onnx_cfg.name,
        backend=raw_onnx_cfg.backend,
        inputs=[{"name": INPUT_NAME, "data_type": "TYPE_FP32", "dims": [1, *INPUT_SHAPE[1:]]}],
        outputs=[{"name": OUTPUT_NAME, "data_type": "TYPE_FP32", "dims": [1, NUM_CLASSES, NUM_TIMESTEPS]}],
    )
    onnx_ens_root = repo_root / "lprnet_onnx_ensemble"
    TritonModelEnsemble("lprnet_onnx_ensemble", [ens_onnx_cfg, post_cfg]).tofile(
        onnx_ens_root, triton_protobuf=triton_protobuf
    )

    return {
        "onnx_model": repo_root / raw_onnx_cfg.name,
        "qpc_model": qpc_model_root,
        "postprocess_model": repo_root / post_cfg.name,
        "onnx_ensemble_model": onnx_ens_root,
    }



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="lprnet", choices=SUPPORTED_VARIANTS)
    parser.add_argument("--repo", default=None, help="Path to a LPRNet_Pytorch checkout (default: clone into --work-dir/repo)")
    parser.add_argument("--work-dir", required=True, help="Directory to write all generated artifacts into")
    parser.add_argument(
        "--image",
        default=None,
        help="Sample plate image used to build input.raw and the CPU golden "
             "(default: the first sorted image in <repo>/data/test)",
    )
    parser.add_argument(
        "--triton-model-repo",
        default=None,
        help="Optional Triton model repository output directory. The lprnet_qpc model "
        "serves the generated qpc_throughput QPC (4 cores, batch 8, four instances "
        "with dynamic batching).",
    )
    parser.add_argument("--triton-device-id", type=int, default=None, help="QAIC QID for the generated Triton model")
    args = parser.parse_args(argv)

    _require_variant(args.variant)
    # Resolve to an absolute path: qaic_compile() runs qaic-compile with
    # cwd=work_dir while also passing -aic-binary-dir=<work_dir>/qpc_*, and a
    # relative work_dir would make that flag resolve relative to itself.
    work_dir = Path(args.work_dir).resolve()

    repo = _ensure_repo(args.repo, work_dir)
    if args.image is not None:
        sample_image = args.image
    else:
        test_images = sorted(repo.joinpath(*SAMPLE_IMAGE_DIR).glob("*.jpg"))
        if not test_images:
            raise FileNotFoundError(
                f"no sample plate images found in {repo.joinpath(*SAMPLE_IMAGE_DIR)!r}; pass --image"
            )
        sample_image = str(test_images[0])
        print(f"[export] default sample image: {sample_image}")
    if not Path(sample_image).is_file():
        raise FileNotFoundError(f"sample image not found at {sample_image!r}; pass --image")

    paths = export_onnx(repo, work_dir, sample_image)
    print(f"[export] wrote {paths['final_onnx']}")

    # Both QPC builds always compile, in parallel: separate artifact
    # directories, one shared ONNX input (the dual-compile contract).
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(compile_latency, paths, work_dir),
            executor.submit(compile_throughput, paths, work_dir),
        ]
        for future in as_completed(futures):
            future.result()  # re-raise the first compile failure
    print(f"[export] compiled latency QPC to {paths['qpc_latency_dir']}")
    print(f"[export] compiled throughput QPC to {paths['qpc_throughput_dir']}")

    if args.triton_model_repo:
        generated = generate_triton_repo(paths, Path(args.triton_model_repo).resolve(), triton_device_id=args.triton_device_id)
        print(f"[export] wrote Triton model repository entries to {Path(args.triton_model_repo).resolve()}")
        for name, path in generated.items():
            print(f"[export] Triton {name}: {path}")


if __name__ == "__main__":
    main()
