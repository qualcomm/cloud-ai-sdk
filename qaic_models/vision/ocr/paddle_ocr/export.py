# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Export and compile the PP-DocLayoutV3 -> PP-OCRv6 OCR stages.

The recipe downloads the layout checkpoint and the det/rec ONNX snapshots,
exports the layout checkpoint, applies the graph repairs validated during
onboarding, writes CPU-reference artifacts, and compiles the four production
QPC variants. All outputs are written below
``--work-dir``.  With ``--triton-model-repo``, a Triton model repository
with one QPC-backed model per pipeline stage (layout, det, and the batched
rec stage) is generated from the work-dir artifacts as well.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from qaic_models.base.compile import qaic_compile
from qaic_models.base.onnx_transforms import (
    ClipUndefinedInputsTransform,
    FP16ClipTransform,
    OnnxSimTransform,
    assert_symbolic_batch_axis,
    run_pipeline,
)
from qaic_models.base.triton import TritonModelConfig

from .model import (
    CHAR_DICT_NAME,
    DET_LATENCY_QPC_SUBDIR,
    DET_ONNX_NAME,
    REC_BATCH,
    REC_ONNX_NAME,
    REC_THROUGHPUT_QPC_SUBDIR,
)
from .model import LAYOUT_NODE_PRECISION_NAME, LAYOUT_ONNX_NAME, LAYOUT_QPC_SUBDIR
from .onnx_transforms import (
    CtcGreedySummaryTransform,
    Eliminate5DQKVTransform,
    EmptyResizeRoiTransform,
    FixPaddleShapeReshapesTransform,
    RankDecompose9x9DWConvTransform,
    StripStaleValueInfoTransform,
    TopoSortTransform,
    UnifySymbolicDimsTransform,
    _LayoutCapSliceSentinelStartsTransform,
    _LayoutFixDanglingClipBoundsTransform,
    _LayoutFP16RangeClipTransform,
    _LayoutFloat64ToFloat32Transform,
)
from .postprocess import (
    LAYOUT_NUM_CLASSES,
    LAYOUT_NUM_QUERIES,
    OUTPUT_NAME,
    REC_MAX_PROBS_NAME,
    REC_NUM_CLASSES,
    REC_TOKEN_IDS_NAME,
    decode_ctc,
    decode_ctc_summary,
)
from .preprocess import (
    _load_bgr,
    DET_INPUT_H,
    DET_INPUT_W,
    INPUT_NAME,
    LAYOUT_INPUT_NAME,
    LAYOUT_MASK_SIZE,
    REC_INPUT_SHAPE,
    REC_INPUT_W,
    preprocess_det_image,
    preprocess_layout_image,
    preprocess_rec_image,
)

log = logging.getLogger(__name__)

DET_MODEL = "pp-ocrv6-medium-det-onnx"
REC_MODEL = "pp-ocrv6-medium-rec-onnx"
DET_REPO = "PaddlePaddle/PP-OCRv6_medium_det_onnx"
REC_REPO = "PaddlePaddle/PP-OCRv6_medium_rec_onnx"
LAYOUT_REPO = "PaddlePaddle/PP-DocLayoutV3_safetensors"

# Hugging Face revisions pinned for reproducible exports. Each stage download
# resolves the snapshot at this commit and prints one
# ``hf-revision <repo_id> <commit>`` line so the functional runner can record
# the exact commits in the run metadata.
HF_REVISIONS = {
    DET_REPO: "61323801669c338b7891481ec7bac61ce31b576a",
    REC_REPO: "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
    LAYOUT_REPO: "97d101e6db2642e162a1d05392d1b0231c91033e",
}
LAYOUT_MODEL = "pp-doclayoutv3"

# The recipe ships no sample images. The default detector/recognizer golden
# inputs are produced at export time from the arXiv source of the validated
# white-paper page (arXiv:1212.1442v1, Ferris & Poulin, "Algorithms for the
# Markov Entropy Decomposition"): page 1 rendered at 144 DPI (1224x1584, the
# capture used during onboarding) feeds layout/detection; the PACS-numbers
# line cropped from the same page feeds recognition.
ARXIV_PDF_URL = "https://arxiv.org/pdf/1212.1442v1"
ARXIV_PDF_NAME = "1212.1442v1.pdf"
ARXIV_DET_SAMPLE_NAME = "arxiv_1212.1442v1_page1.png"
ARXIV_REC_SAMPLE_NAME = "arxiv_1212.1442v1_pacs_line.jpg"
ARXIV_REC_LINE_PREFIX = "PACS numbers"
ARXIV_RENDER_SCALE = 2.0  # 72 pt/in x 2 = 144 DPI


def _ensure_default_samples(work_dir: Path, *, need_det: bool, need_rec: bool) -> dict[str, Path]:
    """Download the arXiv source PDF and render the default golden samples.

    Idempotent: existing sample files are reused. PyMuPDF is imported lazily
    so CPU-runtime installs never need it.
    """
    samples: dict[str, Path] = {}
    if not (need_det or need_rec):
        return samples
    samples_dir = work_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = samples_dir / ARXIV_PDF_NAME
    if not pdf_path.is_file() or pdf_path.open("rb").read(4) != b"%PDF":
        print(f"[export] downloading {ARXIV_PDF_URL} -> {pdf_path}")
        tmp_path = pdf_path.with_name(pdf_path.name + ".part")
        urllib.request.urlretrieve(ARXIV_PDF_URL, tmp_path)
        tmp_path.replace(pdf_path)

    try:
        import pymupdf
    except ImportError:  # PyMuPDF < 1.22.3 exposed only the `fitz` module
        import fitz as pymupdf

    document = pymupdf.open(str(pdf_path))
    try:
        page = document[0]
        matrix = pymupdf.Matrix(ARXIV_RENDER_SCALE, ARXIV_RENDER_SCALE)
        if need_det:
            det_sample = samples_dir / ARXIV_DET_SAMPLE_NAME
            if not det_sample.is_file():
                page.get_pixmap(matrix=matrix).save(str(det_sample))
                print(f"[export] rendered default detector sample {det_sample}")
            samples["det"] = det_sample
        if need_rec:
            rec_sample = samples_dir / ARXIV_REC_SAMPLE_NAME
            if not rec_sample.is_file():
                line_bbox = None
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        text = "".join(span["text"] for span in line["spans"])
                        if text.strip().startswith(ARXIV_REC_LINE_PREFIX):
                            line_bbox = pymupdf.Rect(line["bbox"])
                            break
                    if line_bbox is not None:
                        break
                if line_bbox is None:
                    raise RuntimeError(
                        f"could not locate the {ARXIV_REC_LINE_PREFIX!r} line in {pdf_path}; "
                        "pass --rec-image"
                    )
                # Generous horizontal padding: the text bbox hugs the glyphs
                # and the leading/trailing characters clip at the line ends.
                clip = pymupdf.Rect(
                    line_bbox.x0 - 12, line_bbox.y0 - 3,
                    line_bbox.x1 + 12, line_bbox.y1 + 3,
                )
                page.get_pixmap(matrix=matrix, clip=clip).save(str(rec_sample))
                print(f"[export] rendered default recognizer sample {rec_sample}")
            samples["rec"] = rec_sample
    finally:
        document.close()
    return samples


def _pinned_revision(repo_id: str) -> str | None:
    """Pinned Hugging Face revision for a default recipe repository.

    Override repositories passed through ``--hf-repo`` (and friends) are not
    pinned: the caller chose them explicitly, so they resolve to the default
    branch and no revision line is printed for them.
    """
    return HF_REVISIONS.get(repo_id)


def _download_snapshot(repo_id: str, destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    revision = _pinned_revision(repo_id)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        revision=revision,
        allow_patterns=["inference.onnx", "inference.yml", "*.yaml"],
    )
    onnx_path = destination / "inference.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Hugging Face snapshot {repo_id!r} has no inference.onnx")
    if revision is not None:
        print(f"hf-revision {repo_id} {revision}")
    return onnx_path


def _download_layout_snapshot(repo_id: str, destination: Path) -> Path:
    """Download the complete PP-DocLayoutV3 checkpoint used by the exporter."""
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    revision = _pinned_revision(repo_id)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        revision=revision,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "inference.yml",
            "model.safetensors",
        ],
    )
    required = [destination / name for name in ("config.json", "model.safetensors")]
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Hugging Face layout snapshot {repo_id!r} is missing: {', '.join(missing)}"
        )
    if revision is not None:
        print(f"hf-revision {repo_id} {revision}")
    return destination


def _char_dict(snapshot: Path, output: Path) -> list[str]:
    import yaml

    config_path = snapshot / "inference.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"recognizer snapshot has no {config_path}")
    config = yaml.safe_load(config_path.read_text())
    try:
        chars = list(config["PostProcess"]["character_dict"])
    except (KeyError, TypeError) as exc:
        raise ValueError("inference.yml has no PostProcess.character_dict") from exc
    chars.append(" ")
    output.write_text("\n".join(chars))
    return chars


def validate_onnx_output_contract(
    path: Path,
    stage: str,
    summary: bool = False,
    *,
    allow_symbolic: bool = False,
    out_masks: bool = False,
) -> None:
    import onnx

    graph = onnx.load(str(path)).graph
    outputs = list(graph.output)
    if summary:
        if stage != "rec":
            raise ValueError(f"summary output contract is only defined for rec, got stage {stage!r}")
        expected = [
            (REC_TOKEN_IDS_NAME, onnx.TensorProto.INT32),
            (REC_MAX_PROBS_NAME, onnx.TensorProto.FLOAT),
        ]
        actual = [(output.name, output.type.tensor_type.elem_type) for output in outputs]
        if actual != expected:
            raise ValueError(f"rec summary ONNX must expose exactly {expected}, got {actual}")
        for output in outputs:
            if len(output.type.tensor_type.shape.dim) != 2:
                raise ValueError(f"rec summary outputs must be rank 2 (B,T), got {output.name}")
        return
    if [output.name for output in outputs] != [OUTPUT_NAME]:
        if stage != "layout":
            raise ValueError(f"{stage} ONNX must expose exactly {[OUTPUT_NAME]}, got {[output.name for output in outputs]}")
        expected = ["logits", "pred_boxes", "order_logits"]
        if out_masks:
            expected.append("out_masks")
        if [output.name for output in outputs] != expected:
            raise ValueError(f"layout ONNX must expose exactly {expected}, got {[output.name for output in outputs]}")
        expected_shapes = [
            (1, LAYOUT_NUM_QUERIES, LAYOUT_NUM_CLASSES),
            (1, LAYOUT_NUM_QUERIES, 4),
            (1, LAYOUT_NUM_QUERIES, LAYOUT_NUM_QUERIES),
        ]
        if out_masks:
            expected_shapes.append((1, LAYOUT_NUM_QUERIES, LAYOUT_MASK_SIZE, LAYOUT_MASK_SIZE))
        for output, expected_shape in zip(outputs, expected_shapes):
            dims = output.type.tensor_type.shape.dim
            if len(dims) != len(expected_shape):
                raise ValueError(f"layout output {output.name} must be rank {len(expected_shape)}")
            actual = tuple(
                dim.dim_value if dim.HasField("dim_value") else dim.dim_param if dim.HasField("dim_param") else None
                for dim in dims
            )
            if allow_symbolic:
                invalid = [
                    (index, value, expected)
                    for index, (value, expected) in enumerate(zip(actual, expected_shape))
                    if value is None or (isinstance(value, int) and value != expected)
                ]
                if invalid:
                    raise ValueError(
                        f"layout output {output.name} has incompatible shape {actual}; "
                        f"expected {expected_shape} (symbolic dimensions are allowed during raw export)"
                    )
            elif actual != expected_shape:
                raise ValueError(f"layout output {output.name} must have shape {expected_shape}, got {actual}")
        return
    dims = outputs[0].type.tensor_type.shape.dim
    if stage == "det":
        # det: (1, 1, 640, 640) probability map — rank 4.
        if len(dims) != 4:
            raise ValueError(f"det output must be rank 4, got {len(dims)}")
        if any(dim.HasField("dim_value") and dim.dim_value != value for dim, value in zip(dims[-2:], (DET_INPUT_H, DET_INPUT_W))):
            raise ValueError(f"det output must end in {DET_INPUT_H}x{DET_INPUT_W}")
    else:
        # rec: (B, T, C) CTC class probabilities — rank 3 (no channel dim).
        if len(dims) != 3:
            raise ValueError(f"rec output must be rank 3, got {len(dims)}")
        if dims[-1].HasField("dim_value") and dims[-1].dim_value != REC_NUM_CLASSES:
            raise ValueError(f"rec output class dim must be {REC_NUM_CLASSES}")


def _ort_session(path: Path):
    """Create a CPU ORT session, lowering the model IR version if the runtime is older.

    The upstream ONNX files declare IR 10, but the container's onnxruntime
    (1.16.x) only supports up to IR 9. The graph itself is standard (the ``onnx``
    package loads it fine), so lowering the ``ir_version`` metadata field (graph
    unchanged) lets the older runtime load it. This only affects the ORT
    golden/cpu-output runs — the deployable ONNX is handed to ``qaic-compile``,
    not ORT.
    """
    import re

    import onnx
    import onnxruntime as ort

    try:
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - ORT raises a typed error; match on its message.
        if "Unsupported model IR version" not in str(exc):
            raise
        match = re.search(r"max supported IR version: (\d+)", str(exc))
        if match is None:
            raise
        model = onnx.load(str(path))
        model.ir_version = int(match.group(1))
        return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def _normalize_ir_version(path: Path) -> None:
    """Lower the model's ``ir_version`` metadata to the toolchain's max, in place.

    The upstream PP-OCRv6 ONNX files declare IR 10, but the export toolchain in the
    Cloud AI 100 container (``onnx`` 1.13.x / onnxsim 0.4.x / onnxruntime 1.16.x)
    only supports up to IR 8/9. The graph itself is standard — it passes the full
    ``onnx`` checker at the lowered version — so rewriting the ``ir_version`` field
    (graph unchanged) lets onnxsim, onnxruntime, and ``qaic-compile`` all load it.
    This is a no-op on a toolchain that already supports the model's IR version.
    """
    import onnx

    model = onnx.load(str(path))
    if model.ir_version > onnx.IR_VERSION:
        model.ir_version = onnx.IR_VERSION
        onnx.save(model, str(path))
        print(f"{path.name}: lowered ir_version to {onnx.IR_VERSION} for the export toolchain")


def _run_ort(
    path: Path,
    input_array: np.ndarray,
    *,
    input_name: str = INPUT_NAME,
    all_outputs: bool = False,
):
    session = _ort_session(path)
    outputs = session.run(None, {input_name: input_array})
    if all_outputs:
        return outputs
    if len(outputs) != 1:
        raise ValueError(f"expected one output from {path.name}, got {len(outputs)}")
    return np.ascontiguousarray(outputs[0], dtype=np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).ravel()
    right = np.asarray(right, dtype=np.float32).ravel()
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))


def _write_fidelity(stage: str, framework: np.ndarray, transformed: np.ndarray, *, hard: bool) -> float:
    value = _cosine(framework, transformed)
    print(f"{stage} fidelity cosine: {value:.8f}")
    if hard and value < 0.9999:
        raise SystemExit(f"{stage} ONNX fidelity gate FAILED: cosine={value:.8f}")
    return value


class _LayoutGridSamplePatch:
    """Patch deformable-attention grid_sample with an ONNX-traceable bilinear op."""

    _original = None

    @classmethod
    def apply(cls):
        import torch
        import torch.nn.functional as functional

        if cls._original is not None:
            return
        cls._original = functional.grid_sample

        def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
            if mode != "bilinear":
                return cls._original(input, grid, mode, padding_mode, align_corners)
            _, channels, height, width = input.shape
            out_h, out_w = grid.shape[1], grid.shape[2]
            gx, gy = grid[..., 0], grid[..., 1]
            if align_corners:
                x = (gx + 1.0) * ((width - 1) * 0.5)
                y = (gy + 1.0) * ((height - 1) * 0.5)
            else:
                x = (gx + 1.0) * (width * 0.5) - 0.5
                y = (gy + 1.0) * (height * 0.5) - 0.5
            x0, y0 = x.floor(), y.floor()
            x1, y1 = x0 + 1.0, y0 + 1.0
            wx1, wy1 = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)

            def gather(xi, yi):
                if padding_mode == "zeros":
                    valid = ((xi >= 0.0) & (xi < width) & (yi >= 0.0) & (yi < height)).to(input.dtype)
                    valid = valid.unsqueeze(1)
                else:
                    valid = input.new_ones((input.shape[0], 1, out_h, out_w))
                flat = (yi.clamp(0, height - 1) * width + xi.clamp(0, width - 1)).long()
                flat = flat.unsqueeze(1).expand(input.shape[0], channels, out_h, out_w)
                values = torch.gather(input.reshape(input.shape[0], channels, height * width), 2,
                                       flat.reshape(input.shape[0], channels, out_h * out_w))
                return values.reshape(input.shape[0], channels, out_h, out_w) * valid

            return (
                gather(x0, y0) * (1.0 - wy1) * (1.0 - wx1)
                + gather(x1, y0) * (1.0 - wy1) * wx1
                + gather(x0, y1) * wy1 * (1.0 - wx1)
                + gather(x1, y1) * wy1 * wx1
            )

        functional.grid_sample = grid_sample

    @classmethod
    def undo(cls):
        if cls._original is None:
            return
        import torch.nn.functional as functional

        functional.grid_sample = cls._original
        cls._original = None



def _layout_checkpoint_key(key: str) -> str:
    return (
        key.replace(".fc1.", ".mlp.fc1.")
        .replace(".fc2.", ".mlp.fc2.")
        .replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        .replace("model.encoder.encoder.", "model.encoder.aifi.")
    )


def _load_layout_model(snapshot: Path):
    import torch
    from safetensors import safe_open
    from transformers import PPDocLayoutV3ForObjectDetection

    model = PPDocLayoutV3ForObjectDetection.from_pretrained(str(snapshot), torch_dtype=torch.float32)
    state = model.state_dict()
    loaded = set()
    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            remapped = _layout_checkpoint_key(key)
            if remapped not in state:
                raise RuntimeError(f"layout checkpoint tensor {key!r} maps to unknown key {remapped!r}")
            tensor = handle.get_tensor(key)
            if tuple(state[remapped].shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"layout checkpoint tensor {key!r} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(state[remapped].shape)} for {remapped!r}"
                )
            state[remapped].copy_(tensor)
            loaded.add(remapped)

    # The checkpoint stores tied decoder heads only once.  BatchNorm's
    # num_batches_tracked buffers are also intentionally absent from the
    # safetensors file.  Both are already initialized/tied by from_pretrained.
    missing = set(state) - loaded
    unexpected_missing = {
        key
        for key in missing
        if not key.endswith("num_batches_tracked")
        and not key.startswith("model.decoder.class_embed.")
        and not key.startswith("model.decoder.bbox_embed.")
    }
    if unexpected_missing:
        raise RuntimeError(
            "layout checkpoint did not provide model tensors: "
            + ", ".join(sorted(unexpected_missing))
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _export_layout(work_dir: Path, repo_id: str, sample: Path, *, out_masks: bool = False) -> None:
    """Export PP-DocLayoutV3, optionally retaining its mask-head output."""
    import torch

    snapshot = _download_layout_snapshot(repo_id, work_dir / LAYOUT_MODEL / "hf_snapshot")
    raw = work_dir / f"{LAYOUT_MODEL}.onnx"
    final = work_dir / LAYOUT_ONNX_NAME
    image = _load_bgr(str(sample))
    input_array = preprocess_layout_image(image).astype(np.float32)
    input_array.tofile(work_dir / "layout_input.raw")

    model = _load_layout_model(snapshot)

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values):
            output = self.wrapped(pixel_values)
            heads = (output.logits, output.pred_boxes, output.order_logits)
            return heads + (output.out_masks,) if out_masks else heads

    wrapper = Wrapper(model)
    tensor = torch.from_numpy(input_array)
    _LayoutGridSamplePatch.apply()
    try:
        with torch.no_grad():
            framework = wrapper(tensor)
        for index, output in enumerate(framework):
            np.asarray(output.detach().cpu(), dtype=np.float32).tofile(
                work_dir / f"layout_framework_output_{index}.raw"
            )
        output_names = ["logits", "pred_boxes", "order_logits"]
        if out_masks:
            output_names.append("out_masks")
        torch.onnx.export(
            wrapper,
            (tensor,),
            str(raw),
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=[LAYOUT_INPUT_NAME],
            output_names=output_names,
        )
    finally:
        _LayoutGridSamplePatch.undo()

    _normalize_ir_version(raw)
    validate_onnx_output_contract(raw, "layout", allow_symbolic=True, out_masks=out_masks)
    run_pipeline(
        str(raw),
        str(final),
        [
            ClipUndefinedInputsTransform,
            _LayoutFloat64ToFloat32Transform,
            _LayoutFixDanglingClipBoundsTransform,
            _LayoutCapSliceSentinelStartsTransform,
            OnnxSimTransform,
            _LayoutFP16RangeClipTransform,
        ],
    )
    validate_onnx_output_contract(final, "layout", out_masks=out_masks)
    _write_layout_fp32_cross_attention_config(final, work_dir / LAYOUT_NODE_PRECISION_NAME)
    cpu_outputs = _run_ort(final, input_array, input_name=LAYOUT_INPUT_NAME, all_outputs=True)
    for index, output in enumerate(cpu_outputs):
        np.asarray(output, dtype=np.float32).tofile(work_dir / f"layout_cpu_output_{index}.raw")
    for index, output in enumerate(framework):
        _write_fidelity(
            f"layout output {index}",
            np.asarray(output.detach().cpu(), dtype=np.float32),
            np.asarray(cpu_outputs[index], dtype=np.float32),
            hard=True,
        )


def _export_det(work_dir: Path, repo_id: str, sample: Path) -> None:
    stage_dir = work_dir / DET_MODEL
    snapshot = stage_dir / "hf_snapshot"
    source = _download_snapshot(repo_id, snapshot)
    raw = work_dir / f"{DET_MODEL}.onnx"
    final = work_dir / DET_ONNX_NAME
    shutil.copyfile(source, raw)
    _normalize_ir_version(raw)
    validate_onnx_output_contract(raw, "det")
    inputs, _ = preprocess_det_image(str(sample))
    input_array = inputs[INPUT_NAME]
    input_array.tofile(work_dir / "det_input.raw")
    framework = _run_ort(raw, input_array)
    framework.tofile(work_dir / "det_framework_output_0.raw")
    run_pipeline(str(raw), str(final), [
        ClipUndefinedInputsTransform,
        OnnxSimTransform,
        EmptyResizeRoiTransform,
        UnifySymbolicDimsTransform,
        RankDecompose9x9DWConvTransform,
        FP16ClipTransform,
    ])
    assert_symbolic_batch_axis(final, symbol="DynamicDimension.0")
    validate_onnx_output_contract(final, "det")
    cpu = _run_ort(final, input_array)
    cpu.tofile(work_dir / "det_cpu_output.raw")
    _write_fidelity("det", framework, cpu, hard=True)


def _export_rec(work_dir: Path, repo_id: str, sample: Path) -> None:
    stage_dir = work_dir / REC_MODEL
    snapshot = stage_dir / "hf_snapshot"
    source = _download_snapshot(repo_id, snapshot)
    raw = work_dir / f"{REC_MODEL}.onnx"
    final = work_dir / REC_ONNX_NAME
    preclip = work_dir / f"{REC_MODEL}_preclip.onnx"
    shutil.copyfile(source, raw)
    _normalize_ir_version(raw)
    validate_onnx_output_contract(raw, "rec")
    char_dict = _char_dict(snapshot, work_dir / "char_dict.txt")
    image = _load_bgr(str(sample))
    input_array = preprocess_rec_image(image)
    input_array.tofile(work_dir / "rec_input.raw")
    framework = _run_ort(raw, input_array)
    framework.tofile(work_dir / "rec_framework_output_0.raw")
    # rec output is rank 3 (B, T, C); decode the single row's (T, C) matrix.
    decoded = decode_ctc(framework.reshape(framework.shape[1], framework.shape[2]), char_dict)
    print(f"rec sample decode: {decoded['text']!r}")

    transforms = [
        ClipUndefinedInputsTransform,
        FixPaddleShapeReshapesTransform,
        Eliminate5DQKVTransform,
        CtcGreedySummaryTransform,
        StripStaleValueInfoTransform,
        TopoSortTransform,
        FP16ClipTransform,
    ]
    # onnxsim is intentionally absent: its symbolic shape inference corrupts
    # the residual path of this Paddle PIR recognizer graph.
    run_pipeline(str(raw), str(final), transforms)
    assert_symbolic_batch_axis(final, symbol="DynamicDimension.0")
    validate_onnx_output_contract(final, "rec", summary=True)

    import onnx
    from onnx import TensorProto, helper

    # Pre-head repaired graph (the deployable repairs minus the summary head
    # and the FP16 clip): its full probabilities are the graph-surgery
    # fidelity reference.
    pre_model = onnx.load(str(raw))
    for transform in (
        FixPaddleShapeReshapesTransform,
        Eliminate5DQKVTransform,
        StripStaleValueInfoTransform,
        TopoSortTransform,
    ):
        pre_model, _ = transform.apply(pre_model)
    onnx.save(pre_model, str(preclip))
    pre_output = _run_ort(preclip, input_array)
    _write_fidelity("rec graph surgery", framework, pre_output, hard=True)

    # The deployable graph no longer exposes the full probabilities, so verify
    # the summary head on the final graph itself: re-add fetch_name_0 to a
    # check copy and require the summary to be exactly the per-timestep
    # reduction of its own probabilities, with an identical decode.
    check_path = work_dir / "rec_summary_check.onnx"
    check_model = onnx.load(str(final))
    check_model.graph.output.append(helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, None))
    onnx.save(check_model, str(check_path))
    token_ids, max_probs, full = _ort_session(check_path).run(None, {INPUT_NAME: input_array})
    full = np.ascontiguousarray(full, dtype=np.float32)
    if not np.array_equal(token_ids, full.argmax(axis=2).astype(np.int32)):
        raise SystemExit("rec summary head FAILED: ctc_token_ids != argmax(fetch_name_0)")
    if not np.array_equal(max_probs, full.max(axis=2)):
        raise SystemExit("rec summary head FAILED: ctc_max_probs != max(fetch_name_0)")
    if [decode_ctc(full[b], char_dict) for b in range(full.shape[0])] != [
        decode_ctc_summary(token_ids[b], max_probs[b], char_dict) for b in range(full.shape[0])
    ]:
        raise SystemExit("rec summary decode FAILED: summary decode differs from the full-probability decode")
    print("rec summary head verified: ctc_token_ids/ctc_max_probs == per-timestep argmax/max (decode identical)")

    # rec_cpu_output.raw keeps its established meaning (full probabilities) so
    # existing goldens stay comparable; the summary outputs are validated above.
    full.tofile(work_dir / "rec_cpu_output.raw")
    sensitivity = {"framework_vs_fp16ready_cosine": _write_fidelity("rec FP16 sensitivity", framework, full, hard=False)}
    tiny = np.finfo(np.float16).tiny
    denorm_weights = 0
    denorm_tensors = 0
    from onnx import numpy_helper
    raw_model = onnx.load(str(raw))
    for initializer in raw_model.graph.initializer:
        if initializer.data_type != TensorProto.FLOAT:
            continue
        values = numpy_helper.to_array(initializer)
        count = int(((values != 0) & (np.abs(values) < tiny)).sum())
        if count:
            denorm_weights += count
            denorm_tensors += 1
    sensitivity.update({"denormal_weights_zeroed": denorm_weights, "denormal_tensors": denorm_tensors})
    (work_dir / "fp16_sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n")


def _compile_det_latency(work_dir: Path) -> None:
    qaic_compile(DET_ONNX_NAME, 16, work_dir / "qpc_det_latency", cwd=work_dir, mos=4,
                 extra_args=("-onnx-define-symbol=DynamicDimension.0,1",
                             f"-onnx-define-symbol=DynamicDimension.1,{DET_INPUT_H}",
                             f"-onnx-define-symbol=DynamicDimension.2,{DET_INPUT_W}"))


def _write_layout_fp32_cross_attention_config(onnx_path: Path, config_path: Path) -> None:
    """Emit the FP32 precision island that preserves layout-box geometry.

    PP-DocLayoutV3's deformable decoder cross-attention is numerically
    sensitive in FP16.  Selecting the six named cross-attention subgraphs by
    their final ONNX node names avoids baking transformed node-output names
    into the recipe while keeping all other layout work on the accelerator's
    FP16 HMX path.
    """
    import onnx

    pattern = re.compile(r"^/wrapped/model/decoder/layers\.(\d+)/encoder_attn/")
    node_names: list[str] = []
    layers: set[int] = set()
    for node in onnx.load(str(onnx_path)).graph.node:
        match = pattern.match(node.name)
        if match is None or not node.output or not node.output[0]:
            continue
        layers.add(int(match.group(1)))
        node_names.append(node.output[0])
    if layers != set(range(6)) or not node_names:
        raise RuntimeError(
            "layout ONNX is missing one or more PP-DocLayoutV3 decoder cross-attention blocks"
        )
    config_path.write_text(
        "FP32NodeInstanceNames:\n" + "".join(f"  - {name}\n" for name in node_names)
    )


def _compile_layout_latency(work_dir: Path) -> None:
    """Compile the static 704x704 layout artifact with its FP32 attention island."""
    precision_config = work_dir / LAYOUT_NODE_PRECISION_NAME
    if not precision_config.is_file():
        raise FileNotFoundError(f"Missing generated layout precision config: {precision_config}")
    qaic_compile(
        LAYOUT_ONNX_NAME,
        16,
        work_dir / LAYOUT_QPC_SUBDIR,
        cwd=work_dir,
        mos=1,
        node_precision_info=precision_config,
    )


def _compile_rec_latency(work_dir: Path) -> None:
    qaic_compile(REC_ONNX_NAME, 16, work_dir / "qpc_rec_latency", cwd=work_dir, mos=2, ols=1,
                 extra_args=("-onnx-define-symbol=DynamicDimension.0,1",
                             f"-onnx-define-symbol=DynamicDimension.1,{REC_INPUT_W}"))


def _compile_rec_throughput(work_dir: Path) -> None:
    qaic_compile(REC_ONNX_NAME, 4, work_dir / "qpc_rec_throughput", cwd=work_dir, mos=1, ols=2,
                 extra_args=(f"-onnx-define-symbol=DynamicDimension.0,{REC_BATCH}",
                             f"-onnx-define-symbol=DynamicDimension.1,{REC_INPUT_W}"))


# Triton serving config for the rec model. The rec QPC is compiled for 4
# cores and a fixed batch of REC_BATCH, so the serving config is derived from
# the same constants that drive compilation. Each python instance owns its
# own qaic.Session, so N instances give N concurrent rec sessions and reserve
# N * 4 NSPs; 4 instances x 4 cores = 16 NSPs = one full device, the
# canonical device-filling throughput point.
REC_TRITON_INSTANCES = 4
# All four 4-core recognizer sessions fill one 16-NSP SoC.
REC_TRITON_DEVICE_ID = 3
# Each detector instance uses all 16 NSPs on its SoC.
DET_TRITON_DEVICE_IDS = (1, 2)
DET_TRITON_INSTANCE_MAP_NAME = "model_instance_to_card.json"
# Batch-accumulation window (microseconds) for the rec model's dynamic
# batching.
REC_DYNAMIC_BATCHING_MAX_QUEUE_DELAY_US = 5000
REC_PREFERRED_BATCH_SIZES = (1, 2)


def generate_triton_repo(work_dir: Path, repo_root: Path, triton_protobuf=None) -> dict[str, Path]:
    """Generate a Triton model repository for the PP-OCRv6 pipeline.

    Expects ``work_dir`` to hold the layout-latency, det-latency, and
    rec-throughput QPC builds and ``char_dict.txt`` — i.e. a work dir
    produced by :func:`main`. Creates one QPC-backed model per pipeline
    stage under ``repo_root``:

    * ``paddle_ocr_layout`` — Python backend serving the layout stage on
      Cloud AI 100: one raw BGR document image (UINT8 ``[H, W, 3]``) in, the
      layout regions as a JSON string (STRING, ``{"regions": [...]}`` in
      original-image coordinates) and a region count out. The model owns the
      16-core layout QPC session, whose binary is copied next to
      ``model.py``.
    * ``paddle_ocr_det_qpc`` — Python backend serving the det stage on
      Cloud AI 100: one raw BGR image (UINT8 ``[H, W, 3]``; a layout region
      crop from the client) in, the detected line boxes as a JSON string
      (STRING, ``{"boxes": [[x0, y0, x1, y1, score], ...]}`` in region
      coordinates) and a box count out. Two model instances each own a
      16-core det QPC session; the generated mapping file places them on
      QIDs 1 and 2. The QPC binary is copied next to ``model.py``. The
      client orchestrates the pipeline: layout -> region det -> sort ->
      crop -> prepare -> one concurrent ``paddle_ocr_rec`` request per line.
    * ``paddle_ocr_rec`` — Python backend serving the rec stage on Cloud AI
      100: one prepared text line (FP32 ``[3, 48, 1536]``) in, the decoded
      text (STRING) and a confidence out. Dynamic batching coalesces up to
      ``REC_BATCH`` concurrent lines into one QPC pass within the queue
      delay; four instances each own one 4-core rec QPC session on QID 3.
    """
    work_dir = Path(work_dir).resolve()
    repo_root = Path(repo_root).resolve()

    layout_qpc = work_dir / LAYOUT_QPC_SUBDIR / "programqpc.bin"
    det_qpc = work_dir / DET_LATENCY_QPC_SUBDIR / "programqpc.bin"
    rec_qpc = work_dir / REC_THROUGHPUT_QPC_SUBDIR / "programqpc.bin"
    char_dict = work_dir / CHAR_DICT_NAME
    missing = [p for p in (layout_qpc, det_qpc, rec_qpc, char_dict) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Triton generation requires the generated pipeline artifacts; missing: "
            + ", ".join(str(p) for p in missing)
            + ". Run the paddle_ocr export with --work-dir first."
        )

    # Layout stage model. One document image per request (no batching: the
    # layout stage is a single 16-NSP model). The model owns the layout QPC
    # session; the client crops each returned region and dispatches the
    # region crops to paddle_ocr_det_qpc. Device placement is configurable
    # via the device_id parameter.
    layout_root = repo_root / "paddle_ocr_layout"
    layout_cfg = TritonModelConfig(
        model_path=Path(__file__).resolve().parent / "triton_layout_model.py",
        name="paddle_ocr_layout",
        backend="python",
        inputs=[{"name": "src", "data_type": "TYPE_UINT8", "dims": [-1, -1, 3]}],
        outputs=[
            {"name": "regions", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "region_count", "data_type": "TYPE_INT32", "dims": [1]},
        ],
        parameters={"device_id": "0"},
    )
    layout_cfg.tofile(layout_root, triton_protobuf=triton_protobuf)
    # The python backend writes only model.py; place the layout QPC where
    # PaddleOCRLayoutQaic resolves it (next to model.py).
    layout_target = layout_root / str(layout_cfg.model_version) / LAYOUT_QPC_SUBDIR
    layout_target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(layout_qpc, layout_target / "programqpc.bin")

    # rec stage model. One prepared text line per request (the fixed
    # [3, 48, 1536] preprocess output; lines arrive preprocessed because
    # Triton dynamic batching can only stack identical shapes). With
    # max_batch_size > 0 the base config strips the leading batch dim from
    # each spec, so the input is declared per-item and each output per-item
    # [1]. Dynamic batching coalesces up to REC_BATCH concurrent lines into
    # one QPC pass within the queue delay; each of the
    # REC_TRITON_INSTANCES instances owns its own 4-core session, so Triton
    # loads 4 concurrent rec sessions on the dedicated recognizer SoC.
    rec_root = repo_root / "paddle_ocr_rec"
    rec_cfg = TritonModelConfig(
        model_path=Path(__file__).resolve().parent / "triton_rec_model.py",
        name="paddle_ocr_rec",
        backend="python",
        max_batch_size=REC_BATCH,
        count=REC_TRITON_INSTANCES,
        inputs=[{"name": INPUT_NAME, "data_type": "TYPE_FP32", "dims": list(REC_INPUT_SHAPE)}],
        outputs=[
            {"name": "text", "data_type": "TYPE_STRING", "dims": [1, 1]},
            {"name": "confidence", "data_type": "TYPE_FP32", "dims": [1, 1]},
        ],
        parameters={"device_id": str(REC_TRITON_DEVICE_ID)},
        dynamic_batching={
            "preferred_batch_size": list(REC_PREFERRED_BATCH_SIZES),
            "max_queue_delay_microseconds": REC_DYNAMIC_BATCHING_MAX_QUEUE_DELAY_US,
        },
    )
    rec_cfg.tofile(rec_root, triton_protobuf=triton_protobuf)
    # The python backend writes only model.py; place the rec QPC and char
    # dict where PaddleOCRRecQaic resolves them (next to model.py).
    rec_version_dir = rec_root / str(rec_cfg.model_version)
    rec_qpc_target = rec_version_dir / REC_THROUGHPUT_QPC_SUBDIR
    rec_qpc_target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(rec_qpc, rec_qpc_target / "programqpc.bin")
    shutil.copyfile(char_dict, rec_version_dir / CHAR_DICT_NAME)

    # Det stage model. One image per request (no batching: each det instance
    # owns a 16-NSP model). The image is the raw BGR array (HxWx3 uint8,
    # dims dynamic per request) of a layout region crop; the model
    # letterboxes it, so the client only crops. The model owns the det QPC
    # session; recognition is a separate client-orchestrated step against
    # paddle_ocr_rec. Two instances are mapped to QIDs 1 and 2 by the
    # generated model_instance_to_card.json.
    det_qpc_root = repo_root / "paddle_ocr_det_qpc"
    det_qpc_cfg = TritonModelConfig(
        model_path=Path(__file__).resolve().parent / "triton_det_model.py",
        name="paddle_ocr_det_qpc",
        backend="python",
        count=len(DET_TRITON_DEVICE_IDS),
        inputs=[{"name": "src", "data_type": "TYPE_UINT8", "dims": [-1, -1, 3]}],
        outputs=[
            {"name": "boxes", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "box_count", "data_type": "TYPE_INT32", "dims": [1]},
        ],
    )
    det_qpc_cfg.tofile(det_qpc_root, triton_protobuf=triton_protobuf)
    # The python backend writes only model.py; place the det QPC where
    # PaddleOCRDetQaic resolves it (next to model.py).
    det_qpc_target = det_qpc_root / str(det_qpc_cfg.model_version) / DET_LATENCY_QPC_SUBDIR
    det_qpc_target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(det_qpc, det_qpc_target / "programqpc.bin")
    (det_qpc_root / str(det_qpc_cfg.model_version) / DET_TRITON_INSTANCE_MAP_NAME).write_text(
        json.dumps(
            {str(instance_id): [device_id] for instance_id, device_id in enumerate(DET_TRITON_DEVICE_IDS)},
            indent=2,
        )
        + "\n"
    )

    return {
        "layout_model": layout_root,
        "det_qpc_model": det_qpc_root,
        "rec_model": rec_root,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export and compile PP-DocLayoutV3 -> PP-OCRv6 ONNX models.")
    parser.add_argument("--work-dir", required=True, help="Directory for all downloaded and generated artifacts")
    stages = parser.add_argument_group("stage selection")
    stages.add_argument(
        "--layout",
        action="store_true",
        help="Export and compile only the PP-DocLayoutV3 layout stage.",
    )
    parser.add_argument(
        "--out-masks",
        action="store_true",
        help="Retain PP-DocLayoutV3's optional out_masks layout output (off by default).",
    )
    stages.add_argument(
        "--det",
        action="store_true",
        help="Export and compile only the PP-OCRv6 detector stage.",
    )
    stages.add_argument(
        "--rec",
        action="store_true",
        help="Export and compile only the PP-OCRv6 recognizer stage.",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip a stage's ONNX generation when its deployable artifacts already exist under --work-dir "
        "(the QPC compiles still run). Reuse only artifacts generated by the current export code and flags.",
    )
    parser.add_argument("--hf-repo", default=None, help="Use this repository for detector and recognizer snapshots")
    parser.add_argument("--det-hf-repo", default=None)
    parser.add_argument("--rec-hf-repo", default=None)
    parser.add_argument(
        "--det-image",
        default=None,
        help="Sample document image for the layout/detector golden tensors "
             "(default: page 1 of arXiv:1212.1442v1 rendered at 144 DPI into --work-dir/samples)",
    )
    parser.add_argument(
        "--rec-image",
        default=None,
        help="Sample text-line image for the recognizer golden tensor "
             "(default: the PACS-numbers line cropped from the same arXiv page)",
    )
    parser.add_argument(
        "--triton-model-repo",
        default=None,
        help="Optional Triton model repository output directory. Requires the generated layout-latency, "
        "det-latency, and rec-throughput QPCs and char_dict.txt under --work-dir.",
    )
    return parser


def _selected_stages(args: argparse.Namespace) -> tuple[str, ...]:
    """Return explicitly selected stages, or the complete pipeline by default."""
    selected = tuple(stage for stage in ("layout", "det", "rec") if getattr(args, stage))
    return selected or ("layout", "det", "rec")


def _onnx_export_complete(work_dir: Path, stage: str, *, out_masks: bool = False) -> bool:
    """Whether a stage's deployable ONNX artifacts already exist under work_dir.

    A skipped export trusts the existing artifacts, so they are re-checked
    against the output contract before reuse (this also catches a layout
    ONNX generated with a different --out-masks setting). Only reuse
    artifacts produced by the same export code and flags; delete them to
    force a fresh export.
    """
    artifacts = {
        "layout": (LAYOUT_ONNX_NAME, LAYOUT_NODE_PRECISION_NAME),
        "det": (DET_ONNX_NAME,),
        "rec": (REC_ONNX_NAME, CHAR_DICT_NAME),
    }[stage]
    if not all((work_dir / name).is_file() for name in artifacts):
        return False
    deployable = work_dir / artifacts[0]
    if stage == "layout":
        validate_onnx_output_contract(deployable, "layout", out_masks=out_masks)
    elif stage == "rec":
        validate_onnx_output_contract(deployable, "rec", summary=True)
    else:
        validate_onnx_output_contract(deployable, "det")
    print(f"{stage}: reusing existing {artifacts[0]} (skipping ONNX export)")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    stages = _selected_stages(args)
    if args.triton_model_repo and not {"layout", "det", "rec"}.issubset(stages):
        parser.error("--triton-model-repo requires --layout, --det, and --rec (or no stage selectors)")
    if args.out_masks and "layout" not in stages:
        parser.error("--out-masks requires --layout (or no stage selectors)")
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    det_repo = args.det_hf_repo or args.hf_repo or DET_REPO
    rec_repo = args.rec_hf_repo or args.hf_repo or REC_REPO
    det_image = Path(args.det_image).resolve() if args.det_image else None
    rec_image = Path(args.rec_image).resolve() if args.rec_image else None

    def _sample(kind: str, provided: Path | None) -> Path:
        # Resolve a default only when the stage will actually export: a
        # --skip-onnx run must not need the network when the deployable ONNX
        # is already under work_dir.
        if provided is not None:
            return provided
        return _ensure_default_samples(
            work_dir, need_det=kind == "det", need_rec=kind == "rec"
        )[kind]

    # With no selectors this is the full layout -> det -> rec pipeline. Each
    # selected stage is exported independently; its QPC variants compile in
    # parallel because they have separate artifact directories.
    compile_targets = []
    if "layout" in stages:
        if not (args.skip_onnx and _onnx_export_complete(work_dir, "layout", out_masks=args.out_masks)):
            layout_args = (work_dir, LAYOUT_REPO, _sample("det", det_image))
            if args.out_masks:
                _export_layout(*layout_args, out_masks=True)
            else:
                _export_layout(*layout_args)
        compile_targets.append(_compile_layout_latency)
    if "det" in stages:
        if not (args.skip_onnx and _onnx_export_complete(work_dir, "det")):
            det_image = _sample("det", det_image)
            _export_det(work_dir, det_repo, det_image)
        compile_targets.append(_compile_det_latency)
    if "rec" in stages:
        if not (args.skip_onnx and _onnx_export_complete(work_dir, "rec")):
            rec_image = _sample("rec", rec_image)
            _export_rec(work_dir, rec_repo, rec_image)
        compile_targets.extend((_compile_rec_latency, _compile_rec_throughput))
    with ThreadPoolExecutor(max_workers=len(compile_targets)) as executor:
        futures = [executor.submit(fn, work_dir) for fn in compile_targets]
        for future in as_completed(futures):
            future.result()  # re-raise the first compile failure

    if args.triton_model_repo:
        generated = generate_triton_repo(work_dir, Path(args.triton_model_repo).resolve())
        print(f"[export] wrote Triton model repository entries to {Path(args.triton_model_repo).resolve()}")
        for name, path in generated.items():
            print(f"[export] Triton {name}: {path}")


if __name__ == "__main__":
    main()
