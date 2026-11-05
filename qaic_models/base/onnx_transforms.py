# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, List, Tuple

log = logging.getLogger(__name__)


class BaseOnnxTransform:
    @classmethod
    def apply(cls, model: Any) -> Tuple[Any, bool]:
        raise NotImplementedError


class ClipUndefinedInputsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: Any) -> Tuple[Any, bool]:
        fixed = 0
        for node in model.graph.node:
            if node.op_type != "Clip":
                continue
            inputs = list(node.input)
            while inputs and inputs[-1] == "":
                inputs.pop()
            if len(inputs) != len(node.input):
                node.input[:] = inputs
                fixed += 1
        return model, fixed > 0


class OnnxSimTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: Any) -> Tuple[Any, bool]:
        try:
            import onnxsim
        except ImportError as exc:
            raise RuntimeError("onnxsim is not installed. Run: pip install onnxsim") from exc
        before = len(model.graph.node)
        simplified, check = onnxsim.simplify(model)
        if not check:
            raise RuntimeError("onnxsim.simplify() check=False — simplified model failed validation")
        after = len(simplified.graph.node)
        log.info("OnnxSimTransform: %d -> %d nodes", before, after)
        return simplified, after != before


class FP16ClipTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: Any) -> Tuple[Any, bool]:
        import numpy as np
        from onnx import TensorProto, numpy_helper

        fp16_max = np.float32(np.finfo(np.float16).max)
        fp16_min = np.float32(np.finfo(np.float16).min)
        fp16_tiny = np.float32(np.finfo(np.float16).tiny)
        changed = False
        for tensor in model.graph.initializer:
            if tensor.data_type != TensorProto.FLOAT:
                continue
            arr = numpy_helper.to_array(tensor).copy()
            overflow = (arr > fp16_max) | (arr < fp16_min)
            denormal = (arr != 0.0) & (np.abs(arr) < fp16_tiny)
            if not (overflow.any() or denormal.any()):
                continue
            arr = np.clip(arr, fp16_min, fp16_max)
            arr = np.where(denormal, np.float32(0.0), arr)
            tensor.CopyFrom(numpy_helper.from_array(arr, tensor.name))
            changed = True
        return model, changed


class ReduceMeanAxesInputToAttrTransform(BaseOnnxTransform):
    """Move ReduceMean axes from input tensor to attribute (opset 18 -> opset 13 style).

    The Cloud AI 100 compiler requires ReduceMean axes as an attribute. Newer
    PyTorch ONNX exporters (opset 18) encode axes as an input tensor. This
    transform moves constant axes inputs to attributes and downgrades the
    default-domain opset to 13 so the axes-as-attribute form is valid.
    No-ops gracefully when all ReduceMean nodes already use attribute axes.
    """

    @classmethod
    def apply(cls, model: Any) -> Tuple[Any, bool]:
        from onnx import helper, numpy_helper

        init_map = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
        fixed = 0
        for node in model.graph.node:
            if node.op_type != "ReduceMean" or len(node.input) < 2:
                continue
            axes_name = node.input[1]
            if axes_name not in init_map:
                log.warning("ReduceMeanAxesInputToAttrTransform: axes input %r is not a constant - skipping", axes_name)
                continue
            axes = [int(a) for a in init_map[axes_name].flatten().tolist()]
            del node.input[1]
            for attr in list(node.attribute):
                if attr.name == "noop_with_empty_axes":
                    node.attribute.remove(attr)
            node.attribute.append(helper.make_attribute("axes", axes))
            log.info("ReduceMeanAxesInputToAttrTransform: %s axes=%s moved to attribute", node.output[0], axes)
            fixed += 1

        if fixed:
            for op in model.opset_import:
                if op.domain == "":
                    log.info("ReduceMeanAxesInputToAttrTransform: downgrading opset %d -> 13", op.version)
                    op.version = 13
        return model, fixed > 0


class GlobalAveragePoolFusionTransform(BaseOnnxTransform):
    """Replace all-spatial ReduceMean(keepdims=1) with GlobalAveragePool."""

    @classmethod
    def _get_axes(cls, node: Any, init_map: dict) -> list[int] | None:
        for attr in node.attribute:
            if attr.name == "axes":
                return list(attr.ints)
        if len(node.input) >= 2 and node.input[1] in init_map:
            return init_map[node.input[1]].flatten().tolist()
        return None

    @classmethod
    def apply(cls, model: Any) -> Tuple[Any, bool]:
        from collections import Counter
        from onnx import helper, numpy_helper

        graph = model.graph
        init_map = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
        rank_map = {}
        for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
            tensor_type = value_info.type.tensor_type
            if tensor_type.HasField("shape"):
                rank_map[value_info.name] = len(tensor_type.shape.dim)

        init_refs = Counter(inp for node in graph.node for inp in node.input if inp in init_map)
        replacements = {}
        axes_to_remove = set()

        for node in graph.node:
            if node.op_type != "ReduceMean":
                continue
            keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)
            if keepdims != 1:
                continue
            axes = cls._get_axes(node, init_map)
            rank = rank_map.get(node.input[0])
            if not axes or rank is None or rank < 3:
                continue
            if {int(axis) % rank for axis in axes} != set(range(2, rank)):
                continue

            replacements[node.output[0]] = helper.make_node(
                "GlobalAveragePool",
                inputs=[node.input[0]],
                outputs=list(node.output),
                name=(node.name or node.output[0]) + "_gap",
            )
            if len(node.input) >= 2 and node.input[1] and init_refs.get(node.input[1], 0) == 1:
                axes_to_remove.add(node.input[1])

        if not replacements:
            return model, False

        if axes_to_remove:
            keep_initializers = [tensor for tensor in graph.initializer if tensor.name not in axes_to_remove]
            del graph.initializer[:]
            graph.initializer.extend(keep_initializers)

        new_nodes = [
            replacements[node.output[0]]
            if node.op_type == "ReduceMean" and node.output[0] in replacements
            else node
            for node in graph.node
        ]
        del graph.node[:]
        graph.node.extend(new_nodes)
        return model, True


def _describe_dim(dim) -> str:
    if dim.HasField("dim_param"):
        return dim.dim_param
    if dim.HasField("dim_value"):
        return str(dim.dim_value)
    return "unknown"


def assert_symbolic_batch_axis(model_or_path, symbol: str = "batch_size", require_outputs: bool = True) -> None:
    """Raise if graph inputs/outputs lost the symbolic batch axis.

    Export scripts call this after run_pipeline() so onnxsim or model-specific
    transforms cannot silently turn configurable batch into a static value.
    """
    import onnx

    model = model_or_path if hasattr(model_or_path, "graph") else onnx.load(str(model_or_path))
    values = list(model.graph.input)
    if require_outputs:
        values.extend(model.graph.output)

    bad = []
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape") or not tensor_type.shape.dim:
            continue
        dim0 = tensor_type.shape.dim[0]
        if not (dim0.HasField("dim_param") and dim0.dim_param == symbol):
            bad.append(f"{value.name}: first dim is {_describe_dim(dim0)!r}, expected {symbol!r}")

    if bad:
        details = "\n  - ".join(bad)
        raise RuntimeError(
            "ONNX symbolic batch check failed after run_pipeline(). "
            "Do not set dynamic_axes=None, do not bake batch=1, and do not let "
            "onnxsim/model-specific transforms erase the batch_size symbol. "
            "Fix the export script so the batch axis remains configurable. "
            "Offending tensors:\n  - " + details
        )


def _op_counts(model: Any) -> dict[str, int]:
    return dict(Counter(n.op_type for n in model.graph.node))


def run_pipeline(input_path: str, output_path: str, transforms: List[type[BaseOnnxTransform]]) -> None:
    import onnx

    log.info("Loading %s", input_path)
    model = onnx.load(str(input_path))
    before = _op_counts(model)
    changed_any = False
    for transform in transforms:
        model, changed = transform.apply(model)
        changed_any = changed_any or changed
        log.info("Transform %s: applied=%s", transform.__name__, changed)
    after = _op_counts(model)
    for op in sorted(set(before) | set(after)):
        if before.get(op, 0) != after.get(op, 0):
            log.info("  %-35s %3d -> %3d", op, before.get(op, 0), after.get(op, 0))
    if not changed_any:
        log.warning("No transforms applied — output may be identical to input")
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    log.info("Saved: %s", output_path)
