#!/usr/bin/env python3
####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
#
# base_onnx_transform.py — Base class framework for ONNX graph surgery
#
# Mirrors the QEfficient.base.onnx_transforms pattern used in the efficient-transformers repo.
#
# Library classes (no-op gracefully when pattern not present):
#
#   ClipUndefinedInputsTransform — remove empty-string optional inputs from Clip nodes.
#                                  torch.onnx.export at opset 17+ emits Clip nodes with ""
#                                  placeholders for absent min/max inputs; OnnxSimTransform
#                                  fails to validate graphs containing these. Included in
#                                  the default pipeline — safe to run unconditionally.
#
#   OnnxSimTransform           — run onnxsim on the loaded model proto: constant-folds,
#                                topologically sorts, removes dead nodes, inlines static
#                                shape expressions.
#                                Raises RuntimeError if onnxsim is not installed or check=False.
#
#   FP16ClipTransform          — clip FP32 initializers to FP16 range; zero denormals.
#                                Apply to every FP32 model compiled with -convert-to-fp16.
#
#   assert_symbolic_batch_axis — export-script guardrail: raise if run_pipeline()
#                                or model-specific transforms erased batch_size.
#
#   GlobalAveragePoolFusion    — replace ReduceMean(keepdims=1, axes=all_spatial) with
#                                GlobalAveragePool. The compiler can always infer
#                                GlobalAveragePool output shape; it cannot always infer
#                                ReduceMean output shape even with constant axes.
#                                Included in the default pipeline — no-ops when absent.
#
#   ErfcReplaceTransform       — replace Erfc(x) with Sub(1, Erf(x)).
#                                Erfc is not in ONNX opsets 1-19; any model with Erfc nodes
#                                will fail to compile on standard targets.
#
#   LayerNormFusion            — fuse the expanded PyTorch/HuggingFace LayerNorm subgraph
#                                (ReduceMean→Sub→Pow→ReduceMean→Add→Sqrt→Div→Mul→Add)
#                                into a single ONNX LayerNormalization op (opset 17).
#
# CLI default (python base_onnx_transform.py in.onnx out.onnx):
#   Runs ClipUndefinedInputsTransform + OnnxSimTransform + GlobalAveragePoolFusion
#        + FP16ClipTransform.
#   Pass the raw export ONNX directly.
#
# Usage as a library (export script or optimize_graph.py):
#
#   from base_onnx_transform import (BaseOnnxTransform, OnnxTransformPipeline,
#                                     FP16ClipTransform, ErfcReplaceTransform,
#                                     LayerNormFusion, run_pipeline)
#
#   class MyModelTransform(BaseOnnxTransform):
#       @classmethod
#       def apply(cls, model):
#           # ... pattern match and replace ...
#           return model, changed
#
#   if __name__ == "__main__":
#       import sys
#       run_pipeline(sys.argv[1], sys.argv[2],
#                    [FP16ClipTransform, ErfcReplaceTransform, LayerNormFusion, MyModelTransform])

import sys
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseOnnxTransform:
    """Base class for ONNX graph transforms. Do not instantiate — use the `apply` class method."""

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` class method.")

    @classmethod
    def apply(cls, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
        raise NotImplementedError("Subclasses must implement apply()")


# ---------------------------------------------------------------------------
# ClipUndefinedInputsTransform
# ---------------------------------------------------------------------------

class ClipUndefinedInputsTransform(BaseOnnxTransform):
    """Remove empty-string optional inputs from Clip nodes.

    torch.onnx.export at opset 17+ emits Clip nodes with "" placeholders for
    absent optional min/max inputs (indices 1 and 2). OnnxSimTransform fails
    to validate graphs containing these nodes. This transform trims trailing
    empty-string inputs from every Clip node so OnnxSimTransform receives a
    valid graph.

    Included in the default pipeline — safe to run unconditionally; no-ops
    when no Clip nodes have undefined inputs.
    """

    @classmethod
    def apply(cls, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
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
        if fixed:
            log.info("ClipUndefinedInputsTransform: fixed %d Clip node(s)", fixed)
        return model, fixed > 0


# ---------------------------------------------------------------------------
# OnnxSimTransform
# ---------------------------------------------------------------------------

class OnnxSimTransform(BaseOnnxTransform):
    """Run onnxsim: topological sort, constant-fold, dead-node removal.

    Raises RuntimeError if onnxsim is not installed or if check=False.
    """

    @classmethod
    def apply(cls, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
        try:
            import onnxsim
        except ImportError:
            raise RuntimeError(
                "onnxsim is not installed. Run: pip install onnxsim"
            )
        before = len(model.graph.node)
        simplified, check = onnxsim.simplify(model)
        if not check:
            raise RuntimeError(
                "onnxsim.simplify() check=False — simplified model failed validation. "
                "Inspect the export for unsupported ops or dynamic shapes before retrying."
            )
        after = len(simplified.graph.node)
        log.info("OnnxSimTransform: %d → %d nodes (%+d)", before, after, after - before)
        return simplified, after != before


# ---------------------------------------------------------------------------
# FP16ClipTransform
# ---------------------------------------------------------------------------

class FP16ClipTransform(BaseOnnxTransform):
    """Clip FP32 initializer tensors to the FP16 representable range.

    Prevents sub-normal values from silently underflowing to zero when
    qaic-compile applies -convert-to-fp16. Also zeros denormal values
    (|v| < fp16.tiny, v != 0) since FP16 hardware flushes denormals to zero.

    Apply to every FP32 ONNX model compiled with -convert-to-fp16.
    """

    @classmethod
    def apply(cls, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
        # Use np.float32 bounds — Python float is float64, and np.clip/np.where upcast
        # float32 arrays to float64 when bounds are float64, producing float64 initializers
        # that break downstream Where/Cast nodes in ORT and qaic-compile.
        fp16_max = np.float32(np.finfo(np.float16).max)
        fp16_min = np.float32(np.finfo(np.float16).min)
        fp16_tiny = np.float32(np.finfo(np.float16).tiny)
        changed = False
        clipped = 0

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
            clipped += 1

        if changed:
            log.info("FP16ClipTransform: clipped %d initializer tensors", clipped)
        return model, changed


# ---------------------------------------------------------------------------
# GlobalAveragePoolFusion
# ---------------------------------------------------------------------------

class GlobalAveragePoolFusion(BaseOnnxTransform):
    """Replace ReduceMean(keepdims=1, axes=all_spatial) with GlobalAveragePool.

    When a model uses ReduceMean with keepdims=1 to perform global average pooling
    over all spatial dimensions, the compiler may fail to infer the output shape even
    when the axes are a constant initializer. GlobalAveragePool has an unambiguous
    output shape (N, C, 1, ..., 1) that the compiler always infers correctly.

    Fires when a ReduceMean node has keepdims=1, axes are constant (attribute or
    initializer), and the axes cover every spatial dimension (dim ≥ 2) of the input.
    GlobalAveragePool preserves the output tensor name so downstream consumers
    (e.g. Reshape, Flatten, Gemm) require no changes. Orphaned axes initializers
    used only by the replaced node are removed. No-ops when the pattern is absent.
    """

    @classmethod
    def _get_axes(cls, node: onnx.NodeProto, init_map: dict) -> Optional[List[int]]:
        for attr in node.attribute:
            if attr.name == "axes":
                return list(attr.ints)
        if len(node.input) >= 2 and node.input[1] in init_map:
            return init_map[node.input[1]].flatten().tolist()
        return None

    @classmethod
    def apply(cls, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
        g = model.graph
        init_map = {t.name: numpy_helper.to_array(t) for t in g.initializer}

        rank_map: Dict[str, int] = {}
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                rank_map[vi.name] = len(tt.shape.dim)

        init_refs: Counter = Counter(inp for n in g.node for inp in n.input if inp in init_map)

        # Key nodes by output[0] name — stable across protobuf mutations.
        # id(node) becomes stale after g.initializer[:] = [...] because protobuf
        # may return new Python wrapper objects when the parent message is mutated.
        replacements: Dict[str, onnx.NodeProto] = {}
        axes_to_remove: set = set()
        replaced = 0

        for node in g.node:
            if node.op_type != "ReduceMean":
                continue
            keepdims = next((a.i for a in node.attribute if a.name == "keepdims"), 1)
            if keepdims != 1:
                continue
            axes = cls._get_axes(node, init_map)
            if not axes:
                continue
            rank = rank_map.get(node.input[0])
            if rank is None or rank < 3:
                continue
            spatial_dims = set(range(2, rank))
            if {int(a) % rank for a in axes} != spatial_dims:
                continue

            gap = helper.make_node(
                "GlobalAveragePool",
                inputs=[node.input[0]],
                outputs=list(node.output),
                name=(node.name or node.output[0]) + "_gap",
            )
            replacements[node.output[0]] = gap
            if len(node.input) >= 2 and node.input[1] and init_refs.get(node.input[1], 0) == 1:
                axes_to_remove.add(node.input[1])
            replaced += 1

        if not replaced:
            return model, False

        if axes_to_remove:
            keep = [t for t in g.initializer if t.name not in axes_to_remove]
            del g.initializer[:]
            g.initializer.extend(keep)
        new_nodes = [
            replacements[n.output[0]] if n.op_type == "ReduceMean" and n.output[0] in replacements else n
            for n in g.node
        ]
        del g.node[:]
        g.node.extend(new_nodes)
        log.info(
            "GlobalAveragePoolFusion: replaced %d ReduceMean(keepdims=1, all-spatial) "
            "with GlobalAveragePool",
            replaced,
        )
        return model, True


# ---------------------------------------------------------------------------
# ErfcReplaceTransform
# ---------------------------------------------------------------------------

class ErfcReplaceTransform(BaseOnnxTransform):
    """Replace Erfc(x) → Sub(const_1, Erf(x)).

    Erfc is not defined in ONNX opsets 1-19. Any model with Erfc nodes will
    fail to compile on targets that do not support opset 20+. The replacement
    1 - erf(x) is mathematically exact (erfc(x) = 1 - erf(x) by definition).
    """

    _CONST_ONE = "__base_erfc_const_one__"

    @classmethod
    def apply(cls, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
        g = model.graph
        erfc_nodes = [n for n in g.node if n.op_type == "Erfc"]
        if not erfc_nodes:
            return model, False

        if not any(t.name == cls._CONST_ONE for t in g.initializer):
            g.initializer.append(
                numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=cls._CONST_ONE)
            )

        remove_ids = {id(n) for n in erfc_nodes}
        inserts: Dict[int, list] = {}
        for i, erfc in enumerate(erfc_nodes):
            erf_out = f"__erfc_erf_{i}__"
            inserts[id(erfc)] = [
                helper.make_node("Erf", inputs=[erfc.input[0]], outputs=[erf_out]),
                helper.make_node("Sub", inputs=[cls._CONST_ONE, erf_out], outputs=list(erfc.output)),
            ]

        new_nodes = []
        for n in g.node:
            if id(n) in remove_ids:
                new_nodes.extend(inserts[id(n)])
            else:
                new_nodes.append(n)
        del g.node[:]
        g.node.extend(new_nodes)
        log.info("ErfcReplaceTransform: replaced %d Erfc nodes with Sub(1, Erf(x))", len(erfc_nodes))
        return model, True


# ---------------------------------------------------------------------------
# LayerNormFusion
# ---------------------------------------------------------------------------

class LayerNormFusion(BaseOnnxTransform):
    """Fuse the expanded LayerNorm subgraph into ONNX LayerNormalization (opset 17).

    Matches the standard PyTorch / HuggingFace export pattern:

        X ──┬── ReduceMean(axes, keepdims=1) ──────────────────────── mean
            │                                                            │
            └── Sub(X, mean) ──────────────────────────────────── diff ─┤
                    │                                                    │
                    ├── Pow(2) → ReduceMean → Add(eps) → Sqrt ──── std  │
                    │                                                    │
                    └───────────────── Div(diff, std) → Mul(scale) → Add(bias) → output

    Replaces with: LayerNormalization(X, scale, bias, axis=axis, epsilon=eps)

    Handles both Sub(X, mean) and Sub(mean, X) input orderings. No-ops
    gracefully when the pattern is not present.
    """

    @classmethod
    def apply(cls, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
        g = model.graph

        out2node = {o: n for n in g.node for o in n.output}
        init_map = {t.name: t for t in g.initializer}

        def producer(name: str, op: Optional[str] = None) -> Optional[onnx.NodeProto]:
            n = out2node.get(name)
            if n is None or (op and n.op_type != op):
                return None
            return n

        def scalar_init(name: str) -> Optional[float]:
            t = init_map.get(name)
            if t is None:
                return None
            arr = numpy_helper.to_array(t)
            return float(arr.flat[0]) if arr.size == 1 else None

        def get_axes(node: onnx.NodeProto) -> Optional[List[int]]:
            for a in node.attribute:
                if a.name == "axes":
                    return list(a.ints)
            return None

        def try_match(add_bias: onnx.NodeProto):
            """Return (X, scale, bias, axis, eps, node_id_set, output) or None."""
            # Final Add(scaled, bias) — one input must be an initializer (bias)
            bias_name = next((i for i in add_bias.input if i in init_map), None)
            scaled_out = next((i for i in add_bias.input if i not in init_map), None)
            if not bias_name or not scaled_out:
                return None

            mul = producer(scaled_out, "Mul")
            if not mul:
                return None
            scale_name = next((i for i in mul.input if i in init_map), None)
            normed_out = next((i for i in mul.input if i not in init_map), None)
            if not scale_name or not normed_out:
                return None

            div = producer(normed_out, "Div")
            if not div or len(div.input) < 2:
                return None
            diff_name, std_out = div.input[0], div.input[1]

            sqrt = producer(std_out, "Sqrt")
            if not sqrt:
                return None

            add_eps = producer(sqrt.input[0], "Add")
            if not add_eps:
                return None
            eps = next((scalar_init(i) for i in add_eps.input if scalar_init(i) is not None), None)
            var_out = next((i for i in add_eps.input if scalar_init(i) is None), None)
            if eps is None or not var_out:
                return None

            rm_var = producer(var_out, "ReduceMean")
            if not rm_var:
                return None
            axes_var = get_axes(rm_var)

            pow_node = producer(rm_var.input[0], "Pow")
            if not pow_node or len(pow_node.input) < 2:
                return None
            if not np.isclose(scalar_init(pow_node.input[1]) or 0.0, 2.0):
                return None
            # The Pow input must be the same diff that feeds Div
            if pow_node.input[0] != diff_name:
                return None

            sub = producer(diff_name, "Sub")
            if not sub or len(sub.input) < 2:
                return None

            # Try both Sub(X, mean) and Sub(mean, X)
            X_name = None
            rm1 = None
            for idx in range(2):
                rm_candidate = producer(sub.input[idx], "ReduceMean")
                if (rm_candidate
                        and rm_candidate.input[0] == sub.input[1 - idx]
                        and get_axes(rm_candidate) == axes_var):
                    X_name = sub.input[1 - idx]
                    rm1 = rm_candidate
                    break
            if X_name is None:
                return None

            axis = axes_var[-1] if axes_var else -1
            node_ids = {id(n) for n in [rm1, sub, pow_node, rm_var, add_eps, sqrt, div, mul, add_bias]}
            return (X_name, scale_name, bias_name, axis, eps, node_ids, add_bias.output[0])

        fusions = []
        seen: set = set()
        for node in g.node:
            if node.op_type == "Add" and id(node) not in seen:
                result = try_match(node)
                if result:
                    fusions.append(result)
                    seen.update(result[5])

        if not fusions:
            return model, False

        # Ensure opset ≥ 17 (LayerNormalization added in opset 17)
        for op in model.opset_import:
            if op.domain == "" and op.version < 17:
                op.version = 17

        remove_ids = set().union(*(f[5] for f in fusions))
        new_nodes = [n for n in g.node if id(n) not in remove_ids]
        for X_name, scale_name, bias_name, axis, eps, _, ln_out in fusions:
            new_nodes.append(helper.make_node(
                "LayerNormalization",
                inputs=[X_name, scale_name, bias_name],
                outputs=[ln_out],
                axis=axis,
                epsilon=float(eps),
            ))
        del g.node[:]
        g.node.extend(new_nodes)
        log.info("LayerNormFusion: fused %d LayerNorm subgraphs (axis=%s)", len(fusions), axis)
        return model, True


# ---------------------------------------------------------------------------
# Pipeline and helpers
# ---------------------------------------------------------------------------

class OnnxTransformPipeline:
    """Apply a sequence of BaseOnnxTransform subclasses in order."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def apply(self, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, bool]:
        any_changed = False
        for t in self.transforms:
            model, changed = t.apply(model)
            any_changed = any_changed or changed
            log.info("Transform %s: applied=%s", t.__name__, changed)
        return model, any_changed


def _op_counts(model: onnx.ModelProto) -> Dict[str, int]:
    return dict(Counter(n.op_type for n in model.graph.node))


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


def run_pipeline(input_path: str, output_path: str, transforms: List) -> None:
    log.info("Loading %s", input_path)
    model = onnx.load(input_path)
    before = _op_counts(model)
    log.info("Before: %d total nodes", sum(before.values()))

    pipeline = OnnxTransformPipeline(transforms)
    model, changed = pipeline.apply(model)

    after = _op_counts(model)
    log.info("After:  %d total nodes", sum(after.values()))

    for op in sorted(set(before) | set(after)):
        b, a = before.get(op, 0), after.get(op, 0)
        if b != a:
            log.info("  %-35s %3d → %3d  (%+d)", op, b, a, a - b)

    if not changed:
        log.warning("No transforms applied — output is identical to input")

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    log.info("Saved: %s", output_path)


if __name__ == "__main__":
    # --list: print all available BaseOnnxTransform subclasses and their descriptions.
    # Usage: python base_onnx_transform.py --list
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        import inspect
        subclasses = [
            cls for name, cls in inspect.getmembers(
                sys.modules[__name__], inspect.isclass
            )
            if issubclass(cls, BaseOnnxTransform) and cls is not BaseOnnxTransform
        ]
        print("Available transforms in base_onnx_transform.py:")
        for cls in subclasses:
            doc = (cls.__doc__ or "").strip().splitlines()[0]
            print(f"  {cls.__name__:<30} {doc}")
        sys.exit(0)

    # CLI default: ClipUndefinedInputsTransform + OnnxSimTransform + GlobalAveragePoolFusion
    # + FP16ClipTransform.
    # For model-specific transforms import this module and call run_pipeline() directly.
    # Usage: python base_onnx_transform.py <input.onnx> <output.onnx>
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output.onnx>", file=sys.stderr)
        print(f"       {sys.argv[0]} --list", file=sys.stderr)
        sys.exit(1)
    run_pipeline(sys.argv[1], sys.argv[2], [
        ClipUndefinedInputsTransform,
        OnnxSimTransform,
        GlobalAveragePoolFusion,
        FP16ClipTransform,
    ])
