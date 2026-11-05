# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Recipe-specific ONNX graph repairs for PaddleOCR export."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from qaic_models.base.onnx_transforms import BaseOnnxTransform

from .postprocess import OUTPUT_NAME, REC_MAX_PROBS_NAME, REC_NUM_CLASSES, REC_TOKEN_IDS_NAME
from .preprocess import INPUT_NAME, REC_INPUT_SHAPE


class EmptyResizeRoiTransform(BaseOnnxTransform):
    """Remove empty FLOAT ROI initializers from optional Resize inputs."""

    @classmethod
    def apply(cls, model: Any):
        import onnx
        from onnx import numpy_helper

        empty = {
            init.name
            for init in model.graph.initializer
            if init.data_type == onnx.TensorProto.FLOAT and numpy_helper.to_array(init).size == 0
        }
        changed = False
        for node in model.graph.node:
            if node.op_type == "Resize" and len(node.input) > 1 and node.input[1] in empty:
                node.input[1] = ""
                changed = True
        if changed:
            used = {value for node in model.graph.node for value in node.input if value}
            kept = [init for init in model.graph.initializer if init.name not in empty or init.name in used]
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept)
        return model, changed


class UnifySymbolicDimsTransform(BaseOnnxTransform):
    """Use input symbolic names for output axes with the same positions."""

    @classmethod
    def apply(cls, model: Any):
        changed = False
        if not model.graph.input or not model.graph.output:
            return model, changed
        input_dims = model.graph.input[0].type.tensor_type.shape.dim
        symbols = [dim.dim_param if dim.HasField("dim_param") else None for dim in input_dims]
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            if len(dims) != len(symbols):
                continue
            for index, dim in enumerate(dims):
                if dim.HasField("dim_param") and symbols[index] and dim.dim_param != symbols[index]:
                    dim.dim_param = symbols[index]
                    changed = True
        return model, changed


class RankDecompose9x9DWConvTransform(BaseOnnxTransform):
    """Replace target 9x9 depthwise convolutions with HMX-friendly rank terms."""

    FP16_MAX = 65504.0
    FP16_TINY = 6.103515625e-05

    @classmethod
    def _clip_fp16(cls, array):
        array = np.clip(array.astype(np.float32), -cls.FP16_MAX, cls.FP16_MAX)
        array[np.abs(array) < cls.FP16_TINY] = 0.0
        return array

    @classmethod
    def _is_target(cls, node, initializers):
        if node.op_type != "Conv" or node.input[1] not in initializers:
            return False
        weight = initializers[node.input[1]]
        if len(weight.dims) != 4:
            return False
        channels, input_group, kh, kw = weight.dims
        attrs = {attr.name: attr for attr in node.attribute}
        group = attrs["group"].i if "group" in attrs else 1
        pads = list(attrs["pads"].ints) if "pads" in attrs else [0, 0, 0, 0]
        strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
        dilations = list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1]
        return (
            kh == 9 and kw == 9 and input_group == 1 and group == channels
            and strides == [1, 1] and dilations == [1, 1] and pads == [4, 4, 4, 4]
            and (len(node.input) <= 2 or node.input[2] in initializers)
        )

    @classmethod
    def apply(cls, model: Any):
        from onnx import helper, numpy_helper

        initializers = {item.name: item for item in model.graph.initializer}
        nodes = list(model.graph.node)
        new_initializers = []
        targets = [i for i, node in enumerate(nodes) if cls._is_target(node, initializers)]
        for index in reversed(targets):
            node = nodes[index]
            weight = numpy_helper.to_array(initializers[node.input[1]])
            channels = weight.shape[0]
            bias = node.input[2] if len(node.input) > 2 else ""
            u, singular, vt = np.linalg.svd(weight[:, 0], full_matrices=False)
            height_terms = (u * singular[:, None, :]).transpose(0, 2, 1)
            width_terms = vt
            base = f"rd9_{node.name}"
            term_outputs = []
            replacement = []
            for rank in range(9):
                height = height_terms[:, rank, :]
                top = height[:, :5].copy()
                top[:, 4] *= 0.5
                bottom = np.empty_like(height[:, 4:9])
                bottom[:, 0] = height[:, 4] * 0.5
                bottom[:, 1:] = height[:, 5:]
                kernels = (
                    cls._clip_fp16(top).reshape(channels, 1, 5, 1),
                    cls._clip_fp16(bottom).reshape(channels, 1, 5, 1),
                    cls._clip_fp16(width_terms[:, rank, :]).reshape(channels, 1, 1, 9),
                )
                names = [f"{base}_w_{rank}_{suffix}" for suffix in ("top", "bottom", "width")]
                for name, kernel in zip(names, kernels):
                    new_initializers.append(numpy_helper.from_array(kernel, name))
                top_out, bottom_out = f"{base}_top_{rank}", f"{base}_bottom_{rank}"
                add_out, term_out = f"{base}_add_{rank}", f"{base}_term_{rank}"
                replacement.extend([
                    helper.make_node("Conv", [node.input[0], names[0]], [top_out], kernel_shape=[5, 1], pads=[4, 0, 0, 0], strides=[1, 1], dilations=[1, 1], group=channels, name=f"{base}_conv_top_{rank}"),
                    helper.make_node("Conv", [node.input[0], names[1]], [bottom_out], kernel_shape=[5, 1], pads=[0, 0, 4, 0], strides=[1, 1], dilations=[1, 1], group=channels, name=f"{base}_conv_bottom_{rank}"),
                    helper.make_node("Add", [top_out, bottom_out], [add_out], name=f"{base}_add_height_{rank}"),
                    helper.make_node("Conv", [add_out, names[2]] + ([bias] if rank == 8 and bias else []), [term_out], kernel_shape=[1, 9], pads=[0, 4, 0, 4], strides=[1, 1], dilations=[1, 1], group=channels, name=f"{base}_conv_width_{rank}"),
                ])
                term_outputs.append(term_out)
            level = term_outputs
            add_index = 0
            while len(level) > 1:
                next_level = []
                for left in range(0, len(level) - 1, 2):
                    add_index += 1
                    output = node.output[0] if len(level) == 2 else f"{base}_sum_{add_index}"
                    replacement.append(helper.make_node("Add", [level[left], level[left + 1]], [output], name=f"{base}_sum_node_{add_index}"))
                    next_level.append(output)
                if len(level) % 2:
                    next_level.append(level[-1])
                level = next_level
            nodes[index:index + 1] = replacement
        if not targets:
            return model, False
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        used = {value for node in nodes for value in node.input}
        kept = [item for item in model.graph.initializer if item.name in used]
        kept.extend(new_initializers)
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        return model, True


class TopoSortTransform(BaseOnnxTransform):
    """Stable topological sort for Paddle PIR graphs."""

    @classmethod
    def apply(cls, model: Any):
        import heapq

        nodes = list(model.graph.node)
        producers = {output: node for node in nodes for output in node.output}
        indegree = {id(node): 0 for node in nodes}
        dependents = {id(node): [] for node in nodes}
        positions = {id(node): index for index, node in enumerate(nodes)}
        for node in nodes:
            for value in node.input:
                producer = producers.get(value)
                if producer is not None and producer is not node:
                    indegree[id(node)] += 1
                    dependents[id(producer)].append(node)
        queue = [(positions[id(node)], node) for node in nodes if indegree[id(node)] == 0]
        heapq.heapify(queue)
        ordered = []
        while queue:
            _, node = heapq.heappop(queue)
            ordered.append(node)
            for dependent in dependents[id(node)]:
                indegree[id(dependent)] -= 1
                if indegree[id(dependent)] == 0:
                    heapq.heappush(queue, (positions[id(dependent)], dependent))
        if len(ordered) != len(nodes):
            raise RuntimeError("TopoSortTransform: cycle detected in graph")
        changed = ordered != nodes
        del model.graph.node[:]
        model.graph.node.extend(ordered)
        return model, changed


class FixPaddleShapeReshapesTransform(BaseOnnxTransform):
    """Replace the four runtime-computed rec Reshape shapes with static ops."""

    @classmethod
    def apply(cls, model: Any):
        import onnxruntime as ort
        from onnx import TensorProto, helper

        representative = np.zeros(REC_INPUT_SHAPE, dtype=np.float32)
        initializer_names = {item.name for item in model.graph.initializer}
        value_types = {value.name: value.type.tensor_type.elem_type for value in list(model.graph.value_info) + list(model.graph.input)}
        replacements = {}
        changed = False

        def resolve_shape(raw, data_shape):
            resolved, inferred = [], None
            for index, value in enumerate(raw):
                value = int(value)
                if value == 0:
                    if index >= len(data_shape):
                        return None
                    resolved.append(data_shape[index])
                elif value == -1:
                    if inferred is not None:
                        return None
                    inferred = index
                    resolved.append(None)
                else:
                    resolved.append(value)
            if inferred is None:
                return resolved
            known = [value for value in resolved if value is not None]
            if not known or any(value <= 0 for value in known):
                return None
            total, product = int(np.prod(data_shape)), int(np.prod(known))
            if total % product:
                return None
            resolved[inferred] = total // product
            return resolved

        def inserted_position(target, source):
            if len(target) != len(source) + 1:
                return None
            for position in range(1, len(target)):
                if target[position] == 1 and target[:position] == source[:position] and target[position + 1:] == source[position:]:
                    return position
            return None

        def removed_position(target, source):
            if len(target) != len(source) - 1:
                return None
            for position in range(1, len(source)):
                if source[position] == 1 and target[:position] == source[:position] and target[position:] == source[position + 1:]:
                    return position
            return None

        for node in list(model.graph.node):
            if node.op_type != "Reshape" or len(node.input) != 2 or node.input[1] in initializer_names:
                continue
            temporary = copy.deepcopy(model)
            del temporary.graph.value_info[:]
            del temporary.graph.output[:]
            for name in (node.input[0], node.input[1]):
                temporary.graph.output.append(helper.make_tensor_value_info(name, value_types.get(name, TensorProto.FLOAT), None))
            session = ort.InferenceSession(temporary.SerializeToString(), providers=["CPUExecutionProvider"])
            data_value, shape_value = session.run([node.input[0], node.input[1]], {INPUT_NAME: representative})
            data_shape = list(data_value.shape)
            shape = resolve_shape(shape_value.ravel(), data_shape)
            if shape is None:
                raise RuntimeError(f"FixPaddleShapeReshapesTransform: cannot resolve {node.name}")
            if shape == data_shape:
                replacements[node.output[0]] = (node.input[0], None)
            else:
                position = inserted_position(shape, data_shape)
                op_type = "Unsqueeze" if position is not None else None
                if position is None:
                    position = removed_position(shape, data_shape)
                    op_type = "Squeeze" if position is not None else None
                if op_type is None:
                    raise RuntimeError(f"FixPaddleShapeReshapesTransform: unhandled shape on {node.name}: data={data_shape}, target={shape}")
                replacements[node.output[0]] = (node.output[0], helper.make_node(op_type, [node.input[0]], [node.output[0]], axes=[position], name=f"fix_{node.name}_{op_type.lower()}"))
            changed = True

        for output_name, (producer_name, replacement) in replacements.items():
            nodes = list(model.graph.node)
            index = next(i for i, node in enumerate(nodes) if output_name in node.output)
            for node in model.graph.node:
                for input_index, value in enumerate(node.input):
                    if value == output_name:
                        node.input[input_index] = producer_name
            if replacement is None:
                del model.graph.node[index]
            else:
                model.graph.node[index].CopyFrom(replacement)

        if not changed:
            return model, False
        removed = True
        while removed:
            removed = False
            consumed = {value for node in model.graph.node for value in node.input}
            consumed.update(value.name for value in model.graph.output)
            for node in list(model.graph.node):
                if not any(output in consumed for output in node.output):
                    model.graph.node.remove(node)
                    removed = True
        return model, True


class Eliminate5DQKVTransform(BaseOnnxTransform):
    """Rewrite attention QKV layout chains from 5D to equivalent 4D ops."""

    @classmethod
    def apply(cls, model: Any):
        from onnx import helper, numpy_helper

        initializers = {item.name: item for item in model.graph.initializer}
        consumers = {}
        for node in model.graph.node:
            for value in node.input:
                consumers.setdefault(value, []).append(node)
        changed = False
        for reshape in list(model.graph.node):
            if reshape.op_type != "Reshape" or len(reshape.input) != 2 or reshape.input[1] not in initializers:
                continue
            shape = numpy_helper.to_array(initializers[reshape.input[1]])
            if shape.size != 5 or shape[0] != 0 or shape[1] != -1:
                continue
            qkv, heads, head_dim = int(shape[2]), int(shape[3]), int(shape[4])
            transpose = next((node for node in model.graph.node if reshape.output[0] in node.input and node.op_type == "Transpose" and list(next((attr.ints for attr in node.attribute if attr.name == "perm"), [])) == [2, 0, 3, 1, 4]), None)
            if transpose is None:
                continue
            slices = []
            for node in model.graph.node:
                if node.op_type != "Slice" or transpose.output[0] not in node.input or len(node.input) < 4:
                    continue
                if node.input[1] not in initializers or node.input[2] not in initializers or node.input[3] not in initializers:
                    continue
                axes = numpy_helper.to_array(initializers[node.input[3]]).tolist()
                starts = numpy_helper.to_array(initializers[node.input[1]]).tolist()
                ends = numpy_helper.to_array(initializers[node.input[2]]).tolist()
                if axes == [0] and len(starts) == len(ends) == 1 and int(ends[0]) - int(starts[0]) == 1:
                    slices.append((int(starts[0]), node))
            if len(slices) != qkv:
                continue
            producers = {}
            for slot, node in sorted(slices):
                use = consumers.get(node.output[0], [])
                if len(use) != 1 or use[0].op_type != "Squeeze":
                    break
                axes = list(next((attr.ints for attr in use[0].attribute if attr.name == "axes"), []))
                if axes != [0]:
                    break
                producers[slot] = use[0]
            if len(producers) != qkv:
                continue
            base, replacement, chunk = f"qkv4d_{reshape.name}", [], heads * head_dim
            for slot, squeeze in sorted(producers.items()):
                suffix = str(slot)
                tensors = [
                    numpy_helper.from_array(np.array([slot * chunk], dtype=np.int64), f"{base}_start_{suffix}"),
                    numpy_helper.from_array(np.array([(slot + 1) * chunk], dtype=np.int64), f"{base}_end_{suffix}"),
                    numpy_helper.from_array(np.array([2], dtype=np.int64), f"{base}_axis_{suffix}"),
                    numpy_helper.from_array(np.array([1], dtype=np.int64), f"{base}_step_{suffix}"),
                    numpy_helper.from_array(np.array([0, -1, heads, head_dim], dtype=np.int64), f"{base}_shape_{suffix}"),
                ]
                for tensor in tensors:
                    initializers[tensor.name] = tensor
                sliced, reshaped = f"{base}_slice_{suffix}", f"{base}_reshape_{suffix}"
                replacement.extend([
                    helper.make_node("Slice", [reshape.input[0], tensors[0].name, tensors[1].name, tensors[2].name, tensors[3].name], [sliced], name=f"{base}_slice_node_{suffix}"),
                    helper.make_node("Reshape", [sliced, tensors[4].name], [reshaped], name=f"{base}_reshape_node_{suffix}"),
                    helper.make_node("Transpose", [reshaped], [squeeze.output[0]], perm=[0, 2, 1, 3], name=f"{base}_transpose_{suffix}"),
                ])
            index = next(i for i, node in enumerate(model.graph.node) if node.name == reshape.name)
            old = {reshape.name, transpose.name}
            old.update(node.name for _, node in slices)
            old.update(node.name for node in producers.values())
            model.graph.node.insert(index, replacement[0])
            for offset, node in enumerate(replacement[1:], 1):
                model.graph.node.insert(index + offset, node)
            kept = [node for node in model.graph.node if node.name not in old]
            del model.graph.node[:]
            model.graph.node.extend(kept)
            changed = True
        if changed:
            used = {value for node in model.graph.node for value in node.input}
            del model.graph.initializer[:]
            model.graph.initializer.extend(item for item in initializers.values() if item.name in used)
        return model, changed


class CtcGreedySummaryTransform(BaseOnnxTransform):
    """Replace the recognizer probability output with greedy CTC summaries."""

    @classmethod
    def apply(cls, model: Any):
        from onnx import TensorProto, helper

        outputs = model.graph.output
        if [output.name for output in outputs] != [OUTPUT_NAME]:
            raise ValueError(f"CtcGreedySummaryTransform expects exactly [{OUTPUT_NAME}], got {[o.name for o in outputs]}")
        dims = outputs[0].type.tensor_type.shape.dim
        if len(dims) != 3:
            raise ValueError(f"rec output must be rank 3 (B,T,C), got rank {len(dims)}")
        if dims[-1].HasField("dim_value") and dims[-1].dim_value != REC_NUM_CLASSES:
            raise ValueError(f"rec output class dim must be {REC_NUM_CLASSES}, got {dims[-1].dim_value}")

        def summary_output(name: str, elem_type: int):
            info = helper.make_tensor_value_info(name, elem_type, None)
            for dim in dims[:2]:
                info.type.tensor_type.shape.dim.add().CopyFrom(dim)
            return info

        model.graph.node.extend([
            helper.make_node("ArgMax", [OUTPUT_NAME], ["_ctc_argmax_i64"], axis=2, keepdims=0, name="ctc_argmax"),
            helper.make_node("Cast", ["_ctc_argmax_i64"], [REC_TOKEN_IDS_NAME], to=TensorProto.INT32, name="ctc_token_ids_cast"),
            helper.make_node("ReduceMax", [OUTPUT_NAME], [REC_MAX_PROBS_NAME], axes=[2], keepdims=0, name="ctc_max_probs"),
        ])
        del model.graph.output[:]
        model.graph.output.append(summary_output(REC_TOKEN_IDS_NAME, TensorProto.INT32))
        model.graph.output.append(summary_output(REC_MAX_PROBS_NAME, TensorProto.FLOAT))
        return model, True


class StripStaleValueInfoTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: Any):
        live = {value for node in model.graph.node for value in node.output}
        live.update(value.name for value in model.graph.input)
        live.update(value.name for value in model.graph.output)
        keep = [value for value in model.graph.value_info if value.name in live]
        changed = len(keep) != len(model.graph.value_info)
        del model.graph.value_info[:]
        model.graph.value_info.extend(keep)
        return model, changed


class _LayoutFloat64ToFloat32Transform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model):
        import onnx
        from onnx import numpy_helper

        changed = 0
        for tensor in model.graph.initializer:
            array = numpy_helper.to_array(tensor)
            if array.dtype == np.float64:
                tensor.CopyFrom(numpy_helper.from_array(array.astype(np.float32), tensor.name))
                changed += 1
        for node in model.graph.node:
            if node.op_type == "Cast":
                for attribute in node.attribute:
                    if attribute.name == "to" and attribute.i == onnx.TensorProto.DOUBLE:
                        attribute.i = onnx.TensorProto.FLOAT
                        changed += 1
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == onnx.TensorProto.DOUBLE:
                    array = numpy_helper.to_array(attribute.t)
                    attribute.t.CopyFrom(numpy_helper.from_array(array.astype(np.float32), attribute.t.name))
                    changed += 1
        for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
            if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
                value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                changed += 1
        return model, changed


class _LayoutFixDanglingClipBoundsTransform(BaseOnnxTransform):
    """Repair duplicated torch-exported Clip nodes with undefined bounds."""

    @classmethod
    def apply(cls, model):
        initializers, nodes = {item.name for item in model.graph.initializer}, list(model.graph.node)

        def defined(name):
            return not name or name in initializers or any(name in node.output for node in nodes)

        bounds = {}
        for node in nodes:
            if node.op_type != "Clip" or any(not defined(value) for value in node.input[1:]):
                continue
            bounds.setdefault(node.input[0], (node.input[1] if len(node.input) > 1 and node.input[1] in initializers else None, node.input[2] if len(node.input) > 2 and node.input[2] in initializers else None))
        changed = 0
        for node in nodes:
            if node.op_type != "Clip" or node.input[0] not in bounds:
                continue
            for index, value in ((1, bounds[node.input[0]][0]), (2, bounds[node.input[0]][1])):
                if index < len(node.input) and not defined(node.input[index]) and value:
                    node.input[index] = value
                    changed += 1
        return model, changed


class _LayoutCapSliceSentinelStartsTransform(BaseOnnxTransform):
    """Replace INT64_MIN Slice starts with zero before QAIC compilation."""

    @classmethod
    def apply(cls, model):
        from onnx import numpy_helper

        initializers = {item.name: item for item in model.graph.initializer}
        changed = 0
        for node in model.graph.node:
            if node.op_type != "Slice" or len(node.input) < 2 or node.input[1] not in initializers:
                continue
            tensor = initializers[node.input[1]]
            values = numpy_helper.to_array(tensor).copy()
            if (values <= -2**62).any():
                tensor.CopyFrom(numpy_helper.from_array(np.where(values <= -2**62, 0, values), tensor.name))
                changed += 1
        return model, changed


class _LayoutFP16RangeClipTransform(BaseOnnxTransform):
    """Clip only FP16 overflows; preserve legitimate encoder denormals."""

    @classmethod
    def apply(cls, model):
        import onnx
        from onnx import numpy_helper

        limit = np.float32(np.finfo(np.float16).max)
        changed = 0
        for tensor in model.graph.initializer:
            if tensor.data_type != onnx.TensorProto.FLOAT:
                continue
            array = numpy_helper.to_array(tensor)
            if ((array > limit) | (array < -limit)).any():
                tensor.CopyFrom(numpy_helper.from_array(np.clip(array, -limit, limit), tensor.name))
                changed += 1
        return model, changed
