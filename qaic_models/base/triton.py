####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import re
import shutil
from numbers import Integral
from pathlib import Path
from typing import Optional


ONNX_TO_TRITON_DTYPE = {
    "tensor(float)": "TYPE_FP32",
    "tensor(float16)": "TYPE_FP16",
    "tensor(double)": "TYPE_FP64",
    "tensor(int8)": "TYPE_INT8",
    "tensor(int16)": "TYPE_INT16",
    "tensor(int32)": "TYPE_INT32",
    "tensor(int64)": "TYPE_INT64",
    "tensor(uint8)": "TYPE_UINT8",
    "tensor(uint16)": "TYPE_UINT16",
    "tensor(uint32)": "TYPE_UINT32",
    "tensor(uint64)": "TYPE_UINT64",
    "tensor(bool)": "TYPE_BOOL",
}

_DTYPE_NAME_TO_TRITON = {
    "bool": "TYPE_BOOL",
    "float16": "TYPE_FP16",
    "float32": "TYPE_FP32",
    "float64": "TYPE_FP64",
    "double": "TYPE_FP64",
    "uint8": "TYPE_UINT8",
    "uint16": "TYPE_UINT16",
    "uint32": "TYPE_UINT32",
    "uint64": "TYPE_UINT64",
    "int8": "TYPE_INT8",
    "int16": "TYPE_INT16",
    "int32": "TYPE_INT32",
    "int64": "TYPE_INT64",
}


def _load_triton_protobuf():
    try:
        from google.protobuf import text_format  # noqa: PLC0415
        from tritonclient.grpc import model_config_pb2  # noqa: PLC0415
    except Exception as exc:
        raise ImportError(
            "Triton model-repository generation requires the optional Triton extra. "
            "Install with `pip install 'qaic_models[triton]'` or `pip install 'tritonclient[all]'`."
        ) from exc
    return model_config_pb2, text_format


def _shape_to_triton_dims(shape, max_batch_size: int, *, strip_batch_dim: bool = False) -> list[int]:
    dims = [int(d) if isinstance(d, Integral) else -1 for d in shape]
    if strip_batch_dim and max_batch_size > 0 and dims:
        return dims[1:]
    return dims


def _triton_dims_for_shape(shape, max_batch_size: int) -> list[int]:
    return _shape_to_triton_dims(shape, max_batch_size, strip_batch_dim=max_batch_size > 0)


def _dims_compatible(producer_shape: list[int], consumer_shape: list[int]) -> bool:
    return len(producer_shape) == len(consumer_shape) and all(
        producer_dim == consumer_dim or producer_dim == -1 or consumer_dim == -1
        for producer_dim, consumer_dim in zip(producer_shape, consumer_shape)
    )


def _onnx_dtype_to_triton(onnx_type: str) -> str:
    if onnx_type not in ONNX_TO_TRITON_DTYPE:
        raise ValueError(f"Unsupported ONNX tensor type for Triton config: {onnx_type}")
    return ONNX_TO_TRITON_DTYPE[onnx_type]


def _dtype_to_triton(dtype: object) -> str:
    dtype_text = str(dtype).lower()
    for name, triton_type in _DTYPE_NAME_TO_TRITON.items():
        if re.search(rf"(^|[^a-z0-9]){re.escape(name)}($|[^a-z0-9])", dtype_text):
            return triton_type
    raise ValueError(f"Unsupported tensor dtype for Triton config: {dtype}")


def _dtype_enum(model_config_pb2, dtype: str) -> int:
    try:
        return getattr(model_config_pb2, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported Triton datatype: {dtype}") from exc


def _set_tensor_specs(model_config_pb2, repeated_field, tensors: dict[str, dict[str, object]]) -> None:
    for name, info in tensors.items():
        tensor = repeated_field.add()
        tensor.name = name
        tensor.data_type = _dtype_enum(model_config_pb2, str(info["dtype"]))
        tensor.dims.extend(int(dim) for dim in info["shape"])


def _same_file(src: Path, dst: Path) -> bool:
    return dst.exists() and dst.samefile(src)


class TritonModelConfig:
    def __init__(
        self,
        name: str,
        backend: str,
        model_path: Optional[Path] = None,
        model_source: Optional[str] = None,
        max_batch_size: int = 0,
        count: int = 1,
        model_version: int = 1,
        inputs: Optional[list[dict]] = None,
        outputs: Optional[list[dict]] = None,
        parameters: Optional[dict] = None,
        dynamic_batching: Optional[dict] = None,
        device_id: int | None = None,
    ):
        self.model_path = Path(model_path) if model_path is not None else None
        self.model_source = model_source
        self.name = name
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.count = count
        self.model_version = model_version
        self.device_id = device_id
        self.parameters = dict(parameters) if parameters is not None else None
        if backend == "qaic" and device_id is not None:
            if self.parameters is None:
                self.parameters = {}
            self.parameters.setdefault("device_id", str(device_id))
        # Optional dynamic-batching block. Keys: max_queue_delay_microseconds
        # (int) and preferred_batch_size (list[int]). Emitted only when set.
        self.dynamic_batching = dynamic_batching
        self.input_tensors: dict[str, dict[str, object]] = {}
        self.output_tensors: dict[str, dict[str, object]] = {}
        if backend == "onnxruntime":
            self._require_model_path()
            if inputs is not None or outputs is not None:
                self._init_from_explicit(inputs, outputs)
            else:
                self._init_from_onnx(self.model_path)
        elif backend == "qaic":
            self._require_model_path()
            if inputs is not None or outputs is not None:
                self._init_from_explicit(inputs, outputs)
            else:
                self._init_from_qaic(self.model_path)
        elif backend == "python":
            if (self.model_path is None) == (self.model_source is None):
                raise ValueError(
                    f"Python backend model '{self.name}' requires exactly one of model_path or model_source"
                )
            self._init_from_explicit(inputs, outputs)
        else:
            raise ValueError(f"Unsupported Triton backend: {self.backend}")

    def _require_model_path(self) -> None:
        if self.model_path is None:
            raise ValueError(f"{self.backend} backend model '{self.name}' requires model_path")

    def _init_from_onnx(self, model_path: Path) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        for inp in sess.get_inputs():
            shape = _triton_dims_for_shape(inp.shape, self.max_batch_size)
            self.input_tensors[inp.name] = {"shape": shape, "dtype": _onnx_dtype_to_triton(inp.type)}
        for out in sess.get_outputs():
            shape = _triton_dims_for_shape(out.shape, self.max_batch_size)
            self.output_tensors[out.name] = {"shape": shape, "dtype": _onnx_dtype_to_triton(out.type)}

    def _init_from_qaic(self, model_path: Path) -> None:
        try:
            import qaic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("qaic module not available; install the QAIC SDK to inspect QPC metadata") from exc

        session_kwargs = {} if self.device_id is None else {"dev_id": self.device_id}
        sess = qaic.Session(str(model_path), **session_kwargs)
        input_shapes = getattr(sess, "model_input_shape_dict", None)
        output_shapes = getattr(sess, "model_output_shape_dict", None)
        if not input_shapes or not output_shapes:
            raise ValueError(f"QAIC model '{model_path}' did not expose input/output tensor metadata")
        for name, (shape, dtype) in input_shapes.items():
            self.input_tensors[name] = {
                "shape": _triton_dims_for_shape(shape, self.max_batch_size),
                "dtype": _dtype_to_triton(dtype),
            }
        for name, (shape, dtype) in output_shapes.items():
            self.output_tensors[name] = {
                "shape": _triton_dims_for_shape(shape, self.max_batch_size),
                "dtype": _dtype_to_triton(dtype),
            }

    def _init_from_explicit(self, inputs: Optional[list[dict]], outputs: Optional[list[dict]]) -> None:
        if inputs is None or outputs is None:
            raise ValueError(f"{self.backend} backend model '{self.name}' requires input/output metadata")
        for spec in inputs:
            self.input_tensors[spec["name"]] = {
                "shape": _triton_dims_for_shape(spec["dims"], self.max_batch_size),
                "dtype": spec["data_type"],
            }
        for spec in outputs:
            self.output_tensors[spec["name"]] = {
                "shape": _triton_dims_for_shape(spec["dims"], self.max_batch_size),
                "dtype": spec["data_type"],
            }

    def _to_proto(self, triton_protobuf=None):
        model_config_pb2, _ = triton_protobuf or _load_triton_protobuf()
        config = model_config_pb2.ModelConfig()
        config.name = self.name
        config.backend = self.backend
        config.max_batch_size = self.max_batch_size
        _set_tensor_specs(model_config_pb2, config.input, self.input_tensors)
        _set_tensor_specs(model_config_pb2, config.output, self.output_tensors)
        if self.parameters:
            for name, value in self.parameters.items():
                config.parameters[str(name)].string_value = str(value)
        if self.dynamic_batching:
            dynamic_batching = config.dynamic_batching
            for size in self.dynamic_batching.get("preferred_batch_size", []):
                dynamic_batching.preferred_batch_size.append(int(size))
            if "max_queue_delay_microseconds" in self.dynamic_batching:
                dynamic_batching.max_queue_delay_microseconds = int(self.dynamic_batching["max_queue_delay_microseconds"])
        instance_group = config.instance_group.add()
        instance_group.kind = model_config_pb2.ModelInstanceGroup.KIND_CPU
        instance_group.count = self.count
        return config

    def _artifact_path(self, model_root: Path) -> Path:
        output_file = model_root / str(self.model_version)
        if self.backend == "onnxruntime":
            return output_file / "model.onnx"
        if self.backend == "qaic":
            return output_file / "programqpc.bin"
        if self.backend == "python":
            return output_file / "model.py"
        raise ValueError(f"Unsupported Triton backend: {self.backend}")

    def tofile(self, model_root: Path, triton_protobuf=None):
        _, text_format = triton_protobuf or _load_triton_protobuf()
        output_file = model_root / "config.pbtxt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(text_format.MessageToString(self._to_proto(triton_protobuf)))

        artifact_path = self._artifact_path(model_root)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_source is not None:
            artifact_path.write_text(self.model_source)
        elif not _same_file(self.model_path, artifact_path):
            shutil.copyfile(self.model_path, artifact_path)


class TritonModelEnsemble:
    def __init__(self, name: str, models: list[TritonModelConfig], max_batch_size: int = 0, model_version: int = 1):
        if not models:
            raise ValueError(f"Triton ensemble '{name}' requires at least one model")
        self.name = name
        self.models = models
        self.max_batch_size = max_batch_size
        self.model_version = model_version

    def _to_proto(self, triton_protobuf=None):
        model_config_pb2, _ = triton_protobuf or _load_triton_protobuf()
        config = model_config_pb2.ModelConfig()
        config.name = self.name
        config.platform = "ensemble"
        config.max_batch_size = self.max_batch_size
        first = self.models[0]
        last = self.models[-1]
        _set_tensor_specs(model_config_pb2, config.input, first.input_tensors)
        _set_tensor_specs(model_config_pb2, config.output, last.output_tensors)

        available_tensors = {
            name: {"name": name, "shape": info["shape"], "dtype": info["dtype"]}
            for name, info in first.input_tensors.items()
        }
        for idx, model in enumerate(self.models):
            step = config.ensemble_scheduling.step.add()
            step.model_name = model.name
            step.model_version = model.model_version
            for in_name, in_info in model.input_tensors.items():
                if in_name not in available_tensors:
                    raise ValueError(f"No available tensor found for input '{in_name}' in model '{model.name}'")
                tensor_info = available_tensors[in_name]
                if not _dims_compatible(tensor_info["shape"], in_info["shape"]) or tensor_info["dtype"] != in_info["dtype"]:
                    raise ValueError(f"Tensor '{in_name}' is incompatible with input '{in_name}' in model '{model.name}'")
                step.input_map[in_name] = tensor_info["name"]
            is_last = idx == len(self.models) - 1
            for out_name, out_info in model.output_tensors.items():
                dest_name = out_name if is_last else f"model{idx}_{out_name}"
                step.output_map[out_name] = dest_name
                available_tensors[out_name] = {"name": dest_name, "shape": out_info["shape"], "dtype": out_info["dtype"]}
        return config

    def tofile(self, model_root: Path, triton_protobuf=None) -> None:
        _, text_format = triton_protobuf or _load_triton_protobuf()
        output_file = model_root / "config.pbtxt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(text_format.MessageToString(self._to_proto(triton_protobuf)))
        (model_root / str(self.model_version)).mkdir(parents=True, exist_ok=True)
