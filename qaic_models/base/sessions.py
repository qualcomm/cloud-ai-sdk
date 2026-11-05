####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
"""qaic_models.base.sessions — QAICSessionManager.

Optional helper for constructing qaic.Session / onnxruntime.InferenceSession
instances with centralized defaults. Not required by QAICPredictor's base
contract (which constructs self._session directly, per predict_base.py) —
intended for recipes whose _forward() orchestrates multiple named sessions,
so they don't repeat qaic.Session(...) construction by hand.

Supports two lifecycles seen in production onboardings:
  - construct-once, reuse: build in __init__, keep for the predictor's
    lifetime (the common 2-session detect-then-recognize pattern).
  - sequential setup/run/release: build immediately before use, release()
    immediately after, when co-resident QPCs would exceed the device's
    NSP budget (seen in a 3-QPC / 16-NSP pipeline that tears down each
    session before loading the next stage's QPC).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np

try:
    import qaic
except ImportError:  # QAIC SDK is only available in Cloud AI runtime containers.
    qaic = None


class QAICSessionManager:
    """Constructs qaic.Session / onnxruntime.InferenceSession instances with
    centralized defaults so recipes never construct sessions directly."""

    def __init__(self, *, device_id: Optional[int] = None,
                 oversubscription_name: Optional[str] = None):
        self.device_id = device_id
        self.oversubscription_name = oversubscription_name

    def qaic_session(self, programqpc_path: str, *, setup: bool = True, **overrides) -> "qaic.Session":
        """Construct a qaic.Session for programqpc_path.

        Defaults: num_activations=1, set_size=1. dev_id=self.device_id and
        oversubscription_name=self.oversubscription_name are included only
        when set (both None by default). `overrides` win over all defaults.
        Calls .setup() immediately unless setup=False (some pipelines rely
        on lazy setup at first .run()).
        """
        if qaic is None:
            raise ImportError("qaic module not available; install the QAIC SDK to use QAICSessionManager.qaic_session.")

        kwargs = {"num_activations": 1, "set_size": 1}
        if self.device_id is not None:
            kwargs["dev_id"] = self.device_id
        if self.oversubscription_name is not None:
            kwargs["oversubscription_name"] = self.oversubscription_name
        kwargs.update(overrides)

        session = qaic.Session(programqpc_path, **kwargs)
        if setup:
            session.setup()
        return session

    def onnx_session(self, onnx_path: str, providers: Optional[list] = None) -> "onnxruntime.InferenceSession":
        """Construct an onnxruntime.InferenceSession. providers defaults to
        ["CPUExecutionProvider"]."""
        import onnxruntime as ort

        return ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])

    @staticmethod
    def release(session) -> None:
        """Call session.reset() if the session exposes it (qaic.Session does;
        onnxruntime.InferenceSession does not and is left alone). Used for the
        sequential setup->run->release lifecycle; safe no-op otherwise."""
        reset = getattr(session, "reset", None)
        if callable(reset):
            reset()

    @staticmethod
    def contiguous(array: "np.ndarray") -> "np.ndarray":
        """np.ascontiguousarray(array); the exact call currently done ad hoc
        before every session.run() in QAIC-path code."""
        return np.ascontiguousarray(array)


def _load_qaicrt_bindings():
    try:
        import qaicrt
    except ImportError:
        import platform
        import sys

        sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
        import qaicrt

    try:
        import QAicApi_pb2 as aicapi
    except ImportError:
        import sys

        sys.path.append("/opt/qti-aic/dev/python")
        import QAicApi_pb2 as aicapi

    aic_to_np_dtype_mapping = {
        aicapi.FLOAT_TYPE: np.dtype(np.float32),
        aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
        aicapi.INT8_Q_TYPE: np.dtype(np.int8),
        aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
        aicapi.INT16_Q_TYPE: np.dtype(np.int16),
        aicapi.INT32_Q_TYPE: np.dtype(np.int32),
        aicapi.INT32_I_TYPE: np.dtype(np.int32),
        aicapi.INT64_I_TYPE: np.dtype(np.int64),
        aicapi.INT8_TYPE: np.dtype(np.int8),
    }
    return qaicrt, aicapi, aic_to_np_dtype_mapping


class QAICInferenceSession:
    """Low-level QAICRT inference session for MDP and specialization-aware QPCs.

    Use this when `qaic.Session` is too high-level for the recipe: e.g. a
    QPC run with explicit QIDs needs a runtime `devMapping` (MDP maps every
    partition; since the 1.22 SDK a single explicit QID needs one too), or a
    QPC compiled with multiple network specializations needs output buffer
    dimensions selected from the matched allowed-shape entry before execution.
    """

    def __init__(
        self,
        qpc_path: Union[os.PathLike, str],
        device_ids: Optional[List[int]] = None,
        activate: bool = True,
        enable_debug_logs: bool = False,
    ):
        qaicrt, aicapi, dtype_mapping = _load_qaicrt_bindings()
        self.qaicrt = qaicrt
        self.aicapi = aicapi
        self.aic_to_np_dtype_mapping = dtype_mapping

        if device_ids is not None:
            devices = qaicrt.QIDList(device_ids)
            self.context = qaicrt.Context(devices)
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)
        if enable_debug_logs:
            if self.context.setLogLevel(qaicrt.QLogLevel.QL_DEBUG) != qaicrt.QStatus.QS_SUCCESS:
                raise RuntimeError("Failed to setLogLevel")

        qpc = qaicrt.Qpc(str(qpc_path))
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to getIoDescriptor")
        iodesc.ParseFromString(bytes(iodesc_data))
        self.allowed_shapes = [
            [(dtype_mapping[x.type].itemsize, list(x.dims)) for x in allowed_shape.shapes]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map = {binding.name: binding.index for binding in self.bindings}

        prog_properties = qaicrt.QAicProgramProperties()
        # Since the 1.22 SDK a devMapping is required even when a single
        # explicit QID is requested; same colon-joined form as MDP device
        # maps.
        if device_ids:
            prog_properties.devMapping = ":".join(map(str, device_ids))
        self.program = qaicrt.Program(self.context, None, qpc, prog_properties)
        if self.program.load() != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to load program")

        self.is_active = False
        if activate:
            self.activate()
            self.is_active = True

        self.qbuffers = [qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings]
        self.buf_dims = qaicrt.BufferDimensionsVecRef(
            [(dtype_mapping[binding.type].itemsize, list(binding.dims)) for binding in self.bindings]
        )

    @property
    def input_names(self) -> List[str]:
        return [binding.name for binding in self.bindings if binding.dir == self.aicapi.BUFFER_IO_TYPE_INPUT]

    @property
    def output_names(self) -> List[str]:
        return [binding.name for binding in self.bindings if binding.dir == self.aicapi.BUFFER_IO_TYPE_OUTPUT]

    def activate(self):
        if not self.is_active:
            self.program.activate()
            self.execObj = self.qaicrt.ExecObj(self.context, self.program)
            self.is_active = True

    def deactivate(self):
        if self.is_active:
            del self.execObj
            self.program.deactivate()
            self.is_active = False

    def set_buffers(self, buffers: Dict[str, np.ndarray]):
        for buffer_name, buffer in buffers.items():
            if buffer_name not in self.binding_index_map:
                warn(f'Buffer: "{buffer_name}" not found')
                continue
            buffer_index = self.binding_index_map[buffer_name]
            self.qbuffers[buffer_index] = self.qaicrt.QBuffer(buffer.tobytes())
            self.buf_dims[buffer_index] = (
                buffer.itemsize,
                buffer.shape if len(buffer.shape) > 0 else (1,),
            )

    def _apply_output_shapes_for_matched_specialization(self):
        if not self.allowed_shapes:
            return
        for allowed_shape in self.allowed_shapes:
            matched = True
            for binding, (_, shape) in zip(self.bindings, allowed_shape):
                if binding.dir != self.aicapi.BUFFER_IO_TYPE_INPUT:
                    continue
                buffer_index = self.binding_index_map[binding.name]
                _, passed_shape = self.buf_dims[buffer_index]
                if list(passed_shape) == [0]:
                    continue
                if list(passed_shape) != list(shape):
                    matched = False
                    break
            if not matched:
                continue
            for binding, (elemsize, shape) in zip(self.bindings, allowed_shape):
                if binding.dir != self.aicapi.BUFFER_IO_TYPE_OUTPUT:
                    continue
                buffer_index = self.binding_index_map[binding.name]
                self.buf_dims[buffer_index] = (elemsize, shape)
            return

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.set_buffers(inputs)
        self._apply_output_shapes_for_matched_specialization()
        if self.execObj.setData(self.qbuffers, self.buf_dims) != self.qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to setData")
        if self.queue.enqueue(self.execObj) != self.qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        if self.execObj.waitForCompletion() != self.qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"
            if self.allowed_shapes:
                error_message += '\n\n(Only if "No matching dimension found" error is present above)\nAllowed shapes:'
                for i, allowed_shape in enumerate(self.allowed_shapes):
                    error_message += f"\n{i}\n"
                    for binding, (elemsize, shape), (_, passed_shape) in zip(
                        self.bindings, allowed_shape, self.buf_dims
                    ):
                        if passed_shape == [0]:
                            if not binding.is_partial_buf_allowed:
                                warn(f"Partial buffer not allowed for: {binding.name}")
                            continue
                        error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
                error_message += "\n\nPassed shapes:\n"
                for binding, (elemsize, shape) in zip(self.bindings, self.buf_dims):
                    if shape == [0]:
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            raise ValueError(error_message)

        status, output_qbuffers = self.execObj.getData()
        if status != self.qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to getData")
        outputs = {}
        for output_name in self.output_names:
            buffer_index = self.binding_index_map[output_name]
            if self.qbuffers[buffer_index].size == 0:
                continue
            output_shape = self.buf_dims[buffer_index][1]
            output_size = int(np.prod(output_shape))
            output = np.frombuffer(
                bytes(output_qbuffers[buffer_index]),
                self.aic_to_np_dtype_mapping[self.bindings[buffer_index].type],
            )
            if output.size < output_size:
                raise ValueError(f'Output buffer "{output_name}" has {output.size} elements, expected {output_size}')
            outputs[output_name] = output[:output_size].reshape(output_shape)
        return outputs

