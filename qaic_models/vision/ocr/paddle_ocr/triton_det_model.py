# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Triton python-backend model serving the PP-OCRv6 det stage on Cloud AI 100.

Each request carries one image as a ``UINT8 [H, W, 3]`` BGR tensor (the raw
BGR pixel array of a layout region crop from ``paddle_ocr_layout``; dims are
dynamic per request). The model owns one 16-core det QPC session and runs
one det pass (letterbox included), returning the detected line boxes as a
JSON string (``{"boxes": [[x0, y0, x1, y1, score], ...]}``) plus a box
count. Boxes are in the region image's coordinates in detector score order;
the client offsets them by the region origin, sorts them into reading order,
and crops the lines itself.

Recognition is a separate step, orchestrated by the client: each prepared
line (a fixed 48x1536 float tensor) is sent as its own request to the
``paddle_ocr_rec`` model, issued concurrently so that model's dynamic
batching pairs the lines into batch-2 QPC passes across its instances;
the client then assembles the result.

Each Triton detector instance owns one 16-core det QPC session. The generated
repository has two instances and places them with
``model_instance_to_card.json``: instance 0 uses QID 1 and instance 1 uses
QID 2. The det QPC binary and mapping file sit next to this file under
``qpc_det_latency/`` and are copied by ``export.py``. A scalar ``device_id``
config parameter remains as a fallback for hand-built repositories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Absolute (not relative) imports: Triton's python backend loads model.py as a
# top-level module named after the model version (e.g. "1"), with no package
# context, so relative imports fail at load time. The package is importable in
# the serving container (installed via pip3 install "/cloud-ai-sdk[...]"), so
# the absolute form works both here and standalone in the model repository.
from qaic_models.vision.ocr.paddle_ocr.model import PaddleOCRDetQaic


INSTANCE_DEVICE_MAP_NAME = "model_instance_to_card.json"


def _param(model_config, name: str, default: str) -> str:
    if isinstance(model_config, str):
        model_config = json.loads(model_config)
    params = (
        model_config.get("parameters")
        if isinstance(model_config, dict)
        else getattr(model_config, "parameters", None)
    )
    if params is not None and name in params:
        value = params[name]
        if isinstance(value, dict):
            return value.get("string_value") or default
        return value.string_value or default
    return default


def _resolve_device_id(model_dir: Path, model_instance_name: str, fallback: int) -> int:
    """Resolve one QID from the generated Triton instance mapping."""
    mapping_path = model_dir / INSTANCE_DEVICE_MAP_NAME
    if not mapping_path.is_file():
        return fallback

    instance_id = model_instance_name.rsplit("_", 1)[-1]
    mapping = json.loads(mapping_path.read_text())
    qids = mapping.get(instance_id)
    if not isinstance(qids, list) or len(qids) != 1:
        raise ValueError(
            f"{INSTANCE_DEVICE_MAP_NAME} entry for instance {instance_id!r} must contain exactly one QID"
        )
    return int(qids[0])


class TritonPythonModel:
    def initialize(self, args):
        import triton_python_backend_utils as pb_utils  # noqa: PLC0415

        self.pb_utils = pb_utils
        model_config = args["model_config"]
        # The det QPC sits next to this file (see module docstring).
        model_dir = Path(__file__).resolve().parent
        fallback_device_id = int(_param(model_config, "device_id", "1"))
        device_id = _resolve_device_id(model_dir, args.get("model_instance_name", ""), fallback_device_id)
        self.det = PaddleOCRDetQaic(str(model_dir), device_id=device_id)

    def execute(self, requests):
        responses = []
        for request in requests:
            # One document image per request: raw BGR pixels, HxWx3 uint8.
            img_bgr = np.ascontiguousarray(
                self.pb_utils.get_input_tensor_by_name(request, "src").as_numpy(), dtype=np.uint8
            )
            if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
                raise RuntimeError("the 'src' input must be a UINT8 [H, W, 3] BGR document image")
            boxes, scores = self.det.run_image(img_bgr)
            # The box list is variable-length, so it is served as a JSON blob
            # (one [x0, y0, x1, y1, score] per region, detector score order)
            # plus the count — no padding capacity needed for dense documents.
            # Served as a UTF-8 STRING element: TYPE_STRING is the model
            # config's byte/string data type (the legacy TYPE_BYTES name is
            # not a valid enum value in this Triton build).
            payload = json.dumps(
                {"boxes": [[*box, score] for box, score in zip(boxes, scores)]},
                ensure_ascii=False,
            )
            responses.append(
                self.pb_utils.InferenceResponse(
                    output_tensors=[
                        self.pb_utils.Tensor("boxes", np.array([payload], dtype=object)),
                        self.pb_utils.Tensor("box_count", np.asarray([len(boxes)], dtype=np.int32)),
                    ]
                )
            )
        return responses
