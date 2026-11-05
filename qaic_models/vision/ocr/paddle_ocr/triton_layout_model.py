# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Triton python-backend model serving the PP-DocLayoutV3 layout stage on Cloud AI 100.

Each request carries one document image as a ``UINT8 [H, W, 3]`` BGR tensor
(the raw ``cv2.imread`` array; dims are dynamic per request). The model owns
one 16-core layout QPC session and returns the document's layout regions as
a JSON string — ``{"regions": [{"box": [x0, y0, x1, y1], "label": ...,
"class_id": ..., "score": ..., "order": ...}, ...]}`` in original-image
coordinates, in the layout's reading order — plus a region count.

Detection and recognition are separate steps, orchestrated by the client:
each region crop is sent to ``paddle_ocr_det_qpc`` for its text-line boxes
(in region coordinates, which the client offsets by the region origin), and
each prepared line is sent to ``paddle_ocr_rec`` (see :mod:`demo_triton`).

The layout QPC binary must sit next to this file under
``qpc_layout_latency/``; ``export.py`` copies it there when it generates the
model repository. Device placement comes from the config parameter
``device_id`` (default 0).
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
from qaic_models.vision.ocr.paddle_ocr.model import PaddleOCRLayoutQaic


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


class TritonPythonModel:
    def initialize(self, args):
        import triton_python_backend_utils as pb_utils  # noqa: PLC0415

        self.pb_utils = pb_utils
        model_config = args["model_config"]
        # The layout QPC sits next to this file (see module docstring).
        model_dir = str(Path(__file__).resolve().parent)
        self.layout = PaddleOCRLayoutQaic(model_dir, device_id=int(_param(model_config, "device_id", "0")))

    def execute(self, requests):
        responses = []
        for request in requests:
            # One document image per request: raw BGR pixels, HxWx3 uint8.
            img_bgr = np.ascontiguousarray(
                self.pb_utils.get_input_tensor_by_name(request, "src").as_numpy(), dtype=np.uint8
            )
            if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
                raise RuntimeError("the 'src' input must be a UINT8 [H, W, 3] BGR document image")
            layout_pred = self.layout.run_image(img_bgr)
            # The region list is variable-length, so it is served as a JSON
            # blob (one region object per region, layout reading order) plus
            # the count — no padding capacity needed. Served as a UTF-8 STRING
            # element: TYPE_STRING is the model config's byte/string data type
            # (the legacy TYPE_BYTES name is not a valid enum value in this
            # Triton build).
            payload = json.dumps(layout_pred, ensure_ascii=False)
            responses.append(
                self.pb_utils.InferenceResponse(
                    output_tensors=[
                        self.pb_utils.Tensor("regions", np.array([payload], dtype=object)),
                        self.pb_utils.Tensor("region_count", np.asarray([len(layout_pred["regions"])], dtype=np.int32)),
                    ]
                )
            )
        return responses
