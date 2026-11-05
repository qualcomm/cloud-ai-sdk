# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Triton python-backend model serving the PP-OCRv6 rec stage with dynamic batching.

A QPC is compiled for a fixed batch size (2 for the rec throughput build), so
Triton cannot coalesce requests onto it directly. This model wraps the QPC:
Triton's dynamic batching hands ``execute`` up to ``max_batch_size``
requests, the model stacks them, fills the empty slots up to the QPC's fixed
batch by cycling the real inputs, runs one QPC inference, drops the filler
rows, and CTC-decodes each real line.

The empty slots are filled with *real* inputs, not zeros: the compiled QPC
has cross-batch interaction, so zero-padding shifts the real rows' logits and
corrupts their decode. Cycling the real inputs leaves them unaffected, and
the filler rows are dropped before decoding.

Each line arrives prepared as an FP32 ``[3, 48, 1536]`` tensor (``[1, 3, 48,
1536]`` per request) — the ``preprocess_rec_image`` output minus the
singleton batch axis — because Triton's dynamic batching can only stack
identical shapes, and raw line crops vary in width; the client does the rec
preprocessing before submitting each line to this model.

The QPC's fixed batch is discovered from the QPC's own metadata at
initialization, so the same model source serves either rec build. Each
Triton instance owns its own ``qaic.Session`` (one 4-core rec QPC session),
so the model's 4 instances give 4 concurrent rec sessions — 16 NSP, one full
device. The generated repository pins all four sessions to QID 3 so the
detector and layout stages can reserve their own SoCs. An explicit
``device_id`` parameter can pin another deployment. Each session is activated
during model initialization. The rec QPC binary and ``char_dict.txt`` must
sit next to this file; ``export.py`` copies them there when it generates the
model repository.
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
from qaic_models.vision.ocr.paddle_ocr.model import CHAR_DICT_NAME, REC_THROUGHPUT_QPC_SUBDIR
from qaic_models.vision.ocr.paddle_ocr.postprocess import decode_rec_batch, load_char_dict
from qaic_models.vision.ocr.paddle_ocr.preprocess import INPUT_NAME


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
        import qaic  # noqa: PLC0415  # QAIC SDK is only available in Cloud AI runtime containers.
        import triton_python_backend_utils as pb_utils  # noqa: PLC0415

        self.pb_utils = pb_utils
        model_dir = Path(__file__).resolve().parent
        qpc_path = model_dir / REC_THROUGHPUT_QPC_SUBDIR / "programqpc.bin"
        device_id = _param(args["model_config"], "device_id", "")
        session_kwargs = {"num_activations": 1, "set_size": 1}
        if device_id:
            session_kwargs["dev_id"] = int(device_id)
        self.session = qaic.Session(model_path=str(qpc_path), **session_kwargs)
        # Activate during model load so an unavailable device or exhausted
        # NSP allocation is reported by Triton before serving requests.
        self.session.setup()
        # The QPC's fixed batch is its input's leading dim (2).
        input_shapes = getattr(self.session, "model_input_shape_dict", None)
        if not input_shapes:
            raise RuntimeError(f"QAIC model '{qpc_path}' did not expose input tensor metadata")
        (qpc_shape, _dtype) = next(iter(input_shapes.values()))
        self.qpc_batch = int(qpc_shape[0])
        self.char_dict = load_char_dict(model_dir / CHAR_DICT_NAME)

    def execute(self, requests):
        # Triton's python backend hands each request a tensor that still carries
        # its own (size-1) batch dim -- e.g. [1, 3, 48, 1536] for a single line --
        # so concatenate along axis 0 to stack the lines into one [n, 3, 48, 1536]
        # batch for the QPC. (np.stack would add a 5th axis instead.)
        items = [self.pb_utils.get_input_tensor_by_name(request, INPUT_NAME).as_numpy() for request in requests]
        n = len(items)
        if n > self.qpc_batch:
            raise ValueError(
                f"received {n} requests but the QPC is compiled for batch {self.qpc_batch}; "
                "set max_batch_size in config.pbtxt to the QPC's batch"
            )
        batch = np.concatenate(items, axis=0)
        if n < self.qpc_batch:
            # Fill the empty slots by cycling the real inputs. The QPC has
            # cross-batch interaction, so zero-padding would shift the real
            # rows' logits; real filler rows leave them untouched and are
            # dropped before decoding.
            filler = np.concatenate([batch[i % n : i % n + 1] for i in range(self.qpc_batch - n)], axis=0)
            batch = np.concatenate([batch, filler], axis=0)
        outputs = self.session.run({INPUT_NAME: np.ascontiguousarray(batch)})
        decodes = decode_rec_batch(outputs, self.char_dict)[:n]  # drop the filler rows
        responses = []
        for pred in decodes:
            # text is a single STRING [1] tensor holding the decoded line;
            # confidence is a scalar [1] tensor.
            responses.append(
                self.pb_utils.InferenceResponse(
                    output_tensors=[
                        self.pb_utils.Tensor("text", np.array([pred["text"]], dtype=object)),
                        self.pb_utils.Tensor("confidence", np.asarray([pred["confidence"]], dtype=np.float32)),
                    ]
                )
            )
        return responses
