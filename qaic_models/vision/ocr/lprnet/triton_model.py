####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
"""Triton python-backend model serving an LPRNet QPC with dynamic batching.

A QPC is compiled for a fixed batch size (1 for the latency build, 8 for the
throughput build), so Triton cannot coalesce requests onto it directly. This
model wraps the QPC: Triton's dynamic batching hands ``execute`` up to
``max_batch_size`` requests, the model stacks them, fills the empty slots up
to the QPC's fixed batch by cycling the real inputs, runs one QPC inference,
drops the filler rows, and CTC-decodes each real plate.

The empty slots are filled with *real* inputs, not zeros: the compiled QPC
has cross-batch interaction (a batch-norm / global op in the graph), so
zero-padding shifts the real rows' logits and corrupts their decode. Cycling
the real inputs leaves them unaffected, and the filler rows are dropped
before decoding.

The QPC's fixed batch is discovered from the QPC's own metadata at
initialization, so the same model source serves either the latency or the
throughput QPC. The QPC binary (``programqpc.bin``) must sit next to this file
(``lprnet_qpc/1/programqpc.bin``); ``export.py`` copies it there when it
generates the model repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Absolute (not relative) imports: Triton's python backend loads model.py as a
# top-level module named after the model version (e.g. "1"), with no package
# context, so relative imports fail at load time. The package is importable in
# the serving container (installed via pip3 install "/cloud-ai-sdk[...]"), so
# the absolute form works both here and standalone in the model repository.
from qaic_models.vision.ocr.lprnet.postprocess import OUTPUT_NAME, decode_ctc
from qaic_models.vision.ocr.lprnet.preprocess import INPUT_NAME

class TritonPythonModel:
    def initialize(self, args):
        import qaic  # noqa: PLC0415  # QAIC SDK is only available in Cloud AI runtime containers.
        import triton_python_backend_utils as pb_utils  # noqa: PLC0415

        self.pb_utils = pb_utils
        qpc_path = Path(__file__).resolve().parent / "programqpc.bin"
        parameters = args.get("model_instance_kind", "")  # retain Triton's standard args access
        model_config = args.get("model_repository", "")
        # The generated config carries an optional device_id parameter.  It
        # is read through the model config JSON when Triton supplies it; the
        # default remains SDK-selected for manual repositories.
        device_id = None
        try:
            import json
            config = json.loads(args.get("model_config", "{}"))
            device_id = int(config.get("parameters", {}).get("device_id", {}).get("string_value"))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            pass
        session_kwargs = {"model_path": str(qpc_path), "num_activations": 1}
        if device_id is not None:
            session_kwargs["dev_id"] = device_id
        self.session = qaic.Session(**session_kwargs)
        # The QPC's fixed batch is its input's leading dim (1 or 8).
        input_shapes = getattr(self.session, "model_input_shape_dict", None)
        if not input_shapes:
            raise RuntimeError(f"QAIC model '{qpc_path}' did not expose input tensor metadata")
        (qpc_shape, _dtype) = next(iter(input_shapes.values()))
        self.qpc_batch = int(qpc_shape[0])

    def execute(self, requests):
        # Triton's python backend hands each request a tensor that still carries
        # its own (size-1) batch dim -- e.g. [1, 3, 24, 94] for a single plate --
        # so concatenate along axis 0 to stack the plates into one [n, 3, 24, 94]
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
        logits = self.session.run({INPUT_NAME: np.ascontiguousarray(batch)})[OUTPUT_NAME]
        logits = logits[:n]  # drop the filler rows
        responses = []
        for i in range(n):
            pred = decode_ctc(logits[i : i + 1])  # decode_ctc expects (1, 68, 18)
            responses.append(
                self.pb_utils.InferenceResponse(
                    output_tensors=[
                        self.pb_utils.Tensor("plate", np.array([pred["plate"]], dtype=object)),
                        self.pb_utils.Tensor("confidence", np.asarray([pred["confidence"]], dtype=np.float32)),
                        self.pb_utils.Tensor("num_chars", np.asarray([len(pred["plate"])], dtype=np.int32)),
                    ]
                )
            )
        return responses
