####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
"""Triton python backend model for the F5-TTS recipe.

Serves the full F5-TTS pipeline (CPU preprocess -> QAIC transformer diffusion
loop -> QAIC decode -> host ISTFT) as a single Triton model. The stage
artifacts (F5_Preprocess.onnx, qpc_transformer_<cores>c/, qpc_decode_<cores>c/)
live in this model's version directory next to `model.py`; external inputs
(vocab, F5-TTS source checkout) and pipeline knobs come from `config.pbtxt`
parameters.
"""

import json


def _parameters_from_args(args):
    parameters = json.loads(args["model_config"])["parameters"]
    return {key: value["string_value"] for key, value in parameters.items()}


def _string_value(tensor):
    # STRING tensors deserialize to a numpy object array of bytes; the shape
    # depends on the declared dims ([1] -> 1-D, [1,1] -> 2-D), so flatten.
    value = tensor.as_numpy().reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class TritonPythonModel:
    def initialize(self, args):
        parameters = _parameters_from_args(args)
        # Model version directory (repo/f5_tts/1/) where export_triton() copied
        # the stage artifacts. Prefer the `model_dir` config parameter (written
        # at generation time, portable across Triton versions); some python
        # backends also pass `model_instance_folder` as an init arg. `__file__`
        # is NOT usable here: this module is imported from the installed
        # qaic_models package, not from the repo.
        model_version_dir = parameters.get("model_dir") or args.get("model_instance_folder")
        if not model_version_dir:
            raise ValueError(
                "f5_tts Triton model needs the stage-artifact directory: set the "
                "'model_dir' parameter in config.pbtxt to the f5_tts/1/ version directory."
            )

        from qaic_models.speech.f5_tts.model import F5TTSConfig, F5TTSQaic

        self.pipeline = F5TTSQaic(
            F5TTSConfig.from_model_dir(
                model_version_dir,
                f5_tts_src=parameters["f5_tts_src"],
                vocab_path=parameters["vocab_path"],
                transformer_device_ids=parameters.get("transformer_device_ids", "0,1,2,3"),
                decode_device_ids=parameters.get("decode_device_ids", "0"),
                duration_buckets=parameters.get("duration_buckets", "1536,2944"),
                nfe_steps=int(parameters.get("nfe_steps", "16")),
                cfg_strength=float(parameters.get("cfg_strength", "2.0")),
                sway_coefficient=float(parameters.get("sway_coefficient", "-1.0")),
                random_seed=int(parameters.get("seed", "102")),
            )
        )

    def execute(self, requests):
        import numpy as np
        import triton_python_backend_utils as pb_utils

        from qaic_models.speech.f5_tts.preprocess import normalize_to_int16

        responses = []
        for request in requests:
            wav_tensor = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            wav_len = int(pb_utils.get_input_tensor_by_name(request, "reference_wav_len").as_numpy().reshape(-1)[0])
            reference_text = _string_value(pb_utils.get_input_tensor_by_name(request, "reference_text"))
            target_text = _string_value(pb_utils.get_input_tensor_by_name(request, "target_text"))

            reference_wav = wav_tensor.as_numpy().reshape(-1)[:wav_len].astype(np.float32)
            # Same full-range normalization the demo path applies to the loaded
            # file before the (scale-free) mel computation.
            reference_wav = normalize_to_int16(reference_wav).astype(np.float32)
            try:
                result, _ = self.pipeline._infer_from_audio(reference_wav, reference_text, target_text)
            except Exception as exc:
                return [pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))]

            waveform = result.waveform.astype(np.float32) / 32767.0
            outputs = [pb_utils.Tensor("waveform", waveform)]
            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
        return responses
