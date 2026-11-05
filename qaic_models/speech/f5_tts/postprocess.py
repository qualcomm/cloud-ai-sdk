####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations


def decode_waveform(custom_istft, decode_outputs, *, ref_audio_len, estimated_reference_target_mel_len, hop_length):
    """Convert QAIC decode outputs (magnitude/phase) into a sliced int16 waveform.

    Run ISTFT on the decode outputs, slice out the generated segment
    (discarding the reference audio's own frames), and clamp/scale to int16
    PCM.
    """
    import torch

    torch_denoised = torch.from_numpy(decode_outputs["magnitude"])
    torch_phase = torch.from_numpy(decode_outputs["phase"])
    generated_signal = custom_istft(torch_denoised, torch_phase)
    generated_signal = generated_signal[
        :, :, int((ref_audio_len - 1) * hop_length):int((estimated_reference_target_mel_len - 1) * hop_length)
    ]
    generated_signal = (generated_signal.clamp(min=-1.0, max=1.0) * 32767.0).to(torch.int16)
    return generated_signal.reshape(-1).numpy()
