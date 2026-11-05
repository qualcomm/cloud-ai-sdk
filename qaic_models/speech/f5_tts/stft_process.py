#!/usr/bin/env python

####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

# -*- coding: utf-8 -*-
import torch

NFFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
MAX_SIGNAL_LENGTH = 2048
WINDOW_TYPE = "hann"
PAD_MODE = "constant"

WINDOW_FUNCTIONS = {
    "bartlett": torch.bartlett_window,
    "blackman": torch.blackman_window,
    "hamming": torch.hamming_window,
    "hann": torch.hann_window,
    "kaiser": lambda length: torch.kaiser_window(length, periodic=True, beta=12.0),
}
DEFAULT_WINDOW_FN = torch.hann_window


def create_padded_window(win_length, n_fft, window_type):
    """Return a length-n_fft window, center-padded or cropped as needed."""
    window = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)(win_length).float()
    if win_length == n_fft:
        return window
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        return torch.nn.functional.pad(window, (pad_left, pad_right))
    start = (win_length - n_fft) // 2
    return window[start:start + n_fft]


class STFT_Process(torch.nn.Module):
    def __init__(
        self,
        model_type,
        n_fft=NFFT,
        win_length=WIN_LENGTH,
        hop_len=HOP_LENGTH,
        max_frames=MAX_SIGNAL_LENGTH,
        window_type=WINDOW_TYPE,
    ):
        super().__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.half_n_fft = n_fft // 2

        window = create_padded_window(win_length, n_fft, window_type)
        self.register_buffer("padding_zero", torch.zeros(1, 1, self.half_n_fft, dtype=torch.float32))

        if model_type in ("stft_A", "stft_B"):
            samples = torch.arange(n_fft).float().unsqueeze(0)
            frequencies = torch.arange(self.half_n_fft + 1).float().unsqueeze(1)
            omega = 2 * torch.pi * frequencies * samples / n_fft
            self.register_buffer("cos_kernel", (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1))
            self.register_buffer("sin_kernel", (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1))

        if model_type in ("istft_A", "istft_B"):
            fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
            fourier_basis = torch.vstack([
                torch.real(fourier_basis[:self.half_n_fft + 1]),
                torch.imag(fourier_basis[:self.half_n_fft + 1]),
            ]).float()

            forward_basis = window * fourier_basis.unsqueeze(1)
            inverse_basis = window * torch.linalg.pinv((fourier_basis * n_fft) / hop_len).T.unsqueeze(1)

            total_samples = n_fft + hop_len * (max_frames - 1)
            window_sum = torch.zeros(total_samples, dtype=torch.float32)
            original_window = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)(win_length).float()
            normalized_window = original_window / original_window.abs().max()

            if win_length < n_fft:
                pad_left = (n_fft - win_length) // 2
                pad_right = n_fft - win_length - pad_left
                window_squared = torch.nn.functional.pad(normalized_window ** 2, (pad_left, pad_right))
            else:
                window_squared = normalized_window ** 2

            for frame in range(max_frames):
                start = frame * hop_len
                window_sum[start:start + n_fft] += window_squared[:max(0, min(n_fft, total_samples - start))]

            self.register_buffer("forward_basis", forward_basis)
            self.register_buffer("inverse_basis", inverse_basis)
            self.register_buffer("window_sum_inv", n_fft / (window_sum * hop_len + 1e-7))

    def forward(self, *args):
        if self.model_type == "stft_A":
            return self.stft_A_forward(*args)
        if self.model_type == "stft_B":
            return self.stft_B_forward(*args)
        if self.model_type == "istft_A":
            return self.istft_A_forward(*args)
        if self.model_type == "istft_B":
            return self.istft_B_forward(*args)
        raise ValueError(self.model_type)

    def _pad_input(self, x, mode):
        if mode == "reflect":
            return torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode="reflect")
        return torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)

    def stft_A_forward(self, x, pad_mode="reflect" if PAD_MODE == "reflect" else "constant"):
        return torch.nn.functional.conv1d(self._pad_input(x, pad_mode), self.cos_kernel, stride=self.hop_len)

    def stft_B_forward(self, x, pad_mode="reflect" if PAD_MODE == "reflect" else "constant"):
        x_padded = self._pad_input(x, pad_mode)
        real = torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)
        imag = torch.nn.functional.conv1d(x_padded, self.sin_kernel, stride=self.hop_len)
        return real, imag

    def istft_A_forward(self, magnitude, phase):
        inp = torch.cat((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
        start, end = self.half_n_fft, inv.size(-1) - self.half_n_fft
        return inv[:, :, start:end] * self.window_sum_inv[start:end]

    def istft_B_forward(self, real, imag):
        inp = torch.cat((real, imag), dim=1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
        start, end = self.half_n_fft, inv.size(-1) - self.half_n_fft
        return inv[:, :, start:end] * self.window_sum_inv[start:end]
