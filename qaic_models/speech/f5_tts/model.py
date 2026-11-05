####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from qaic_models.base.predict_base import QAICPredictor
from qaic_models.speech.f5_tts.postprocess import decode_waveform
from qaic_models.speech.f5_tts.preprocess import (
    build_transformer_batch_inputs,
    convert_char_to_pinyin,
    list_str_to_idx,
    normalize_to_int16,
    parse_device_ids,
    parse_duration_buckets,
    select_duration_bucket,
)

F5_VOCAB_RELATIVE = "ckpts/models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base/vocab.txt"


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float


def latency_stats(values_ms) -> LatencyStats:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.size == 0:
        return LatencyStats(0.0, 0.0, 0.0, 0.0)
    return LatencyStats(
        mean_ms=float(np.mean(values)),
        p50_ms=float(np.percentile(values, 50)),
        p90_ms=float(np.percentile(values, 90)),
        p99_ms=float(np.percentile(values, 99)),
    )


@dataclass(frozen=True)
class InferResult:
    waveform: np.ndarray
    sample_rate: int
    output_path: Optional[str]
    selected_duration_bucket: int
    latencies: dict


@dataclass(frozen=True)
class F5TTSConfig:
    model_dir: Path
    model: str
    f5_tts_src: Path
    preprocess_onnx: Path
    transformer_qpc: Path
    decode_qpc: Path
    vocab_path: Path
    transformer_device_ids: list
    decode_device_ids: list
    duration_buckets: list
    nfe_steps: int = 16
    cfg_strength: float = 2.0
    sway_coefficient: float = -1.0
    random_seed: int = 102
    preprocess_context_length: int = 2944
    max_signal_length: int = 4096

    @classmethod
    def from_model_dir(
        cls,
        model_dir="./",
        *,
        model: str = "F5TTS_v1_Base",
        f5_tts_src=None,
        preprocess_onnx=None,
        transformer_qpc=None,
        decode_qpc=None,
        vocab_path=None,
        transformer_device_ids="0,1,2,3",
        decode_device_ids="0",
        duration_buckets="1536,2944",
        nfe_steps: int = 16,
        cfg_strength: float = 2.0,
        sway_coefficient: float = -1.0,
        random_seed: int = 102,
    ):
        root = Path(model_dir)
        src = Path(f5_tts_src) if f5_tts_src is not None else Path("src/F5-TTS/src")
        return cls(
            model_dir=root,
            model=model,
            f5_tts_src=src,
            preprocess_onnx=Path(preprocess_onnx) if preprocess_onnx is not None else root / "F5_Preprocess.onnx",
            transformer_qpc=Path(transformer_qpc) if transformer_qpc is not None else root / "qpc_transformer_12c/programqpc.bin",
            decode_qpc=Path(decode_qpc) if decode_qpc is not None else root / "qpc_decode_4c/programqpc.bin",
            vocab_path=Path(vocab_path) if vocab_path is not None else Path(F5_VOCAB_RELATIVE),
            transformer_device_ids=parse_device_ids(transformer_device_ids),
            decode_device_ids=parse_device_ids(decode_device_ids),
            duration_buckets=parse_duration_buckets(duration_buckets),
            nfe_steps=int(nfe_steps),
            cfg_strength=float(cfg_strength),
            sway_coefficient=float(sway_coefficient),
            random_seed=int(random_seed),
        )


class F5TTSQaic(QAICPredictor):
    """Multi-session QAIC implementation of the F5-TTS pipeline.

    Preprocess runs on host CPU via ONNX Runtime. The transformer diffusion loop
    runs as a batch=2 (classifier-free guidance) QAIC session, potentially spread
    across multiple QIDs via MDP/tensor-slicing. Decode runs as a separate
    single-QID QAIC session. Both QAIC sessions use `runtime.QAICInferenceSession`
    (not `qaic.Session`) because the transformer QPC is MDP- and
    multi-specialization-aware, which `qaic.Session` does not expose.

    Expected directory layout (example):
      model_dir/
        F5_Preprocess.onnx
        qpc_transformer_12c/programqpc.bin
        qpc_decode_4c/programqpc.bin
    """

    def __init__(self, config: F5TTSConfig):
        self.config = config
        self._closed = False

        required_paths = [config.preprocess_onnx, config.transformer_qpc, config.decode_qpc, config.vocab_path]
        missing = [p for p in required_paths if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing F5-TTS artifact(s) in F5TTSQaic model_dir. "
                f"Missing: {', '.join(str(p) for p in missing)}. "
                "Please export and compile the model first and try again."
            )

        import torch
        import torchaudio
        from omegaconf import OmegaConf
        from pydub import AudioSegment
        import soundfile as sf

        from qaic_models.base.sessions import QAICInferenceSession, QAICSessionManager
        from qaic_models.speech.f5_tts.stft_process import STFT_Process

        f5_tts_root = str(config.f5_tts_src)
        if f5_tts_root not in sys.path:
            sys.path.insert(0, f5_tts_root)

        self.torch = torch
        self.AudioSegment = AudioSegment
        self.sf = sf

        model_cfg = OmegaConf.load(config.f5_tts_src / f"f5_tts/configs/{config.model}.yaml")
        self.target_sample_rate = int(model_cfg.model.mel_spec.target_sample_rate)
        self.n_mel_channels = int(model_cfg.model.mel_spec.n_mel_channels)
        self.hop_length = int(model_cfg.model.mel_spec.hop_length)
        self.win_length = int(model_cfg.model.mel_spec.win_length)
        self.n_fft = int(model_cfg.model.mel_spec.n_fft)

        self.ort_preprocess = QAICSessionManager().onnx_session(str(config.preprocess_onnx))
        self.qaic_transformer = QAICInferenceSession(str(config.transformer_qpc), device_ids=config.transformer_device_ids)
        self.qaic_decode = QAICInferenceSession(str(config.decode_qpc), device_ids=config.decode_device_ids)

        self.custom_stft = STFT_Process(model_type="stft_B", n_fft=self.n_fft, win_length=self.win_length, hop_len=self.hop_length, max_frames=config.max_signal_length, window_type="hann").eval()
        self.custom_istft = STFT_Process(model_type="istft_A", n_fft=self.n_fft, win_length=self.win_length, hop_len=self.hop_length, max_frames=config.max_signal_length, window_type="hann").eval()
        self.fbank = (
            torchaudio.functional.melscale_fbanks(
                self.n_fft // 2 + 1, 0, self.target_sample_rate // 2,
                self.n_mel_channels, self.target_sample_rate, None, "htk",
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )

        self.vocab_char_map = {}
        with open(config.vocab_path, "r", encoding="utf-8") as vocab_file:
            for i, char in enumerate(vocab_file):
                self.vocab_char_map[char[:-1]] = i

        self.delta_t = self._build_time_schedule()

    def _build_time_schedule(self):
        torch = self.torch
        epss = {
            5: [0, 2, 4, 8, 16, 32],
            6: [0, 2, 4, 6, 8, 16, 32],
            7: [0, 2, 4, 6, 8, 16, 24, 32],
            10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
            12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        }
        t = 1 / 32 * torch.tensor(epss.get(self.config.nfe_steps, list(range(self.config.nfe_steps + 1))), dtype=torch.float32)
        time_step_arr = t + self.config.sway_coefficient * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
        return torch.diff(time_step_arr).to(torch.float32)

    def infer(self, ref_file, ref_text, gen_text, *, file_wave=None, seed=None, warmup=0, runs=1) -> InferResult:
        warmup = int(warmup)
        runs = int(runs)
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if runs < 1:
            raise ValueError("runs must be >= 1")

        result = None
        for _ in range(warmup):
            self._infer_once(ref_file, ref_text, gen_text, file_wave=None, seed=seed)

        latency_values = {}
        for run_index in range(runs):
            write_path = file_wave if run_index == runs - 1 else None
            result, timings = self._infer_once(ref_file, ref_text, gen_text, file_wave=write_path, seed=seed)
            for key, value in timings.items():
                latency_values.setdefault(key, []).append(value)

        return InferResult(
            waveform=result.waveform,
            sample_rate=result.sample_rate,
            output_path=result.output_path,
            selected_duration_bucket=result.selected_duration_bucket,
            latencies={key: latency_stats(values) for key, values in latency_values.items()},
        )

    def _infer_once(self, ref_file, ref_text, gen_text, *, file_wave=None, seed=None):
        start = time.perf_counter()
        audio = np.array(self.AudioSegment.from_file(ref_file).set_channels(1).set_frame_rate(self.target_sample_rate).get_array_of_samples(), dtype=np.float32)
        audio = normalize_to_int16(audio)
        audio = audio.reshape(1, 1, -1)
        timings = {"audio_load": (time.perf_counter() - start) * 1000.0}
        result, result_timings = self._infer_from_audio(audio, ref_text, gen_text, file_wave=file_wave, seed=seed)
        timings.update(result_timings)
        return result, timings

    def _infer_from_audio(self, audio, ref_text, gen_text, *, file_wave=None, seed=None):
        """Run the synthesis pipeline on a loaded mono 24 kHz float32 waveform.

        `audio` may have any amplitude scale (the mel path is scale-free after
        the clamped log); shape (1, 1, samples) or (samples,).
        """
        torch = self.torch
        timings = {}
        e2e_start = time.perf_counter()

        audio = np.asarray(audio, dtype=np.float32).reshape(1, 1, -1)
        audio_len = audio.shape[-1]

        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, gen_text))
        ref_audio_len = audio_len // self.hop_length + 1

        start = time.perf_counter()
        audio = torch.Tensor(audio)
        audio = audio.float() * float(1.0 / 32768.0)
        mel_signal_real, mel_signal_imag = self.custom_stft(audio, "reflect")
        mel_signal = torch.matmul(self.fbank, torch.sqrt(mel_signal_real * mel_signal_real + mel_signal_imag * mel_signal_imag)).transpose(1, 2).clamp(min=1e-5).log()
        timings["stft"] = (time.perf_counter() - start) * 1000.0

        estimated_reference_target_mel_len = mel_signal.shape[1] * (1 + gen_text_len / ref_text_len)
        pad_len = self.config.preprocess_context_length - mel_signal.shape[1]
        ref_mel = mel_signal.permute(0, 2, 1)
        padded_ref_mel = torch.nn.functional.pad(ref_mel, (0, pad_len), value=0)
        mel_signal = padded_ref_mel.permute(0, 2, 1)

        text_ids = list_str_to_idx(convert_char_to_pinyin([ref_text + gen_text]), self.vocab_char_map)
        input_length = np.array([estimated_reference_target_mel_len]).astype(np.int32)

        inference_start = time.perf_counter()
        start = time.perf_counter()
        rope_cos_q_pad, rope_sin_q_pad, rope_cos_k_pad, rope_sin_k_pad, cat_mel_text_pad, cat_mel_text_drop_pad = self.ort_preprocess.run(
            ["rope_cos_q", "rope_sin_q", "rope_cos_k", "rope_sin_k", "cat_mel_text", "cat_mel_text_drop"],
            {"mel_signal": mel_signal.numpy(), "text_ids": text_ids.numpy(), "input_lengths": input_length.astype(np.int64)},
        )
        timings["onnx_preprocess"] = (time.perf_counter() - start) * 1000.0

        rope_cos_q = np.ascontiguousarray(rope_cos_q_pad[:1])
        rope_sin_q = np.ascontiguousarray(rope_sin_q_pad[:1])
        rope_cos_k = np.ascontiguousarray(rope_cos_k_pad[:1])
        rope_sin_k = np.ascontiguousarray(rope_sin_k_pad[:1])

        required_duration = int(np.ceil(float(input_length[0])))
        bucket = select_duration_bucket(required_duration, self.config.duration_buckets)
        if bucket > self.config.preprocess_context_length:
            raise ValueError(f"Selected duration bucket {bucket} exceeds preprocess context {self.config.preprocess_context_length}")
        input_length = np.array([min(required_duration, bucket)], dtype=np.int32)

        rope_cos_q = np.ascontiguousarray(rope_cos_q[:, :, :bucket, :])
        rope_sin_q = np.ascontiguousarray(rope_sin_q[:, :, :bucket, :])
        rope_cos_k = np.ascontiguousarray(rope_cos_k[:, :, :, :bucket])
        rope_sin_k = np.ascontiguousarray(rope_sin_k[:, :, :, :bucket])
        cat_mel_text_pad = np.ascontiguousarray(cat_mel_text_pad[:, :bucket, :])
        cat_mel_text_drop_pad = np.ascontiguousarray(cat_mel_text_drop_pad[:, :bucket, :])

        torch.manual_seed(self.config.random_seed if seed is None else int(seed))
        noise = torch.randn(1, bucket, self.n_mel_channels)
        time_step = np.array([0], dtype=np.int32)

        start = time.perf_counter()
        for _ in range(self.config.nfe_steps):
            batch_inputs = build_transformer_batch_inputs(noise.numpy(), cat_mel_text_pad, cat_mel_text_drop_pad, input_length)
            out = self.qaic_transformer.run({
                "cat_mel_text": batch_inputs["cat_mel_text"],
                "noise": batch_inputs["noise"],
                "rope_cos_k": rope_cos_k,
                "rope_cos_q": rope_cos_q,
                "rope_sin_k": rope_sin_k,
                "rope_sin_q": rope_sin_q,
                "time_step": time_step,
                "input_lengths": batch_inputs["input_lengths"],
            })

            denoised = torch.Tensor(out["denoised"])
            denoised_cond = denoised[0:1]
            denoised_uncond = denoised[1:2]
            noise += (denoised_cond + (denoised_cond - denoised_uncond) * self.config.cfg_strength) * self.delta_t[int(time_step[0])]
            time_step += 1
        timings["transformer"] = (time.perf_counter() - start) * 1000.0

        if bucket < self.config.preprocess_context_length:
            noise = torch.nn.functional.pad(noise, (0, 0, 0, self.config.preprocess_context_length - bucket), value=0.0)
        sliced_noise = np.ascontiguousarray(noise.numpy())

        start = time.perf_counter()
        decode_outputs = self.qaic_decode.run({"denoised": sliced_noise})
        timings["decode"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        waveform = decode_waveform(
            self.custom_istft,
            decode_outputs,
            ref_audio_len=ref_audio_len,
            estimated_reference_target_mel_len=estimated_reference_target_mel_len,
            hop_length=self.hop_length,
        )
        timings["istft"] = (time.perf_counter() - start) * 1000.0
        timings["inference"] = (time.perf_counter() - inference_start) * 1000.0

        output_path = None
        if file_wave is not None:
            start = time.perf_counter()
            self.sf.write(file_wave, waveform, self.target_sample_rate, format="WAVEX")
            timings["write_audio"] = (time.perf_counter() - start) * 1000.0
            output_path = file_wave

        timings["e2e"] = (time.perf_counter() - e2e_start) * 1000.0
        return InferResult(waveform=waveform, sample_rate=self.target_sample_rate, output_path=output_path, selected_duration_bucket=bucket, latencies={}), timings

    def close(self):
        if self._closed:
            return
        for session in (getattr(self, "qaic_transformer", None), getattr(self, "qaic_decode", None)):
            if session is not None:
                session.deactivate()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class F5TTS:
    """Public factory for F5-TTS inference on Cloud AI 100.

    API naming matches upstream `f5_tts.api.F5TTS`: `infer(ref_file, ref_text,
    gen_text, ...)` returning a structured `InferResult`.
    """

    def __init__(
        self,
        model_dir: str = "./",
        *,
        model: str = "F5TTS_v1_Base",
        provider: str = "QAicExecutionProvider",
        f5_tts_src=None,
        preprocess_onnx=None,
        transformer_qpc=None,
        decode_qpc=None,
        vocab_path=None,
        transformer_device_ids="0,1,2,3",
        decode_device_ids="0",
        duration_buckets="1536,2944",
        nfe_steps: int = 16,
        cfg_strength: float = 2.0,
        sway_coefficient: float = -1.0,
        random_seed: int = 102,
    ):
        self.config = F5TTSConfig.from_model_dir(
            model_dir,
            model=model,
            f5_tts_src=f5_tts_src,
            preprocess_onnx=preprocess_onnx,
            transformer_qpc=transformer_qpc,
            decode_qpc=decode_qpc,
            vocab_path=vocab_path,
            transformer_device_ids=transformer_device_ids,
            decode_device_ids=decode_device_ids,
            duration_buckets=duration_buckets,
            nfe_steps=nfe_steps,
            cfg_strength=cfg_strength,
            sway_coefficient=sway_coefficient,
            random_seed=random_seed,
        )
        if provider != "QAicExecutionProvider":
            raise ValueError(f"Unsupported provider: {provider}")
        self._impl = None

    @property
    def impl(self) -> F5TTSQaic:
        if self._impl is None:
            self._impl = F5TTSQaic(self.config)
        return self._impl

    def infer(
        self,
        ref_file: str,
        ref_text: str,
        gen_text: str,
        *,
        file_wave: Optional[str] = None,
        seed: Optional[int] = None,
        warmup: int = 0,
        runs: int = 1,
    ) -> InferResult:
        if not ref_text:
            raise ValueError("ref_text must be provided; automatic ASR transcription is not supported in this recipe")
        if int(warmup) < 0:
            raise ValueError("warmup must be >= 0")
        if int(runs) < 1:
            raise ValueError("runs must be >= 1")
        return self.impl.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=file_wave,
            seed=seed,
            warmup=warmup,
            runs=runs,
        )

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
            self._impl = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
