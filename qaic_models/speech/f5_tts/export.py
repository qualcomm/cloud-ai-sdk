####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import types
from pathlib import Path

import onnx

from qaic_models.base.compile import qaic_compile_parallel
from qaic_models.base.onnx_transforms import FP16ClipTransform, run_pipeline
from qaic_models.base.triton import TritonModelConfig

EXPECTED_OUTPUTS = {
    "F5_Preprocess.onnx": ["rope_cos_q", "rope_sin_q", "rope_cos_k", "rope_sin_k", "cat_mel_text", "cat_mel_text_drop"],
    "F5_Transformer.onnx": ["denoised"],
    "F5_Decode.onnx": ["magnitude", "phase"],
}

DEFAULT_TRANSFORMER_CORES = 12
DEFAULT_DECODE_CORES = 4
DEFAULT_MAX_DURATION = 2944
DEFAULT_TRANSFORMER_BATCH = 2
DEFAULT_DURATION_SPECIALIZATIONS = [1536, 2944]


def validate_onnx_output_contract(onnx_path, *, expected_output_names):
    """Confirm the exported graph exposes exactly the expected output tensor names.

    Raises ValueError naming the mismatch so export failures are caught immediately
    instead of surfacing later as a confusing runtime shape error.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    actual_names = [o.name for o in model.graph.output]
    if actual_names != list(expected_output_names):
        raise ValueError(
            f"{onnx_path}: expected output names {expected_output_names}, got {actual_names}"
        )


def _mdp_partition_config(transformer_cores):
    return {
        "connections": [{"devices": [0, 1, 2, 3], "type": "p2p"}],
        "partitions": [
            {
                "name": "Partition0",
                "devices": [{"deviceId": qid, "numCores": transformer_cores} for qid in range(4)],
            }
        ],
    }


def _duration_specialization_config(duration_buckets):
    return {"specializations": [{"max_duration": str(bucket)} for bucket in duration_buckets]}


def export_onnx(f5_tts_src, work_dir, *, model="F5TTS_v1_Base", ckpt_dir=None):
    """Export the three F5-TTS ONNX graphs (Preprocess/Transformer/Decode) to `work_dir`.

    Exports the preprocess, transformer, and decode stages with the rotary
    attention rewrite, Vocos ONNX-friendly forward patches, and symbolic export
    axes required by this recipe. Requires a pinned F5-TTS/Vocos source checkout
    at `f5_tts_src` (see README "Environment setup") and downloaded checkpoints
    under `ckpt_dir`.
    """
    import torch
    import torchaudio
    from omegaconf import OmegaConf

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    f5_tts_src = Path(f5_tts_src).resolve()
    f5_tts_model_src = f5_tts_src / "f5_tts" / "model"
    vocos_src = f5_tts_src.parent / "vocos" if (f5_tts_src.parent / "vocos").exists() else f5_tts_src.parent.parent / "vocos"

    for path in (str(f5_tts_src), str(vocos_src)):
        if path not in sys.path:
            sys.path.insert(0, path)

    f5_tts_model = types.ModuleType("f5_tts.model")
    f5_tts_model.__path__ = [str(f5_tts_model_src)]
    sys.modules["f5_tts.model"] = f5_tts_model

    from qaic_models.speech.f5_tts.stft_process import STFT_Process
    from f5_tts.model.backbones.dit import DiT  # noqa: F401 (registered via globals())
    from f5_tts.model.cfm import CFM
    from vocos import Vocos

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else work_dir / "ckpts"
    safetensors_path = ckpt_dir / "models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base/model_1250000.safetensors"
    vocab_path = ckpt_dir / "models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base/vocab.txt"
    vocos_model_path = ckpt_dir / "models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"

    nfe_step = 16
    sample_rate = 24000
    sway_coefficient = -1.0
    hop_length = 256
    n_mels = 100
    nfft = 1024
    window_length = 960
    window_type = "hann"
    max_signal_length = 4096
    context_length = 2944
    text_ids_length = 286
    text_embed_length = 512 + n_mels
    max_duration = context_length

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)

    def load_checkpoint(torch_model, ckpt_path, device="cpu", use_ema=True):
        ckpt_type = str(ckpt_path).split(".")[-1]
        if ckpt_type == "safetensors":
            from safetensors.torch import load_file
            checkpoint = load_file(str(ckpt_path), device=device)
        else:
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch -- weights_only=True refuses pickle payloads (tensors/containers only).
            checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)

        if use_ema:
            if ckpt_type == "safetensors":
                checkpoint = {"ema_model_state_dict": checkpoint}
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                checkpoint["model_state_dict"].pop(key, None)
        elif ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}

        torch_model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        return torch_model.to(device)

    def load_vocos_model(model_dir):
        model_dir = Path(model_dir)
        vocos_model = Vocos.from_hparams(str(model_dir / "config.yaml"))
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch -- weights_only=True refuses pickle payloads (tensors/containers only).
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        vocos_model.load_state_dict(state_dict)
        vocos_model.eval()
        return vocos_model

    def export_conv_position_embedding_forward(conv_pos_embed, x, mask=None):
        x = x.permute(0, 2, 1)
        if mask is not None:
            mask = mask[:, None, :].to(x.dtype)
            x = x * mask
        x = conv_pos_embed.conv1d[0](x)
        x = conv_pos_embed.conv1d[1](x)
        if mask is not None:
            x = x * mask
        x = conv_pos_embed.conv1d[2](x)
        x = conv_pos_embed.conv1d[3](x)
        if mask is not None:
            x = x * mask
        return x.permute(0, 2, 1)

    def export_input_embedding_forward(input_embed, x, cond, mask=None):
        x = input_embed.proj(torch.cat((x, cond), dim=-1))
        x = export_conv_position_embedding_forward(input_embed.conv_pos_embed, x, mask=mask) + x
        return x

    def export_adaln_forward(adaln, x, emb=None):
        emb = adaln.linear(adaln.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=-1)
        x = adaln.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def export_adaln_final_forward(adaln, x, emb):
        emb = adaln.linear(adaln.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        return adaln.norm(x) * (1 + scale) + shift

    def rotate_half_q(x, heads, head_dim, head_dim_half):
        batch = x.shape[0]
        x = x.view(batch, heads, -1, head_dim_half, 2)
        x1, x2 = torch.split(x, [1, 1], dim=-1)
        return torch.cat((-x2, x1), dim=-1).reshape(batch, heads, -1, head_dim)

    def rotate_half_k(x, heads, head_dim, head_dim_half):
        batch = x.shape[0]
        x = x.view(batch, heads, head_dim_half, 2, -1)
        x1, x2 = torch.split(x, [1, 1], dim=-2)
        return torch.cat((-x2, x1), dim=-2).reshape(batch, heads, head_dim, -1)

    def apply_rotary_q(x, rope_cos, rope_sin, heads, head_dim, head_dim_half):
        return x * rope_cos + rotate_half_q(x, heads, head_dim, head_dim_half) * rope_sin

    def apply_rotary_k(x, rope_cos, rope_sin, heads, head_dim, head_dim_half):
        return x * rope_cos + rotate_half_k(x, heads, head_dim, head_dim_half) * rope_sin

    def inner_matmul(hidden_states, query_states, key_states, head_dim, attention_mask, value_states, blk_size):
        seq_len = hidden_states.size(1)
        block_size = min(blk_size, seq_len)
        num_blocks = max(1, (seq_len + block_size - 1) // block_size)
        attn_output_blocks = []
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, seq_len)
            query_block = query_states[:, :, start_idx:end_idx, :]
            attn_weights = torch.matmul(query_block, key_states)
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output_blocks.append(torch.matmul(attn_weights, value_states))
        return torch.cat(attn_output_blocks, dim=2)

    def export_attention_forward(attn, x, mask=None, rope_cos_q=None, rope_sin_q=None, rope_cos_k=None, rope_sin_k=None):
        head_dim = attn.inner_dim // attn.heads
        head_dim_half = head_dim // 2
        batch = x.shape[0]
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        query = apply_rotary_q(query, rope_cos_q, rope_sin_q, attn.heads, head_dim, head_dim_half)
        key = key.permute(0, 1, 3, 2)
        key = apply_rotary_k(key, rope_cos_k, rope_sin_k, attn.heads, head_dim, head_dim_half)
        attn_mask = (1 - mask[:, None, None, :].to(query.dtype)) * torch.tensor(-65504.0, dtype=query.dtype)
        x = inner_matmul(x, query, key, head_dim, attn_mask, value, 64)
        x = x.transpose(1, 2).reshape(batch, -1, attn.dim)
        x = attn.to_out[0](x)
        if mask is not None:
            mask = mask[:, :, None].expand(-1, -1, x.shape[-1])
            x = x * mask
        return x

    def export_dit_block_forward(block, x, t, mask=None, rope_cos_q=None, rope_sin_q=None, rope_cos_k=None, rope_sin_k=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = export_adaln_forward(block.attn_norm, x, emb=t)
        attn_output = export_attention_forward(
            block.attn, norm, mask=mask, rope_cos_q=rope_cos_q, rope_sin_q=rope_sin_q, rope_cos_k=rope_cos_k, rope_sin_k=rope_sin_k
        )
        x = x + gate_msa * attn_output
        norm = block.ff_norm(x) * (1 + scale_mlp) + shift_mlp
        ff_output = block.ff(norm)
        return x + gate_mlp * ff_output

    def export_dit_forward(transformer, x, cond, time_emb, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, input_lengths):
        batch, seq_len = x.shape[0], x.shape[1]
        position_ids_buffer = torch.arange(4096, dtype=torch.int32).reshape(1, -1)
        position_ids = position_ids_buffer[:, :seq_len].expand(batch, seq_len)
        attention_mask = position_ids < input_lengths.unsqueeze(1).expand(batch, seq_len)
        x = export_input_embedding_forward(transformer.input_embed, x, cond, mask=attention_mask)
        if transformer.long_skip_connection is not None:
            residual = x
        for block in transformer.transformer_blocks:
            x = export_dit_block_forward(
                block, x, time_emb, mask=attention_mask, rope_cos_q=rope_cos_q, rope_sin_q=rope_sin_q, rope_cos_k=rope_cos_k, rope_sin_k=rope_sin_k
            )
        if transformer.long_skip_connection is not None:
            x = transformer.long_skip_connection(torch.cat((x, residual), dim=-1))
        return transformer.proj_out(export_adaln_final_forward(transformer.norm_out, x, time_emb))

    class F5Preprocess(torch.nn.Module):
        def __init__(self, f5_model, n_mels, num_head, head_dim):
            super().__init__()
            self.f5_text_embed = f5_model.transformer.text_embed
            self.num_channels = n_mels
            base_rescale_factor = 1.0
            interpolation_factor = 1.0
            base = 10000.0 * base_rescale_factor ** (head_dim / (head_dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            freqs = torch.outer(torch.arange(max_signal_length, dtype=torch.float32), inv_freq) / interpolation_factor
            freqs = freqs.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0).repeat(2, num_head, 1, 1)
            self.rope_cos = freqs.cos().half()
            self.rope_sin = freqs.sin().half()

        def forward(self, mel_signal, text_ids, input_lengths):
            duration = torch.tensor([max_duration])
            zeros = torch.zeros((1, duration, self.num_channels), dtype=torch.float32)
            rope_cos_q = self.rope_cos[:, :, :duration]
            rope_sin_q = self.rope_sin[:, :, :duration]
            rope_cos_k = rope_cos_q.transpose(-1, -2)
            rope_sin_k = rope_sin_q.transpose(-1, -2)
            text = self.f5_text_embed(text_ids, input_lengths[0], drop_text=False)
            text_drop = self.f5_text_embed(text_ids, input_lengths[0], drop_text=True)
            text = torch.nn.functional.pad(text, (0, 0, 0, duration[0] - text.shape[1], 0, 0), value=0)
            text_drop = torch.nn.functional.pad(text_drop, (0, 0, 0, duration[0] - text_drop.shape[1], 0, 0), value=0)
            cat_mel_text = torch.cat((mel_signal, text), dim=-1)
            cat_mel_text_drop = torch.cat((zeros, text_drop), dim=-1)
            return rope_cos_q.float(), rope_sin_q.float(), rope_cos_k.float(), rope_sin_k.float(), cat_mel_text, cat_mel_text_drop

    class F5Transformer(torch.nn.Module):
        def __init__(self, f5_model, steps, sway_coef, dtype):
            super().__init__()
            self.f5_transformer = f5_model.transformer
            self.time_mlp = f5_model.transformer.time_embed.time_mlp
            freq_embed_dim = 256
            time_mlp_dim = 1024
            self.sway_sampling_coef = sway_coef
            epss = {
                5: [0, 2, 4, 8, 16, 32],
                6: [0, 2, 4, 6, 8, 16, 32],
                7: [0, 2, 4, 6, 8, 16, 24, 32],
                10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
                12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
                16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            }
            t = 1 / 32 * torch.tensor(epss.get(steps, list(range(steps + 1))), dtype=torch.float32)
            time_step = t + self.sway_sampling_coef * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
            self.time_expand = torch.zeros((1, len(time_step), time_mlp_dim), dtype=torch.float32)
            half_dim = freq_embed_dim // 2
            emb_factor = math.log(10000) / (half_dim - 1)
            emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
            for i in range(len(time_step)):
                emb = time_step[i] * emb_factor
                emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
                self.time_expand[:, [i]] = self.time_mlp(emb)
            self.time_expand = self.time_expand.to(dtype)

        def forward(self, noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, cat_mel_text, time_step, input_lengths):
            return export_dit_forward(
                self.f5_transformer,
                x=noise,
                cond=cat_mel_text,
                time_emb=self.time_expand[:, time_step],
                rope_cos_q=rope_cos_q,
                rope_sin_q=rope_sin_q,
                rope_cos_k=rope_cos_k,
                rope_sin_k=rope_sin_k,
                input_lengths=input_lengths,
            )

    class F5Decode(torch.nn.Module):
        def __init__(self, vocos_model):
            super().__init__()
            self.vocos = vocos_model

        def forward(self, denoised):
            return self.vocos.decode(denoised.transpose(1, 2))

    def load_model(ckpt_path):
        model_cfg = OmegaConf.load(f5_tts_src / "f5_tts" / "configs" / f"{model}.yaml")
        model_cls = {"DiT": DiT}[model_cfg.model.backbone]
        cfm_model = CFM(
            transformer=model_cls(**model_cfg.model.arch, text_num_embeds=vocab_size, mel_dim=n_mels),
            mel_spec_kwargs=dict(target_sample_rate=sample_rate, n_mel_channels=n_mels, hop_length=hop_length),
            odeint_kwargs=dict(method="euler"),
            vocab_char_map=vocab_char_map,
        ).to("cpu")
        return load_checkpoint(cfm_model, ckpt_path), model_cfg.model.arch.heads, model_cfg.model.arch.dim

    with torch.inference_mode():
        text_ids = torch.ones((1, text_ids_length), dtype=torch.int32)
        f5_model, num_head, hidden_size = load_model(safetensors_path)
        head_dim = hidden_size // num_head
        custom_stft = STFT_Process(model_type="stft_B", n_fft=nfft, win_length=window_length, hop_len=hop_length, max_frames=0, window_type=window_type).eval()

        audio = torch.ones((1, 1, ((max_duration - 1) * hop_length)), dtype=torch.int16)
        audio = audio.float() * float(1.0 / 32768.0)
        fbank = torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, None, "htk").transpose(0, 1).unsqueeze(0)
        mel_signal_real, mel_signal_imag = custom_stft(audio, "reflect")
        mel_signal = torch.matmul(fbank, torch.sqrt(mel_signal_real * mel_signal_real + mel_signal_imag * mel_signal_imag)).transpose(1, 2).clamp(min=1e-5).log()
        input_lengths = torch.tensor([100], dtype=torch.int64)

        f5_preprocess = F5Preprocess(f5_model, n_mels, num_head, head_dim)
        torch.onnx.export(
            f5_preprocess,
            (mel_signal, text_ids, input_lengths),
            str(work_dir / "F5_Preprocess.onnx"),
            input_names=["mel_signal", "text_ids", "input_lengths"],
            output_names=EXPECTED_OUTPUTS["F5_Preprocess.onnx"],
            dynamic_axes={"text_ids": {1: "text_ids_len"}},
            do_constant_folding=True,
            opset_version=17,
        )
        del custom_stft, f5_preprocess, audio, text_ids
        gc.collect()

        scale_factor = math.pow(head_dim, -0.25)
        dtype = torch.float32
        for i in range(len(f5_model.transformer.transformer_blocks)):
            block = f5_model.transformer.transformer_blocks._modules[f"{i}"]
            block.attn.to_q.weight.data *= scale_factor
            block.attn.to_q.bias.data *= scale_factor
            block.attn.to_k.weight.data *= scale_factor
            block.attn.to_k.bias.data *= scale_factor

        noise = torch.ones((1, max_duration, n_mels), dtype=dtype)
        rope_cos_q = torch.ones((1, num_head, max_duration, head_dim), dtype=dtype)
        rope_sin_q = torch.ones((1, num_head, max_duration, head_dim), dtype=dtype)
        rope_cos_k = rope_cos_q.transpose(-1, -2)
        rope_sin_k = rope_sin_q.transpose(-1, -2)
        cat_mel_text = torch.randn((1, max_duration, text_embed_length), dtype=dtype)
        time_step = torch.tensor([0], dtype=torch.int32)
        input_lengths = torch.tensor([850], dtype=torch.int32)

        f5_transformer = F5Transformer(f5_model, steps=nfe_step, sway_coef=sway_coefficient, dtype=dtype)
        torch.onnx.export(
            f5_transformer,
            (noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, cat_mel_text, time_step, input_lengths),
            str(work_dir / "F5_Transformer.onnx"),
            input_names=["noise", "rope_cos_q", "rope_sin_q", "rope_cos_k", "rope_sin_k", "cat_mel_text", "time_step", "input_lengths"],
            output_names=EXPECTED_OUTPUTS["F5_Transformer.onnx"],
            dynamic_axes={
                "noise": {0: "batch", 1: "max_duration"},
                "rope_cos_q": {2: "max_duration"},
                "rope_sin_q": {2: "max_duration"},
                "rope_cos_k": {3: "max_duration"},
                "rope_sin_k": {3: "max_duration"},
                "cat_mel_text": {0: "batch", 1: "max_duration"},
                "input_lengths": {0: "batch"},
                "denoised": {0: "batch", 1: "max_duration"},
            },
            do_constant_folding=True,
            opset_version=17,
        )
        del f5_transformer, noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, cat_mel_text, time_step
        gc.collect()

        denoised = torch.ones((1, max_duration, n_mels), dtype=dtype)
        vocos_model = load_vocos_model(vocos_model_path)
        from qaic_models.speech.f5_tts.pytorch_transforms import apply_vocos_export_overlays, undo_vocos_export_overlays

        vocos_model, vocos_transforms, _ = apply_vocos_export_overlays(vocos_model)
        try:
            f5_decode = F5Decode(vocos_model)
            torch.onnx.export(
                f5_decode,
                (denoised,),
                str(work_dir / "F5_Decode.onnx"),
                input_names=["denoised"],
                output_names=EXPECTED_OUTPUTS["F5_Decode.onnx"],
                dynamic_axes={"denoised": {1: "max_duration"}, "magnitude": {2: "generated_len"}},
                do_constant_folding=True,
                opset_version=17,
            )
        finally:
            undo_vocos_export_overlays(vocos_transforms)
        del f5_decode, denoised, vocos_model
        gc.collect()

    for onnx_name, expected_names in EXPECTED_OUTPUTS.items():
        validate_onnx_output_contract(work_dir / onnx_name, expected_output_names=expected_names)

    fixed_name = "F5_Transformer_fixed_for_fp16"
    run_pipeline(str(work_dir / "F5_Transformer.onnx"), str(work_dir / f"{fixed_name}.onnx"), [FP16ClipTransform])
    validate_onnx_output_contract(work_dir / f"{fixed_name}.onnx", expected_output_names=EXPECTED_OUTPUTS["F5_Transformer.onnx"])

    return {
        "preprocess_onnx": work_dir / "F5_Preprocess.onnx",
        "transformer_onnx": work_dir / f"{fixed_name}.onnx",
        "decode_onnx": work_dir / "F5_Decode.onnx",
    }


def transformer_compile_spec(work_dir, *, cores=DEFAULT_TRANSFORMER_CORES, duration_buckets=None, batch=DEFAULT_TRANSFORMER_BATCH) -> dict:
    """qaic_compile kwargs for the batch=2 TS4 transformer QPC (writes its
    MDP and duration-specialization config files into work_dir first)."""
    work_dir = Path(work_dir).resolve()
    duration_buckets = duration_buckets or DEFAULT_DURATION_SPECIALIZATIONS
    onnx_path = work_dir / "F5_Transformer_fixed_for_fp16.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Transformer ONNX not found at {onnx_path}; run export_onnx() first")

    mdp_config_path = work_dir / "mdp_ts4.json"
    mdp_config_path.write_text(json.dumps(_mdp_partition_config(cores), indent=2))

    specializations_path = work_dir / "f5_transformer_duration_specializations.json"
    specializations_path.write_text(json.dumps(_duration_specialization_config(duration_buckets), indent=2))

    qpc_dir = work_dir / f"qpc_transformer_{cores}c"
    return {
        "model": str(onnx_path),
        "cores": cores,
        "aic_binary_dir": qpc_dir,
        "cwd": work_dir,
        "mos": 4,
        "extra_args": (
            "-mxfp6-matmul",
            f"-onnx-define-symbol=batch,{batch}",
            f"-network-specialization-config={specializations_path}",
            "-vvv",
            f"-mdp-load-partition-config={mdp_config_path}",
            "-mdts-mos=1",
        ),
    }


def decode_compile_spec(work_dir, *, cores=DEFAULT_DECODE_CORES, max_duration=DEFAULT_MAX_DURATION) -> dict:
    """qaic_compile kwargs for the single-QID decode QPC."""
    work_dir = Path(work_dir).resolve()
    onnx_path = work_dir / "F5_Decode.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Decode ONNX not found at {onnx_path}; run export_onnx() first")

    return {
        "model": str(onnx_path),
        "cores": cores,
        "aic_binary_dir": work_dir / f"qpc_decode_{cores}c",
        "cwd": work_dir,
        "extra_args": (f"-onnx-define-symbol=max_duration,{max_duration}", "-vvv"),
    }


_F5_TTS_MODEL_STUB = (
    "# Generated by qaic_models F5-TTS export. Triton loads this as the python\n"
    "# backend model; the implementation lives in the installed qaic_models package.\n"
    "from qaic_models.speech.f5_tts.triton_model import TritonPythonModel\n"
)


def export_triton(
    work_dir,
    *,
    repo_root,
    model_count=1,
    f5_tts_src,
    vocab_path,
    transformer_cores=DEFAULT_TRANSFORMER_CORES,
    decode_cores=DEFAULT_DECODE_CORES,
    triton_protobuf=None,
):
    """Generate a single-model Triton repository serving the full F5-TTS pipeline.

    Writes `repo_root/f5_tts/` (python backend) and copies the generated stage
    artifacts into the model version directory so the repository is
    self-contained: the diffusion loop is stateful (host-side noise/CFG update
    between QAIC calls), so the stages are not exposed as separate Triton
    models and there is no static ensemble.
    """
    work_dir = Path(work_dir).resolve()
    repo_root = Path(repo_root).resolve()

    transformer_qpc_dir = work_dir / f"qpc_transformer_{transformer_cores}c"
    decode_qpc_dir = work_dir / f"qpc_decode_{decode_cores}c"
    preprocess_onnx = work_dir / "F5_Preprocess.onnx"
    missing = [
        path
        for path in (transformer_qpc_dir / "programqpc.bin", decode_qpc_dir / "programqpc.bin", preprocess_onnx)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing generated artifact(s) for Triton repo: "
            f"{', '.join(str(p) for p in missing)}. Run export.py with --compile first."
        )

    model_root = repo_root / "f5_tts"
    version_dir = model_root / str(1)
    version_dir.mkdir(parents=True, exist_ok=True)

    TritonModelConfig(
        name="f5_tts",
        backend="python",
        model_source=_F5_TTS_MODEL_STUB,
        count=model_count,
        inputs=[
            {"name": "reference_wav", "data_type": "TYPE_FP32", "dims": [-1]},
            {"name": "reference_wav_len", "data_type": "TYPE_INT32", "dims": [1]},
            {"name": "reference_text", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "target_text", "data_type": "TYPE_STRING", "dims": [1]},
        ],
        outputs=[
            {"name": "waveform", "data_type": "TYPE_FP32", "dims": [-1]},
        ],
        parameters={
            "vocab_path": str(Path(vocab_path).resolve()),
            "f5_tts_src": str(Path(f5_tts_src).resolve()),
            # Version directory receiving the stage artifacts below; the python
            # model locates them there instead of relying on Triton init args
            # (not all Triton python backends expose model_instance_folder).
            "model_dir": str(version_dir),
        },
    ).tofile(model_root, triton_protobuf=triton_protobuf)

    shutil.copyfile(preprocess_onnx, version_dir / "F5_Preprocess.onnx")
    for qpc_dir in (transformer_qpc_dir, decode_qpc_dir):
        shutil.copytree(qpc_dir, version_dir / qpc_dir.name, dirs_exist_ok=True)

    print(f"[triton] Populated {model_root} (backend: python, count: {model_count})")
    return model_root


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export and (optionally) compile F5-TTS for Cloud AI 100")
    parser.add_argument("--work-dir", required=True, help="Directory to write generated ONNX/QPC artifacts into")
    parser.add_argument("--f5-tts-src", required=True, help="Path to a pinned F5-TTS source checkout's src/ directory (see README)")
    parser.add_argument("--ckpt-dir", default=None, help="Directory containing downloaded F5-TTS/Vocos checkpoints (default: <work-dir>/ckpts)")
    parser.add_argument("--model", default="F5TTS_v1_Base", choices=["F5TTS_v1_Base"])
    parser.add_argument("--transformer-cores", type=int, default=DEFAULT_TRANSFORMER_CORES)
    parser.add_argument("--decode-cores", type=int, default=DEFAULT_DECODE_CORES)
    parser.add_argument("--max-duration", type=int, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--compile", action="store_true", help="Compile transformer and decode QPCs after export")
    parser.add_argument("--triton-model-repo", type=str, default=None, help="Generate a Triton model repository at the given path (requires --compile)")
    parser.add_argument("--triton-model-count", type=int, default=1, help="Number of instances for the Triton python model")
    args = parser.parse_args(argv)

    export_onnx(args.f5_tts_src, args.work_dir, model=args.model, ckpt_dir=args.ckpt_dir)
    if args.compile:
        # The two QPCs share no inputs or targets; qaic-compile writes only
        # into its own binary dir, so they compile concurrently.
        qaic_compile_parallel([
            transformer_compile_spec(args.work_dir, cores=args.transformer_cores),
            decode_compile_spec(args.work_dir, cores=args.decode_cores, max_duration=args.max_duration),
        ])
    if args.triton_model_repo:
        if not args.compile:
            raise SystemExit("--triton-model-repo requires --compile so the QPC artifacts exist in the work directory")
        ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(args.work_dir).resolve() / "ckpts"
        export_triton(
            args.work_dir,
            repo_root=args.triton_model_repo,
            model_count=args.triton_model_count,
            f5_tts_src=args.f5_tts_src,
            vocab_path=ckpt_dir / "models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base/vocab.txt",
            transformer_cores=args.transformer_cores,
            decode_cores=args.decode_cores,
        )


if __name__ == "__main__":
    main()
