#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# gRPC benchmark client for the F5-TTS Triton model on Cloud AI 100.
#
# Runs the same reference clip + prompt N times against the `f5_tts` Triton
# model and reports per-request latency and Real-Time Factor (RTF = wall-clock
# / generated-audio duration). Mirrors the benchmarking approach from the
# upstream F5-TTS Triton runtime:
#   https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/runtime/triton_trtllm
#
# Usage (from the recipe directory, server running on the gRPC port):
#   python3 benchmark_grpc.py \
#       --server-addr localhost \
#       --server-port   8001 \
#       --ref-audio   src/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
#       --ref-text    "Some call me nature, others call me mother nature." \
#       --target-text "I don't really care what you call me." \
#       --warmup \
#       --repeat      5 \
#       --log-dir     ./benchmark_results
#
# The reference transcript must actually describe the reference audio — F5-TTS
# uses it as a conditioning input, so a placeholder transcript degrades the
# output.

import argparse
import asyncio
import json
import os
import shutil
import time

import numpy as np
import soundfile as sf
import torchaudio
import tritonclient.grpc.aio as grpcclient


# ─── Audio / tensor helpers ────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 24000):
    """Load WAV/MP3/FLAC and return (float32 1-D array, num_samples)."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:  # mix down to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    samples = waveform.squeeze().numpy().astype(np.float32)
    return samples, len(samples)


def make_input(name: str, data: np.ndarray, dtype: str):
    inp = grpcclient.InferInput(name, list(data.shape), dtype)
    inp.set_data_from_numpy(data)
    return inp


def text_input(name: str, text: str):
    """Encode UTF-8 text as a BYTES (STRING) tensor with shape [1].

    The model config declares `reference_text`/`target_text` as
    TYPE_STRING [1]; the python backend decodes the single element.
    """
    encoded = np.array([text.encode("utf-8")], dtype="S")
    inp = grpcclient.InferInput(name, [1], "BYTES")
    inp.set_data_from_numpy(encoded)
    return inp


# ─── Server statistics ─────────────────────────────────────────────────────────

def write_triton_stats(stats: dict, out_path: str):
    with open(out_path, "w") as f:
        f.write(
            "Triton inference statistics (triton_client.get_inference_statistics())\n"
            "Ref: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md\n"
            "     https://github.com/triton-inference-server/server/issues/5374\n\n"
        )
        for ms in stats.get("model_stats", []):
            if "last_inference" not in ms:
                continue
            ist   = ms["inference_stats"]
            q_s   = int(ist["queue"]["ns"])          / 1e9
            ci_s  = int(ist["compute_infer"]["ns"])  / 1e9
            cin_s = int(ist["compute_input"]["ns"])  / 1e9
            co_s  = int(ist["compute_output"]["ns"]) / 1e9
            f.write(f"model: {ms['name']}\n")
            f.write(
                f"  queue {q_s:.2f}s  compute_infer {ci_s:.2f}s  "
                f"compute_input {cin_s:.2f}s  compute_output {co_s:.2f}s\n"
            )
            for batch in ms.get("batch_stats", []):
                bs   = int(batch["batch_size"])
                cnt  = int(batch["compute_infer"]["count"])
                t_ms = int(batch["compute_infer"]["ns"]) / 1e6
                avg  = t_ms / cnt if cnt else 0.0
                f.write(
                    f"  batch_size={bs:<3} count={cnt:<6} "
                    f"total_infer={t_ms:.1f}ms  avg_infer={avg:.2f}ms\n"
                )
            f.write("\n")


# ─── Argument parser ───────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description=(
            "gRPC benchmark client for the F5-TTS Triton model on Cloud AI 100. "
            "Runs the same prompt N times and reports per-request latency and RTF."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Connection ─────────────────────────────────────────────────────────────
    p.add_argument("--server-addr", default="localhost",
                   help="Triton gRPC server address")
    p.add_argument("--server-port", type=int, default=8001,
                   help="Triton gRPC port")
    p.add_argument("--ssl",         action="store_true",
                   help="Use SSL for the gRPC connection")
    p.add_argument("--model-name",  default="f5_tts",
                   help="Triton model name")

    # ── Prompt ─────────────────────────────────────────────────────────────────
    p.add_argument("--ref-audio",   required=True,
                   help="Reference speaker audio file")
    p.add_argument("--ref-text",    required=True,
                   help="Transcription of the reference audio")
    p.add_argument("--target-text", required=True,
                   help="Text to synthesise in the reference speaker's voice")
    p.add_argument("--output",      default="generated.wav",
                   help="Output WAV path for the last generated audio")

    # ── Repetition ─────────────────────────────────────────────────────────────
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat inference N times")
    p.add_argument("--warmup", action="store_true",
                   help="Run one warmup inference before the timed benchmark")

    # ── Output & logging ───────────────────────────────────────────────────────
    p.add_argument("--log-dir",      default="./benchmark_results",
                   help="Directory for generated audio files and statistics")
    p.add_argument("--sample-rate",  type=int, default=24000,
                   help="Expected output audio sample rate in Hz")

    return p.parse_args()


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main():
    args = get_args()
    url  = f"{args.server_addr}:{args.server_port}"

    print(f"Connecting to Triton gRPC server at {url} …")
    client = grpcclient.InferenceServerClient(url=url, verbose=False, ssl=args.ssl)

    if not await client.is_server_live():
        raise RuntimeError(f"Triton server at {url} is not live")
    if not await client.is_model_ready(args.model_name):
        raise RuntimeError(f"Model '{args.model_name}' is not ready on the server")
    print(f"Server live.  Model '{args.model_name}' is ready.")

    os.makedirs(args.log_dir, exist_ok=True)

    # ── Load reference audio ───────────────────────────────────────────────────
    ref_audio, ref_audio_len = load_audio(args.ref_audio, target_sr=args.sample_rate)
    print(
        f"Reference: {ref_audio_len} samples "
        f"({ref_audio_len / args.sample_rate:.2f} s at {args.sample_rate} Hz)\n"
        f"  ref_text    : {args.ref_text!r}\n"
        f"  target_text : {args.target_text!r}"
    )

    def build_inputs():
        return [
            make_input("reference_wav",     ref_audio,                                  "FP32"),
            make_input("reference_wav_len", np.array([ref_audio_len], dtype=np.int32),  "INT32"),
            text_input("reference_text",    args.ref_text),
            text_input("target_text",       args.target_text),
        ]

    outputs = [grpcclient.InferRequestedOutput("waveform")]

    # ── Warmup ─────────────────────────────────────────────────────────────────
    if args.warmup:
        print("Warmup request …", end=" ", flush=True)
        try:
            await client.infer(args.model_name, build_inputs(), request_id="warmup-0", outputs=outputs)
            print("done")
        except Exception as exc:
            print(f"WARNING: warmup failed — {exc}")

    # ── Timed inference ────────────────────────────────────────────────────────
    n = max(args.repeat, 1)
    latency_list  = []
    rtf_list      = []
    total_audio_s = 0.0
    waveform      = None

    print(f"\nBenchmarking {n} request(s) …\n")
    wall_t0 = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        response = await client.infer(
            args.model_name, build_inputs(), request_id=str(100_000_000 + i), outputs=outputs
        )
        latency_s = time.perf_counter() - t0

        # The model returns an FP32 waveform in [-1, 1]; convert to int16 for the WAV.
        waveform = response.as_numpy("waveform").reshape(-1)
        waveform_int16 = (waveform * 32767.0).round().astype(np.int16)
        audio_dur_s = len(waveform_int16) / args.sample_rate

        out_wav = os.path.join(args.log_dir, f"generated_{i:03d}.wav")
        sf.write(out_wav, waveform_int16, args.sample_rate, "PCM_16")

        latency_list.append(latency_s)
        total_audio_s += audio_dur_s
        if audio_dur_s > 0:
            rtf_list.append(latency_s / audio_dur_s)

        print(
            f"  {i+1}/{n}  latency={latency_s*1e3:.0f} ms  "
            f"generated={audio_dur_s:.2f} s  RTF={latency_s/audio_dur_s:.3f}"
        )
    elapsed = time.perf_counter() - wall_t0

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    lat_arr = np.array(latency_list)
    rtf_arr = np.array(rtf_list)
    wall_rtf = elapsed / total_audio_s if total_audio_s > 0 else float("inf")

    sep = "=" * 60
    s  = f"\n{sep}\n"
    s += f"Benchmark summary — {n} request(s)\n"
    s += f"{sep}\n"
    s += f"total_audio_duration      : {total_audio_s:.3f} s  ({total_audio_s/3600:.4f} h)\n"
    s += f"wall_clock_time           : {elapsed:.3f} s  ({elapsed/3600:.4f} h)\n"
    s += f"wall_clock_RTF            : {wall_rtf:.4f}\n"
    s += f"\nPer-request latency (ms):\n"
    s += f"  average                 : {np.mean(lat_arr)*1e3:.2f}\n"
    s += f"  std                     : {np.std(lat_arr)*1e3:.2f}\n"
    s += f"  min                     : {np.min(lat_arr)*1e3:.2f}\n"
    s += f"  p50                     : {np.percentile(lat_arr, 50)*1e3:.2f}\n"
    s += f"  p90                     : {np.percentile(lat_arr, 90)*1e3:.2f}\n"
    s += f"  p95                     : {np.percentile(lat_arr, 95)*1e3:.2f}\n"
    s += f"  p99                     : {np.percentile(lat_arr, 99)*1e3:.2f}\n"
    s += f"  max                     : {np.max(lat_arr)*1e3:.2f}\n"
    if len(rtf_arr) > 0:
        s += f"\nPer-request RTF:\n"
        s += f"  average                 : {np.mean(rtf_arr):.4f}\n"
        s += f"  min                     : {np.min(rtf_arr):.4f}\n"
        s += f"  max                     : {np.max(rtf_arr):.4f}\n"
    s += f"{sep}\n"

    print(s)

    rtf_file = os.path.join(args.log_dir, "rtf-single.txt")
    with open(rtf_file, "w") as f:
        f.write(s)
    print(f"RTF summary saved     → {rtf_file}")

    # ── Persist server inference statistics ────────────────────────────────────
    try:
        stats      = await client.get_inference_statistics(model_name="", as_json=True)
        stats_file = os.path.join(args.log_dir, "stats_summary-single.txt")
        write_triton_stats(stats, stats_file)
        print(f"Server stats saved    → {stats_file}")
    except Exception as exc:
        print(f"WARNING: could not retrieve server statistics: {exc}")

    # ── Persist model configuration ────────────────────────────────────────────
    try:
        metadata  = await client.get_model_config(model_name=args.model_name, as_json=True)
        cfg_file  = os.path.join(args.log_dir, "model_config-single.json")
        with open(cfg_file, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Model config saved    → {cfg_file}")
    except Exception as exc:
        print(f"WARNING: could not retrieve model config: {exc}")

    # ── Copy last generated audio to --output ──────────────────────────────────
    if waveform is not None and args.output:
        src = os.path.join(args.log_dir, f"generated_{n-1:03d}.wav")
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(args.output):
            shutil.copy2(src, args.output)
            print(f"Last generated audio  → {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
