####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import argparse

from .model import F5TTS, F5_VOCAB_RELATIVE


def main() -> None:
    ap = argparse.ArgumentParser(description="Run F5-TTS inference on Cloud AI 100.")
    ap.add_argument("--ref-audio", required=True, help="Path to the reference audio used for voice cloning")
    ap.add_argument("--ref-text", required=True, help="Transcript of the reference audio")
    ap.add_argument("--gen-text", required=True, help="Text to synthesize")
    ap.add_argument(
        "--model",
        required=True,
        help="Work directory produced by export.py (contains F5_Preprocess.onnx and "
             "qpc_transformer_<cores>c/, qpc_decode_<cores>c/), or a direct path resolvable by F5TTSConfig",
    )
    ap.add_argument(
        "--ckpt-dir",
        default=None,
        help="Parent of the Hugging Face cache directory (vocab.txt resolves as "
             "<ckpt-dir>/ckpts/models--SWivid--F5-TTS/...). Note this is one level ABOVE "
             "export.py --ckpt-dir, which is the cache root itself (<ckpt-dir>/models--...). "
             "If omitted, vocab.txt resolves relative to the current directory as "
             "./ckpts/models--...",
    )
    ap.add_argument("--f5-tts-src", default=None, help="Path to the pinned F5-TTS source checkout's src/ directory (default: ./src/F5-TTS/src, relative to the current directory)")
    ap.add_argument(
        "--transformer-device-ids",
        default="0,1,2,3",
        help="Comma-separated QIDs for the transformer QAIC session (default: 0,1,2,3, matching TS4)",
    )
    ap.add_argument(
        "--decode-device-ids",
        default="0",
        help="Comma-separated QIDs for the decode QAIC session (default: 0)",
    )
    ap.add_argument(
        "--provider",
        default="QAicExecutionProvider",
        choices=["QAicExecutionProvider"],
        help="Only QAicExecutionProvider is supported by this recipe today",
    )
    ap.add_argument("--out", default=None, help="Optional output WAV path")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    vocab_path = f"{args.ckpt_dir}/{F5_VOCAB_RELATIVE}" if args.ckpt_dir else None

    with F5TTS(
        args.model,
        provider=args.provider,
        f5_tts_src=args.f5_tts_src,
        vocab_path=vocab_path,
        transformer_device_ids=args.transformer_device_ids,
        decode_device_ids=args.decode_device_ids,
    ) as f5tts:
        result = f5tts.infer(
            ref_file=args.ref_audio,
            ref_text=args.ref_text,
            gen_text=args.gen_text,
            file_wave=args.out,
            seed=args.seed,
            warmup=args.warmup,
            runs=args.runs,
        )

    print(f"sample_rate: {result.sample_rate}")
    print(f"selected_duration_bucket: {result.selected_duration_bucket}")
    print(f"waveform samples: {result.waveform.shape[0]}")
    for name, stats in result.latencies.items():
        print(f"{name}: mean={stats.mean_ms:.3f}ms p50={stats.p50_ms:.3f}ms p90={stats.p90_ms:.3f}ms p99={stats.p99_ms:.3f}ms")

    if args.out:
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
