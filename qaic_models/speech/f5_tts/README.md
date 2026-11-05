# F5-TTS on Cloud AI 100 (`qaic_models` recipe)

F5-TTS is a diffusion-based text-to-speech model. Given a short reference audio
clip, its transcript, and new text to speak, it synthesizes speech in the
reference speaker's voice. Upstream implementation:
https://github.com/SWivid/F5-TTS/

This recipe runs the `F5TTS_v1_Base` variant on Cloud AI 100.

## What this runs

F5-TTS is exported as three ONNX stages:

| Stage | Runs on | Purpose |
|---|---|---|
| `F5_Preprocess` | Host CPU (ONNX Runtime) | Mel-spectrogram + rotary embedding + text-conditioning prep |
| `F5_Transformer` | Cloud AI 100 | Diffusion denoising loop (16 steps) |
| `F5_Decode` | Cloud AI 100 | Vocos vocoder: mel features → magnitude/phase for ISTFT |

- **Input**: a reference audio file, its transcript (`ref_text`, required —
  no ASR auto-transcription), and text to synthesize (`gen_text`).
- **Output**: a mono 24 kHz `int16` waveform, optionally written to a WAV file.
- **Accelerator target**: one AI 100 Ultra card (transformer: 4 QIDs tensor-sliced, 12
  cores/QID; decode: single QID, 4 cores).

## Generated files

Running `export.py` against a `--work-dir` produces:

```
<work-dir>/
  F5_Preprocess.onnx
  F5_Transformer.onnx
  F5_Transformer_fixed_for_fp16.onnx     # FP16-clipped, compiled from this one
  F5_Decode.onnx
  mdp_ts4.json                            # generated MDP partition config (if --compile)
  f5_transformer_duration_specializations.json
  qpc_transformer_12c/programqpc.bin       # if --compile
  qpc_decode_4c/programqpc.bin             # if --compile
```

Pass `<work-dir>` as `--model` to `demo.py` or as `model_dir` to `F5TTS(...)`.

## Environment setup

Requires Python 3.12, the QAIC SDK installed under `/opt/qti-aic`, and Cloud AI
100 devices for the QAIC inference/compile steps.

Create a Python environment in a location outside the recipe source tree:

```bash
python3.12 -m venv env_f5_tts
source env_f5_tts/bin/activate
python -m pip install --upgrade pip
```

Install the package extra (export/runtime dependencies):

```bash
pip install -e "/path/to/cloud-ai-sdk-root[f5-tts]"          # export + compile + runtime
# or, to also generate a Triton model repository from export.py:
pip install -e "/path/to/cloud-ai-sdk-root[f5-tts,triton]"
# or, for a deployment that only runs inference (no export/compile):
pip install -e "/path/to/cloud-ai-sdk-root[f5-tts-infer]"
```

F5-TTS and Vocos are not part of the package extra — install them from
pinned checkouts with `--no-deps` to avoid pulling in their unpinned
training dependencies:

```bash
mkdir -p src
git clone https://github.com/charactr-platform/vocos.git src/vocos
git -C src/vocos checkout 11b4628853c9e79dbc4577b01ecb02cfbc78255d
pip install -e src/vocos --no-deps

git clone https://github.com/SWivid/F5-TTS.git src/F5-TTS
git -C src/F5-TTS checkout 77d3ec623b861ea3cebd3a11f0f9a14489112464
pip install -e src/F5-TTS --no-deps
```

Download the F5-TTS and Vocos checkpoints (used by `export.py`):

```bash
huggingface-cli download SWivid/F5-TTS \
  --revision 84e5a410d9cead4de2f847e7c9369a6440bdfaca \
  --cache-dir ./ckpts

huggingface-cli download charactr/vocos-mel-24khz \
  --revision 0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21 \
  --cache-dir ./ckpts
```

Quick check:

```bash
python3 -c "import qaic_models.speech.f5_tts; print('ok')"
```

## Export and compile

```bash
python3 -m qaic_models.speech.f5_tts.export \
  --work-dir build/f5_tts \
  --f5-tts-src ./src/F5-TTS/src \
  --ckpt-dir ./ckpts \
  --compile
```

- `--work-dir` (required): where generated ONNX/QPC files are written.
- `--f5-tts-src` (required): pinned F5-TTS checkout's `src/` directory (see
  "Environment setup").
- `--ckpt-dir` (optional): the Hugging Face cache root — the directory passed
  as `--cache-dir` above, containing `models--SWivid--F5-TTS/` directly;
  defaults to `<work-dir>/ckpts`. (Note `demo.py`'s `--ckpt-dir` is one level
  *above* this — see "Demo CLI".)
- `--transformer-cores` / `--decode-cores` (optional): default 12/4.
- `--compile`: run `qaic-compile` after export (omit for ONNX-only export).
- `--triton-model-repo` (optional): generate a Triton model repository at the
  given path (requires `--compile`). See "Triton serving".
- `--triton-model-count` (optional): instance count for the Triton python
  model; default 1.

Re-running `export.py --compile` removes and
rebuilds the QPC directories in `--work-dir`.

## Demo CLI

`sudo` is required for QAIC device access, or add your user to the `qaic` group:

```
sudo usermod -aG qaic $USER
newgrp qaic
```

Run from the directory used for "Environment setup" and "Export and compile":
with `--ckpt-dir` omitted, `vocab.txt` resolves to `./ckpts/models--…`
relative to the current directory, and `--f5-tts-src` defaults to
`./src/F5-TTS/src`. When passing `--ckpt-dir` explicitly, point it at the
*parent* of the HF cache directory (`vocab.txt` resolves as
`<ckpt-dir>/ckpts/models--…`) — one level above what `export.py --ckpt-dir`
expects, which is the cache root itself:

```bash
python3 -m qaic_models.speech.f5_tts.demo \
  --ref-audio ./src/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature." \
  --gen-text "I don't really care what you call me." \
  --model build/f5_tts \
  --out generated.wav
```

Other flags: `--ckpt-dir`/`--f5-tts-src` to override the checkpoint/source
checkout paths; `--transformer-device-ids`/`--decode-device-ids` (default
`0,1,2,3`/`0`) to place the QAIC sessions on different QIDs at runtime. The
 transformer QPC runs on 4 QIDs.

Example output:

```
sample_rate: 24000
selected_duration_bucket: 1536
waveform samples: 123456
transformer: mean=812.340ms p50=810.120ms p90=825.400ms p99=830.100ms
...
wrote: generated.wav
```

## Python API

```python
from qaic_models.speech.f5_tts import F5TTS

with F5TTS("build/f5_tts", provider="QAicExecutionProvider") as f5tts:
    result = f5tts.infer(
        ref_file="./src/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
        ref_text="Some call me nature, others call me mother nature.",
        gen_text="I don't really care what you call me.",
        file_wave="generated.wav",
        seed=102,
        warmup=1,
        runs=1,
    )

print(result.waveform.shape)
print(result.sample_rate)
print(result.selected_duration_bucket)
print(result.latencies)
```

The returned `InferResult` has `waveform` (mono `int16` numpy array),
`sample_rate` (24000), `output_path` (the `file_wave` path if one was written,
else `None`), `selected_duration_bucket` (`1536` or `2944`), and `latencies`
(a `dict[str, LatencyStats]` keyed by pipeline stage — `audio_load`, `stft`,
`onnx_preprocess`, `transformer`, `decode`, `istft`, `inference`, `e2e`, and
`write_audio` if `file_wave` was given).

## Triton serving

The recipe generates a Triton model repository with a single python-backend
model, `f5_tts`, which runs the full pipeline (CPU preprocess → QAIC
transformer diffusion loop → QAIC decode → host ISTFT) inside one Triton
instance. The generated repository is self-contained:
`export.py` copies `F5_Preprocess.onnx`, `qpc_transformer_<cores>c/`, and
`qpc_decode_<cores>c/` into `f5_tts/1/` alongside the model code.

Generate the repository (requires `--compile`, run from the recipe directory
so `build/f5_tts/triton-model-repo` lands where the container commands below
expect it).
Repository generation additionally needs the `triton` extra, so the
prerequisite install is:

```bash
pip install -e "/cloud-ai-sdk[f5-tts,triton]"
```

```bash
cd qaic_models/speech/f5_tts
python3 -m qaic_models.speech.f5_tts.export \
  --work-dir build/f5_tts \
  --f5-tts-src ./src/F5-TTS/src \
  --ckpt-dir ./ckpts \
  --compile \
  --triton-model-repo build/f5_tts/triton-model-repo
```

Start Triton with the generated repository mounted. Replace
`</path/to/cloud-ai-sdk root>` with this repository checkout path. The
container command installs the serving runtime and starts
`tritonserver --model-repository=...`:

```bash
docker run -it \
  --name qaic-f5-tts-triton \
  --shm-size=8g \
  --network host \
  --device /dev/accel/ \
  --mount type=bind,source=</path/to/cloud-ai-sdk root>,target=/cloud-ai-sdk \
  ghcr.io/qualcomm/cloud_ai_triton_server:1.23.2.0 \
  bash -lc 'pip3 install "/cloud-ai-sdk[f5-tts-serve]" && \
    /opt/tritonserver/bin/tritonserver \
      --model-repository=/cloud-ai-sdk/qaic_models/speech/f5_tts/build/f5_tts/triton-model-repo'
```

The server listens on HTTP `0.0.0.0:8000` and gRPC `0.0.0.0:8001` by default
(host network, so these are host addresses/ports). To run multiple serving
stacks side by side, add `--http-address/--http-port/--grpc-address/--grpc-port`
flags to the `tritonserver` invocation, e.g. `--http-port=8091
--grpc-port=8092`.

The python model reads `model_dir`, `vocab_path`, and `f5_tts_src` from
`config.pbtxt` parameters. They are written with the absolute paths used at
generation time (`model_dir` points at the model version directory
`f5_tts/1/` where the stage artifacts were copied); if those paths differ
inside the container (e.g., the SDK is mounted at `/cloud-ai-sdk`), edit the
values in `triton-model-repo/f5_tts/config.pbtxt` before starting the server. Pipeline knobs (`nfe_steps`, `cfg_strength`,
`sway_coefficient`, `seed`, `duration_buckets`, device ids) fall back to the
recipe defaults; pass them as `config.pbtxt` parameters to override.

Request the `f5_tts` model. Inputs: `reference_wav` (FP32, `[N]` mono 24 kHz
samples, any amplitude scale), `reference_wav_len` (INT32, `[1]`, valid sample
count), `reference_text` (STRING, `[1]`, transcript of the reference audio),
`target_text` (STRING, `[1]`, text to synthesize). Output: `waveform` (FP32,
`[M]`) in `[-1.0, 1.0]`; convert with `int16 = round(w * 32767)` to write a
WAV.

The values below match the sample clip from the "Demo CLI" section (the
reference transcript must actually describe the reference audio — F5-TTS uses
it as a conditioning input, so a placeholder transcript degrades the output):

```python
import numpy as np
import soundfile as sf
import tritonclient.http as triton_http

client = triton_http.InferenceServerClient(url="localhost:8000")
ref = sf.read(
    "src/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
    dtype="float32",
)[0].reshape(-1)

response = client.infer(
    "f5_tts",
    inputs=[
        triton_http.InferInput("reference_wav", ref.shape, "FP32").set_data_from_numpy(ref),
        triton_http.InferInput("reference_wav_len", [1], "INT32").set_data_from_numpy(np.array([len(ref)], dtype=np.int32)),
        triton_http.InferInput("reference_text", [1], "BYTES").set_data_from_numpy(
            np.array([b"Some call me nature, others call me mother nature."], dtype="S")
        ),
        triton_http.InferInput("target_text", [1], "BYTES").set_data_from_numpy(
            np.array([b"I don't really care what you call me."], dtype="S")
        ),
    ],
)
waveform = response.as_numpy("waveform").reshape(-1)
sf.write("generated_triton.wav", (waveform * 32767.0).round().astype(np.int16), 24000)
```

(The example uses the `InferInput`/`set_data_from_numpy` client API, which is
what the server image's tritonclient 2.41.1 provides; newer tritonclient
releases also accept the `InferenceInput` constructor form.)

Limitations: batch size 1, no streaming/chunked synthesis, and no dynamic
batching — one synthesis runs end-to-end per request.

### Benchmarking

`benchmark_grpc.py` runs the same reference clip + prompt N times against the
running `f5_tts` model and reports per-request latency (mean/std/min/p50/p90/
p95/p99/max) and Real-Time Factor (RTF = wall-clock / generated-audio
duration). It also saves the generated WAVs, an RTF summary, the server
inference statistics, and the model config under `--log-dir`.

Install the gRPC client dependencies on the host (or any machine that can
reach the gRPC port):

```bash
pip install "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cpu
pip install tritonclient[grpc] soundfile numpy
```

Run a single-utterance benchmark from the recipe directory (the default
`--ref-audio` path is relative to it). Point `--server-addr`/`--server-port`
at the running server (gRPC defaults to `localhost:8001`):

```bash
python3 benchmark_grpc.py \
    --server-addr localhost \
    --server-port   8001 \
    --ref-audio   src/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
    --ref-text    "Some call me nature, others call me mother nature." \
    --target-text "I don't really care what you call me." \
    --warmup \
    --repeat      5 \
    --log-dir     ./benchmark_results
```

The reference transcript must actually describe the reference audio — F5-TTS
uses it as a conditioning input, so a placeholder transcript degrades the
output.

## Input and output details

- `ref_file`: any audio format `pydub`/`ffmpeg` can decode; resampled internally
  to 24 kHz mono.
- `ref_text` / `gen_text`: plain text; mixed English/Chinese text is converted
  to pinyin internally for the Chinese portion.
- The transformer runs at one of the configured duration buckets (`1536` or
  `2944` frames by default), chosen by rounding the estimated combined
  reference+generated length up to the smallest bucket that fits (falling back
  to the largest bucket if the estimate exceeds it).

## Expected smoke result

Using the sample command in "Demo CLI" above with the default
`F5TTS_v1_Base` checkpoint: `generated.wav` is written, non-empty, mono,
24 kHz; `selected_duration_bucket` is `1536` or `2944`; `latencies["transformer"]`
and `latencies["decode"]` are non-zero.

## Troubleshooting

- **`ModuleNotFoundError: f5_tts` / `vocos`**: install the pinned upstream
  checkouts with `--no-deps` (see "Environment setup") — not part of the
  `f5-tts` extra.
- **`FileNotFoundError` from `F5TTSQaic`**: `--model` is missing
  `F5_Preprocess.onnx`, `qpc_transformer_12c/programqpc.bin`, or
  `qpc_decode_4c/programqpc.bin` — re-run `export.py --compile`.
- **`FileNotFoundError` for `vocab.txt`**: run `demo.py` from the same
  directory as export, or pass `--ckpt-dir` pointing at the *parent* of the
  cache directory (`vocab.txt` resolves as `<ckpt-dir>/ckpts/models--…`).
- **QAIC session load / device errors**: `--transformer-device-ids` count
  must match the transformer QPC's MDP device count (4) and must be contiguous
  on a single Ultra card
- **`qaic-compile` fails on `-mdp-load-partition-config`**: confirm 4 QAIC
  devices (one Ultra card) are visible via
  `/opt/qti-aic/tools/qaic-util -q | grep QID`.
- **No devices found at runtime**: check device status/reset with
  `/opt/qti-aic/tools/qaic-util -q` / `sudo /opt/qti-aic/tools/qaic-util -s`.
- **`ValueError: Unsupported provider`**: only `"QAicExecutionProvider"` is
  implemented
- **Text with Chinese characters mis-segments**: verify `jieba` and `pypinyin`
  are installed via the `f5-tts`/`f5-tts-infer` extra.
- **Triton python model fails to import `qaic_models` / `torch`**: the
  container's `pip3 install "/cloud-ai-sdk[f5-tts-serve]"` must succeed —
  check `docker logs qaic-f5-tts-triton` (the SDK checkout must be mounted
  at `/cloud-ai-sdk`, see "Triton serving").
- **Triton model errors about missing artifacts/vocab**: the
  `model_dir`/`vocab_path`/`f5_tts_src` parameters in `f5_tts/config.pbtxt`
  were written with generation-time paths; edit them to the in-container
  paths.
- **`--triton-model-repo` without `--compile` fails**: the repository embeds
  the compiled QPCs, so export must run with `--compile` in the same invocation.

## License / distribution

F5-TTS is MIT-licensed (upstream: https://github.com/SWivid/F5-TTS). Vocos is
MIT-licensed (upstream: https://github.com/charactr-platform/vocos). Checkpoint
weights are downloaded from the `SWivid/F5-TTS` and `charactr/vocos-mel-24khz`
Hugging Face Hub repositories at the pinned revisions above; review their
respective model card terms before redistributing generated checkpoints/QPCs.
