# Audio and Generative Output Contracts

## Contents

- [Fixed-length output cropping](#fixed-length-output-cropping)
- [TTS predict.py CLI contract](#tts-predictpy-cli-contract)
- [ASR predict.py CLI contract](#asr-predictpy-cli-contract)
- [TTS shape-budgeting guardrails](#tts-shape-budgeting-guardrails)

## Fixed-length output cropping

A compiled QPC always returns its full output buffer — the compiled upper bound (max tokens, frames, samples) — whether or not generation actually stopped earlier. Output size cannot vary per request, so cropping is `predict.py`'s job.

**Fix:** track the real output length from whatever ended generation (EOS/stop-token position, stop condition, or a valid-length input already threaded through the pipeline) and slice the returned array to it. Otherwise the caller gets un-conditioned tail positions — repeats, silence, noise, or garbage — with no error raised. For continuous signals (audio, frames), fade at the cut point rather than truncating hard.

**Applies to:** LLM/seq2seq token buffers past EOS, diffusion/flow frame buffers, vocoder sample buffers, detection/segmentation query slots past the real object count.

**Detection signal:** output size barely varies with input content (e.g. every waveform is exactly the compiled maximum duration), or valid content ends in a sharp discontinuity at a fixed offset.

---

## TTS predict.py CLI Contract

The complete argument and stdout contract for text-to-speech predict.py. No other section adds arguments. Required predict.py behavior is not all here - the TTS shape-budgeting section adds valid-length passing, output cropping, and over-budget handling. Read both before declaring predict.py done.

| Argument | Required | Purpose |
|---|---|---|
| `--text <string>` | yes | Text to synthesize |
| `--output <path>` | yes | Write 16-bit PCM WAV here |
| `--voice-id <id>` | yes | Selects the reference voice; when omitted, fall back to the default voice registered in `reference_voices` |
| `--warmup-calls N` | yes for ProgramGroup multi-stage (default 0) | Discard N runs before the timed call; see the warmup-calls rule in the template-routing reference |

**Return exactly this JSON on stdout** — `prediction` must be this dict, not a status string:
```
{'prediction': {'sample_rate': int, 'duration_s': float, 'text': str}, 'latency_ms': float}
```
Include generated frame count and any other debugging metadata inside `prediction`.

**`--voice-id` is the only voice-selection input in this contract.** A caller selects a voice by ID alone and never passes model-specific reference-audio paths or flags, so predict.py owns the mapping from voice ID to whatever the model needs: a bundled WAV plus transcript, `--audio`/`--ref-text`, `--audio_prompt_path`, built-in conditioning baked into weights, or a speaker embedding index.

## ASR predict.py CLI Contract

For speech recognition, accept audio only through:

```bash
python3 predict.py --audio <path|url|base64>
```

Accept WAV, MP3, FLAC, and OGG. The script owns audio decoding, downmixing, resampling, and any model-specific preprocessing. It must emit its final stdout JSON object as:

```json
{"prediction":{"text":"<transcript>"},"latency_ms":123.4}
```

`prediction.text` must be the transcript string and `latency_ms` must cover the complete compiled inference request. Write diagnostics, warnings, and progress to stderr. Do not require `--text`, `--image`, or image-only `--output` arguments for ASR.

Write `demo_config.json` alongside the script:

```json
{"input_type":"audio"}
```

### Long-audio ASR with fixed-window QPCs

A fixed-window encoder QPC limits one **QPC call**, not the submitted audio clip. `predict.py` must transcribe the complete clip; never silently truncate it to one compiled window.

Keep the source model's long-audio splitter and transcript merge semantics. When source code is available, locate its chunker before writing one. For a fixed QPC window, retain its contiguous, no-overlap behavior but constrain every chosen boundary to the QPC cap or earlier. If the source searches for a low-energy/silence boundary around a nominal cut, search only in the trailing lookback inside that cap; fall back to the exact cap when no usable boundary exists. Do not add overlap or text deduplication unless the source implementation does.

Run every chunk through the same initialized QPC sessions/ProgramGroup, preserve chunk order, and merge text using the source model's rule. Keep `prediction.text` as the complete transcript; add `chunk_count`, `audio_duration_s`, and `chunk_window_s` inside `prediction`. `latency_ms` covers the complete clip.

Smoke-test with audio longer than one compiled window. Assert more than one chunk ran, no chunk exceeded the QPC window, and the final JSON contains the complete merged transcript. A one-window smoke test is insufficient.

### TTS voice selection

- Never make a model-specific reference-audio flag required. If `--audio`/`--ref-text`/`--audio_prompt_path` exist, they must stay optional overrides — a run with only `--text`, `--output`, and `--voice-id` must succeed.
- Every ID predict.py accepts must appear in `reference_voices` in demo_config.json, and every ID listed there must work. Exit non-zero with a clear stderr message on an unknown ID.
- The voice predict.py uses when `--voice-id` is omitted must be the same voice registered as the default in `reference_voices`.

---

## TTS Shape-Budgeting Guardrails

For text-to-speech, fixed QPC shapes must budget **both** conditioning text and audio duration:

- Text budget: `TEXT_LEN` covers `ref_text + gen_text` after the model's tokenizer. If the reference transcript alone consumes most of `TEXT_LEN`, generated speech will ignore or truncate the user's input.
- Duration budget: `SEQ_LEN`/`max_duration` covers `reference_mel_frames + generated_mel_frames`. If the reference audio consumes most of the frame budget, every output will be capped to a short clip.
- Keep the reference-audio mel budget and reference transcript aligned. Do not truncate reference audio to 1-2 seconds while still conditioning on a full-sentence transcript.
- Validate compiled QPCs **on device** with realistic inference-shaped inputs: actual reference mel, `ref_text + gen_text`, and zero padding to the fixed shape. CPU ONNX cosine on dense random tensors is not enough; large fixed shapes can compile successfully yet produce input-independent saturated output on device.
- If the exported QPC takes a per-request valid-length input (`input_lengths` / `attention_mask` — see `cloud-ai-model-conversion/references/qaic-model-requirements.md`, "Valid-Length Inputs for Padded Variable-Length Dimensions"), predict.py MUST pass the true length (reference + generated frames), not the padded maximum. Passing the full fixed length makes the model treat padding as real audio and append filler/repeats past the intended end.
- The vocoder output is a fixed-length buffer with the same problem on the output side — see "Fixed-Length Output Cropping" above. Crop the returned waveform to the real generated duration.
- If a large shape produces all-same or saturated values (for example FP16 max/min) for different inputs, bisect shape size and use the largest shape that is input-dependent and non-saturating on device.
- If the text→speech-token stage is autoregressive, look for a hard-coded `max_new_tokens` or similarly named generation cap and adjust it to the QPC shape boundaries through the model adapter.

**TTS smoke test before declaring demo-ready:**
1. Run `predict.py --text "<distinct sentence>" --output out.wav` through the compiled QPCs, not CPU fallback.
2. Confirm the JSON echoes the requested `text`, reports `sample_rate`, `duration_s`, and generated frame count/metadata useful for debugging.
3. Inspect the WAV: correct sample rate, duration matches the real generated length (not always the compiled maximum), non-zero RMS, finite samples, and no full-scale clipping or FP16-saturation-derived garbage.
4. Run a second, different prompt and confirm the generated mel/audio changes; identical output across prompts means the compiled QPC or host loop is not using inputs correctly.
5. If the source pipeline supports splitting across bounded invocations, run and assemble the complete request through that source-controlled flow; a QPC bound is not a whole-request limit. If `ref_text + gen_text` exceeds `TEXT_LEN` or `reference_mel_frames + generated_mel_frames` exceeds `SEQ_LEN` and no source-controlled split covers it, exit non-zero with a clear stderr message — do not return a `prediction` result for a request this pipeline cannot serve. Never silently clamp or truncate the request.
6. If the text→speech-token stage is autoregressive, run two more smoke tests: (a) a prompt long enough that unclamped generation would exceed the compiled decode budget (several sentences), and (b) a short prompt containing digits, symbols, or addresses/IDs (IP addresses, currency amounts, version numbers) likely to be out-of-distribution for the tokenizer. Confirm `predict.py` either chunks the request or clamps the decode step count (see the `max_new_tokens`/`max_length`/`max_steps` guidance above) and still returns valid audio for both — it must not crash with a KV-cache/decode-position assertion. Length alone does not predict this failure: a short, digit/symbol-heavy prompt can fail to emit an end-of-generation token and run away in step count just as an overlong prompt can. A long-prompt-only smoke test will not catch this failure mode.

---
