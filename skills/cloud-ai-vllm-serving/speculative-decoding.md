# Speculative Decoding

Speculative decoding (SpD) uses a draft model that proposes multiple next tokens before the target model verifies them. Accepted draft tokens reduce target decoding work, but gains depend on pair compatibility, draft speed, and acceptance rate.

For SDK 1.22, use the installed image as the source of truth. The most useful
references are `/opt/qti-aic/integrations/vllm/examples/offline_inference/qaic_spd.py`,
`qaic_ngram_suffix.py`,
`$VLLM_ROOT/v1/spec_decode/qaic_draft_model.py`, and
`$VLLM_ROOT/model_executor/model_loader/qaic.py`, where `$VLLM_ROOT` is the
installed package directory discovered in [serving-reference.md](serving-reference.md).
These files expose implementation details that are intentionally not repeated
in this short guide.

## Requirements

SpD requires AOT compilation: do not pass `--enforce-eager`. Synchronous
scheduling is required, so leave `--async-scheduling` disabled.

## Select a Compatible Draft Model

Do not pair arbitrary models for `method: draft_model`:

- Use the same tokenizer and token-ID mapping. vLLM checks that target and
  draft models have the same vocabulary size; equal sizes alone do not prove
  tokenizer compatibility.
- Use a compatible text-model architecture and prompt format. Both models
  must be supported by the installed QAIC/QEfficient integration and compile
  as separate QPCs.
- Choose a draft model that is smaller and faster than the target. The benefit
  depends on its acceptance rate and its compile/runtime cost; a poor pair can
  reduce end-to-end performance even when it loads successfully.
- Validate the pair with representative prompts and measure end-to-end
  latency and throughput. Treat the Llama pair in the example as illustrative,
  not as a universal pairing rule.

## Target/Draft Device Placement

Target and draft models compile as separate QPCs. `num_devices` applies to
each QPC, not to the combined target-plus-draft deployment. With no explicit
groups, let the automapper assign devices and verify the resulting placement
in the startup logs.

Pass the following objects as the value of `--additional-config`.

Use disjoint `device_group` lists to spread the models across separate device
pools and give each QPC its own core allocation:

```json
{
  "override_qaic_config": {
    "device_group": [0, 1, 2, 3],
    "num_cores": 16
  },
  "draft_override_qaic_config": {
    "device_group": [4, 5, 6, 7],
    "num_cores": 16
  }
}
```

Use identical `device_group` lists to place both QPCs on the same devices and
partition the NSP cores between them:

```json
{
  "override_qaic_config": {
    "device_group": [0, 1, 2, 3],
    "num_cores": 8
  },
  "draft_override_qaic_config": {
    "device_group": [0, 1, 2, 3],
    "num_cores": 8
  }
}
```

For shared devices, the sum of the target and draft core allocations must fit
the available NSP cores per SoC. The combined target+draft QPC footprints must
also fit in each device's available DDR. Validate the actual load result and
memory reported by QAIC-specific tooling for the selected QPCs; do not infer
this from generic vLLM memory controls. If core counts are omitted for a
shared placement, the integration chooses a split; set both values explicitly
when the placement must be deterministic.

For draft-model SpD on SDK 1.22.8.1, vLLM's CPU-side rejection
sampler used Triton JIT compilation on the first request. Here, Triton means
the kernel compiler/runtime used by vLLM, not Triton Inference Server; this
does not imply GPU execution. Cloud AI vLLM images omit C/C++ build tools.
Draft-model SpD's first-request Triton JIT requires a compiler, so always build
and use a derived image:

```dockerfile
# Dockerfile.spd
FROM ghcr.io/qualcomm/cloud_ai_inference_vllm:1.23.2.0

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
```

```bash
docker build -t cloud-ai-vllm-spd:1.23.2.0 -f Dockerfile.spd .
```

Use the derived image for draft-model SpD. Do not install the compiler in a
running serving container.

`num_speculative_tokens` controls how many tokens the draft proposes per
target verification step. It is a generic vLLM SpD setting, not a QAIC compile or device setting.
There is no universal optimum. The installed QAIC examples use `3` for
draft-model SpD and `5` for N-gram; treat those as starting points and
benchmark the end-to-end workload. Larger values can reduce target
verification steps when draft acceptance is high, but add draft work and can
hurt latency when acceptance is low.

## Draft-model SpD

```bash
--additional-config '{"override_qaic_config":{"num_devices":4}}' \
--speculative-config '{"method":"draft_model","model":"meta-llama/Llama-3.2-1B","num_speculative_tokens":3}'
```

Without `draft_override_qaic_config`, `override_qaic_config` applies to both
QPCs. With `num_devices: 4`, the target and draft QPC each use four devices,
so speculative decoding maps eight devices total.

Set `num_cores` only when an explicit core allocation is needed. Use a
power-of-two value such as `8` or `16`.

Use `draft_override_qaic_config` only when the draft QPC needs different
overrides. The draft does not merge it with `override_qaic_config`, so repeat
every required setting, including `num_devices`.

## N-gram Prompt Lookup

N-gram prompt lookup has no draft model and is useful when repeated prompt
tokens are likely:

```bash
--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5}'
```

## Supported Modes and Limits

The SDK 1.22 QAIC integration supports only `ngram`, `suffix`, and
`draft_model`. It does not support SpD with LoRA, multimodality, or on-device
sampling. The v1 QAIC engine disables prefix caching, so do not rely on it in
an SpD configuration.
