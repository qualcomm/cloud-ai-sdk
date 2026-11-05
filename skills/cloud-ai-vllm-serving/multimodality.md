# Multimodal vLLM Serving

Use the installed vLLM and QEfficient integration as the compatibility source
for vision-language and audio models. Compile a single or dual QPC according
to the architecture and serving topology.

| Architecture | Single QPC | Dual QPC |
|---|---|---|
| Encoder and language graph are jointly supported | Verify in the installed integration | Verify in the installed integration |
| Separate vision and language graphs | Not applicable | Required |

Dual QPC serving uses:

```bash
export VLLM_QAIC_QPC_PATH="path/to/vision_qpc:path/to/language_qpc"
```

Dual QPC is recommended for production serving. Batch size greater than 1
requires Dual QPC or E-PD serving. Audio-model support, batching, and feature
compatibility depend on the QPC layout. Verify prefix caching, speculative
decoding, and LoRA support before enabling them together.

## Multi-resolution inputs

Compile resolution and frame-count buckets so one server can accept the
intended input shapes. Keep image or frame dimensions consistent within each
request. Different requests may use different compiled buckets.
At runtime, the integration selects the closest compatible compiled bucket.

```bash
--additional-config \
  '{"override_qaic_config":{
      "height":[512,1024],
      "width":[512,768],
      "num_frames":[1,4],
      "kv_offload":true,
      "comp_ctx_lengths_prefill":[1024,8192,32768],
      "comp_ctx_lengths_decode":[1024,8192,32768]
    }}'
```

Budget vision tokens within the available context length. If `vision_size` is
not set, QEfficient derives it from the configured resolution and frame-count
buckets. Derive patch size and processor limits from the installed model
processor or configuration. Let QEfficient generate model-specific
specialization metadata and inspect the resulting `specializations.json`; the
encoder QPC holds per-resolution grid and vision-size metadata. Replace the
illustrative resolution, frame, and context buckets with values supported by
the target model and workload.

For image processor limits, use the installed vLLM help and model processor
documentation. Set `--limit-mm-per-prompt` to the workload's maximum image or
frame count when the integration does not derive it from the compiled buckets.
