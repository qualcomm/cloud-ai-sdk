# Disaggregated vLLM Serving

Use `qaic_disagg` when prefill and decode or encode and decode need separate
device pools. The proxy presents one OpenAI-compatible endpoint.

## Topology

`xEyPzD` means `x` encode workers, `y` prefill-plus-decode workers, and `z`
the total prefill-plus-decode device count. Confirm the device ranges against
the available hardware before launching workers.

For scale context, `4E30PD` is four one-SoC encode workers and thirty two-SoC
prefill-plus-decode workers: 64 SoCs in total. Map that topology to the
available hardware before launch.

## Text PD example

```bash
MODEL_ID="replace-with-supported-text-model"

python3 -m qaic_disagg \
    --prefill-port 9800 \
    --decode-port 9900 \
    --port 8000 \
    --prefill-device-group 0,1,2,3 \
    --decode-device-group 4,5,6,7 \
    --model "$MODEL_ID" \
    --prefill-max-num-seqs 1 \
    --decode-max-num-seqs 1 \
    --prefill-long-prefill-token-threshold 256 \
    --max-model-len 8192 \
    --kv-handOff-port 5066 \
    --dtype bfloat16 \
    --kv-cache-dtype mxint8 \
    --prefill-override-qaic-config \
        '{"split_retained_state_io":true,"enable_chunking":true}' \
    --decode-override-qaic-config '{"retain_full_kv":true}'
```

The SDK 1.22 disaggregated parser expects JSON for each
`--*-override-qaic-config` value. Supply `node_precision_info` only when
required by the selected integration.

## E-PD example

```bash
MODEL_ID="replace-with-supported-multimodal-model"

python3 -m qaic_disagg \
    --encode-port 8000,8001 \
    --encode-device-group 0,1 \
    --decode-port 8002,8003 \
    --decode-device-group 2,3 \
    --port 8686 \
    --model "$MODEL_ID" \
    --proxy-workers 3 \
    --router-policy least_outstanding \
    --quantization mxfp6 --kv-cache-dtype mxint8 \
    --encode-max-num-seqs 1 --decode-max-num-seqs 8 \
    --limit-mm-per-prompt '{"image":1}' \
    --encode-override-qaic-config \
        '{"height":[512,1024],"width":[512,768],"num_frames":[1,4],"kv_offload":true}' \
    --decode-override-qaic-config \
        '{"height":[512,1024],"width":[512,768],"num_frames":[1,4],"kv_offload":true,
          "comp_ctx_lengths_prefill":[1024,8192,32768],
          "comp_ctx_lengths_decode":[1024,8192,32768]}'
```

Use one port and device range per worker. The device parser accepts
comma-separated IDs and ranges such as `0:4`. Replace the illustrative
resolution, frame, and context buckets with values supported by the target
model and workload. Confirm the supported flags with
`python3 -m qaic_disagg --help` in the installed image.

`--router-policy` accepts `round_robin` (the default) or
`least_outstanding`. Use `--build-grafana-json` once to generate Prometheus
and Grafana artifacts for the selected topology, then rerun it only when that
topology changes. Omit that flag from the steady-state launch command.

## Runtime checks

- Encode workers produce embeddings; decode workers run generation.
- KV cache transfer uses the configured handoff port. SDK 1.22.8.1 defaults
  to `5656`; set it explicitly when another service uses that port.
- The proxy serves `GET /docs`, `GET /openapi.json`, `GET /v1/models`,
  `POST /v1/completions`, `POST /v1/chat/completions`, and `GET /status`.
  `GET /metrics` forwards decode-worker metrics when metrics forwarding is
  configured. The proxy does not expose `GET /health`.
- A matched per-worker request count can include both encode and decode legs.
  For end-to-end throughput, use the proxy result or divide the matched
  per-worker count by two.
- Check the installed integration for CCL bucket limits and collision handling.
- Prefix caching is not supported for the prefill stage; verify multimodal
  and other feature combinations against the installed integration.
