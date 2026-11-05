# Compute Context Length (CCL)

Compute Context Length (CCL) is required for a long-context QPC. It compiles
one QPC with multiple prefill and decode context specializations. Include
smaller prefill and decode buckets so short requests avoid the largest bucket's
TTFT and decode throughput cost. The runtime starts at the fitting
specialization and advances as the prompt or KV-cache fill grows.

## Compatibility

Before a costly compile, inspect the model configuration and active QEfficient
implementation:

```bash
QEFF_PYTHON="${QEFF_PYTHON:-/opt/qeff-env/bin/python}"
QEFF_SRC=$("$QEFF_PYTHON" -c "import QEfficient, os; print(os.path.dirname(QEfficient.__file__))")
MODEL_TYPE=$("$QEFF_PYTHON" -c "import json; print(json.load(open('config.json'))['model_type'])")
```

Check the matching model override under
`$QEFF_SRC/transformers/models/$MODEL_TYPE/` when the model has non-standard
attention or `config.json` has `partial_rotary_factor`. A CCL compile error in
attention shape operations is another compatibility signal.

Do not patch the installed QEfficient package. If the architecture needs an
attention or export change, use `qeff-transform-authoring` or
`qeff-model-onboarding` in an upstream fork. If no extension is available,
compile separate fixed-context QPCs.

## Compile Inputs

```python
from QEfficient import QEFFAutoModelForCausalLM

NUM_CORES = <nsp_cores_per_soc>
DEVICE_IDS = [<qid>, ...]
PREFILL_BUCKETS = [<ctx_len>, ...]
DECODE_BUCKETS = [<ctx_len>, ...]

model = QEFFAutoModelForCausalLM.from_pretrained("<model-id>")
model.compile(
    num_cores=NUM_CORES,
    num_devices=len(DEVICE_IDS),
    batch_size=<batch_size>,
    prefill_seq_len=<prefill_seq_len>,
    comp_ctx_lengths_prefill=PREFILL_BUCKETS,
    comp_ctx_lengths_decode=DECODE_BUCKETS,
)
```

The lists enable CCL. Do not also pass `ctx_len`: QEfficient uses the largest
decode bucket. `qaic_config={"ccl_enabled": True}` can generate a doubling
progression; verify it includes smaller prefill and decode buckets suitable for
the deployment.

## Select Buckets

- Make the largest bucket cover the required maximum context. Add smaller
  buckets spanning the prompt lengths and KV-cache fills the deployment serves.
- Select prefill and decode buckets independently: use the TTFT measurements
  to place prefill buckets and decode-throughput measurements to place decode
  buckets.
- Use no more than five entries per list. Start with a doubling progression
  when measurements are unavailable, then refine the smaller boundaries with
  the deployment workload's measurements.
- Run `references/perf_sweep.py` with the target model, device configuration,
  prompt, and generation lengths. The runtime cannot force a bucket; to
  measure one, prefill enough tokens for the requested KV-cache fill.

Do not assume per-bucket parity with separately compiled fixed-context QPCs;
measure both when that comparison affects deployment selection.
