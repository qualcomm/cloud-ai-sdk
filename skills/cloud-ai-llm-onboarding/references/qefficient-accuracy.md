# QEfficient Accuracy

Use this reference when a QEff artifact regresses against the source model or
the selected upstream example. Keep the model revision, processor, static
shapes, prompts or inputs, and compile configuration fixed. Compare a
task-appropriate metric; autoregressive models require full autoregressive
decode validation, not only first-token or step-zero logits.

## Isolate QEff Precision Choices

Start from the closest QEfficient example and preserve its model-specific
compile inputs. Compile and measure a reference artifact, then change one
precision choice per candidate:

1. Test `mxfp6_matmul=False` against `True`. False disables MXFP6 weight
   compression and keeps the model's uncompressed compile precision, typically
   FP16 for this workflow.
2. Test `mxint8_kv_cache=False` against `True`. With `False`, QEfficient uses
   the model target dtype for retained KV tensors, typically FP16; this is
   independent of weight compression.
3. Preserve any example-provided `node_precision_info`. For a model class that
   supports generated NPI, such as Gemma 4, `node_precision_info=True` creates
   a model-specific file from the final ONNX. Other models, such as Gemma 3,
   may provide a matching NPI file in their example.

Do not reuse an NPI file across model revisions, exports, or shapes. Do not
build node names from the pre-export PyTorch graph. QEfficient applies its
transforms before the final ONNX is compiled.

Record the metric and exact precision settings for each candidate. If no
candidate preserves quality, inspect the selected QEfficient model implementation
and use the upstream contributor workflow for a model-specific transform or
precision fix.

## Measuring Perplexity of a QPC

Use the upstream `scripts/perplexity_computation/calculate_perplexity.py` from
an efficient-transformers checkout. It is NOT in the pip wheel or Cloud AI
container - copy it into a writable dir inside the container first. It needs
the `datasets` package (present in the Cloud AI container) and works for
`--model_type torch|onnx|qpc`; for QPC pass `--model_path` the compiled `qpc`
directory.

Trap 1 - `prompt_len` must equal the compiled `prefill_seq_len`, not the
documented default of 1. A QPC binds prefill width, so run e.g.
`--prompt_len 128` for a prefill-128 QPC. With any value > 1 the script then
crashes at the loss step with `Expected input batch_size (N) to match target
batch_size (N+1)`: an upstream off-by-one. Fix it in the local copy by changing
the post-prefill increment (first `cache_index += 1`, just before the first
decode input is fetched) to `cache_index += padded_len`. Leave the in-loop
increments at 1. The default of 1 only "works" on prefill-1 models, which
current compiles never are.

Trap 2 - wall-clock. The script decodes a full `ctx_len`-token window per
sample (~ctx_len / decode_tps per window), so pass `ctx_len` equal to the
compiled context, cap the run with `num_samples`, and report coverage
("16 of 238 windows") with the score.

Trap 3 - no sharding support. To parallelize across QIDs, patch
`WikiTextDataLoader` to slice windows `[offset : offset + num_samples]` behind
a `--sample_offset` flag, run one process per `--qpc_device_id` with a
per-shard `--log_file`, and merge by averaging every `avg_loss:` line across
shard logs; PPL = exp(mean loss). Streams on QIDs of the same card contend for
bandwidth, so expect sub-linear speedup, not exact N x.

Compare scores only against the same protocol (dataset split, window size,
prompt_len, token coverage); the card-reported PPL of the source checkpoint
uses different windows and pre-quantization precision.
