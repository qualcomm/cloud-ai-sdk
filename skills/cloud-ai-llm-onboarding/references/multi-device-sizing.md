# Multi-Device Tensor-Slicing Sizing

Tensor slicing (TS) partitions a QPC across AI100 Ultra SoCs. Start with the
smallest TS level that fits weights, KV cache, and activation headroom. Increase
TS only when measured decode throughput needs more compute.

`mdp_ts_num_devices` counts SoCs (QIDs), not cards. An Ultra card has four
SoCs, so TS4 uses one full card, TS8 uses two cards, and TS16 uses four cards.
TS16 is the maximum supported configuration for one QPC.

Choose from 1, 2, 4, 8, or 16 SoCs (TS1, TS2, TS4, TS8, or TS16). Start with
the smallest level that fits, then move to the next level only when measured
decode throughput requires it.

## Fit Before Throughput

For MXFP6 dense weights, use roughly 0.75 bytes per parameter as a first memory
estimate. Add KV cache and activation headroom, then verify the result with the
QEfficient and compiler memory reports. Do not size from parameter count alone.

| Dense model size | Initial TS range | Typical next step |
|---|---|---|
| Up to 8B | TS1 | Increase toward TS4 only for decode throughput |
| 8B-32B | TS2-TS4 | Measure TS4-TS8 if decode is insufficient |
| 32B-70B | TS4-TS8 | Measure TS8-TS16 if decode is insufficient |
| 70B-200B | TS8-TS16 | Verify fit and keep at or below TS16 |

MoE memory sizing uses total parameters: all experts must fit even though each
token activates only a subset. Verify the actual model with memory reports;
do not infer fit from active parameters.

## QEfficient Configuration

```json
"compilation": {
  "mdp_ts_num_devices": 8,
  "aic_num_cores": 16
}
```

QEfficient generates the partition configuration from `mdp_ts_num_devices`. Add
`mdp_dump_partition_config` when the generated layout needs inspection.

## Multi-Module Pipelines

Each module receives its own QPC and TS allocation. Loaded QPCs do not share
NSPs, so size a concurrently loaded pipeline as:

```text
total_SoCs = sum(ts_per_module)
```

For architecture and memory-hierarchy context, use `cloud-ai-architecture`. For
partition-config mechanics, use `cloud-ai-qaic-compile`.
