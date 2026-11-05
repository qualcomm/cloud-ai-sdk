# Custom Op Configuration and Memory Placement

## Contents

- Package registration and configuration schema
- Memory tiers and placement decision guide
- Whole-tensor placement versus tiled VTCM staging
- Opstats placement signals
- `requiredFor`
- Memory best practices

## Configuration file (`custom_op_config.yaml`)

```yaml
version: 6.0.0
CustomOps:
  - type: MyOp           # must match ONNX node op_type
    package: QAic        # must match ONNX node domain
    inputs:
      - name: input_tensor
        maxDims: 2       # maximum tensor rank
    parameters:
      - name: width
        dataType: int    # int | float | bool
        scalar: true     # false = 1D array, requires size:
    outputs:
      - name: output_tensor
        maxDims: 2
      - name: status
        maxDims: 1
    functionsLibrary: ./MyOp/src/myop_lib.so
    implementations:
      - backend: Interpreter
        type: MyOpInterpreter
        impl: ./MyOp/src/myop_lib.so
      - backend: AIC
        type: MyOpAIC
        impl: ./MyOp/src/myop_aic.cpp
        memoryConfig:
          CacheableDDR: [input_tensor, output_tensor]
          VTCM: []
          requiredFor: []
```

### Memory placement

Buffer placement in `custom_op_config.yaml` controls where each named tensor
buffer lives on the NSP and can materially affect kernel performance. **The goal
is to keep the active working set in fast VTCM (Vector Tightly Coupled Memory)**
for data accessed in tight loops. The active working set can be a complete
tensor or only the current tile, plus its input/output overlap, scratch, and
metadata. Measure the effect with `qaic-opstats` on representative inputs.

For the architectural context and full memory hierarchy (VTCM, L2, DDR), see
the `cloud-ai-architecture` skill's Hexagon v68 DSP Architecture and SoC
Architecture references.

```yaml
memoryConfig:
  DDR:
  CacheableDDR: [status]
  VTCM: [input_tensor, output_tensor]
  requiredFor:
```

#### Memory tiers

| Tier | Capacity | Bandwidth | Best for |
|------|----------|-----------|----------|
| `VTCM` | ~8 MB per core | Highest — direct NSP SRAM | Working tensors the kernel reads/writes in a tight loop |
| `CacheableDDR` | Unlimited | Medium — DDR with L2 cache | Tensors too large for VTCM; scalar/random access patterns |
| `DDR` | Unlimited | Lowest — uncached DDR | Rarely needed; use `CacheableDDR` instead unless cache thrashing is a concern |

#### Decision guide

**Use whole-tensor `VTCM` placement when:**
- The tensor, together with every other resident tensor and the kernel's scratch
  state, fits within the ~8 MB per-core VTCM budget.
- The kernel accesses the tensor with a sequential or strided HVX vector loop — VTCM delivers full HVX bandwidth without cache misses.
- `qaic-opstats` shows `aiccopysamevtcm HVX CacheableDDR` entries — these are compiler-inserted DDR↔VTCM staging copies that disappear when the tensor is declared `VTCM` directly.

**Use tiled VTCM staging when the full tensor does not fit:**
- Keep the logical input or output tensor in `CacheableDDR` or `DDR`, and stage
  only bounded slices or tiles into the VTCM working buffers used by the kernel.
- Size the tile from the total concurrent working set: input tiles, output tiles,
  halo or overlap, weights, intermediates, metadata, alignment padding, and any
  double buffers must fit together within the per-core VTCM budget.
- Make the tile loop, copy/synchronization sequence, and edge handling explicit
  in the custom kernel or in an SDK-supported tiling mechanism. A tensor-level
  `memoryConfig: VTCM` entry does not by itself guarantee that an arbitrary
  custom-op kernel will be tiled automatically.
- Compare tiled and untiled versions with `qaic-opstats`; staging overhead can
  outweigh the locality benefit when tiles are too small or have little reuse.

**Use `CacheableDDR` when:**
- The full tensor exceeds VTCM capacity and the kernel does not have an explicit
  tiled-staging path for it.
- The tensor is accessed infrequently (e.g. a small `status` scalar) — cache overhead is negligible.
- Per-block metadata (`params`), scalar status outputs, and other read-once tensors do not benefit from VTCM; keeping them in `CacheableDDR` preserves VTCM for hot tensors.
- Multiple ops share the device and VTCM is already heavily allocated.

**Use `DDR` when:**
- You need to guarantee no cache pollution (rare).
- The tensor is write-once / read-once with no reuse — caching provides no benefit.

#### What `qaic-opstats` tells you

qaic-opstats profiling gives you valuable insights into how well the custom op is optimized.
See the `cloud-ai-performance-optimization` skill for tool usage instructions.

The `opdetails` column in the `qaic-opstats` summary directly reveals placement
decisions and their cost:

| `opdetails` entry | Meaning | Action |
|-------------------|---------|--------|
| `customop HVX CacheableDDR` | Kernel running with DDR-backed tensors | Move a hot tensor to `VTCM` if the whole working set fits, or inspect whether explicit tile staging can keep its active slices in VTCM |
| `customop HVX VTCM` | Kernel running with VTCM-backed tensors | Use as the placement baseline; confirm end-to-end impact with profiling |
| `aiccopysamevtcm HVX CacheableDDR` | Compiler staging DDR→VTCM before kernel | If the whole tensor fits, declare it `VTCM`; otherwise verify that the staging is the intended tile-level behavior before changing placement |
| `aiccopysamevtcm DMAIssue TCM` | PCIe input DMA landing in TCM | Unavoidable — this is the PCIe transfer itself |

Moving a hot working tensor from `CacheableDDR` to `VTCM` can reduce memory-
staging overhead when the whole working set fits. When it does not, reducing
the active tile and staging only the reused slices can provide the same locality
without requiring the full logical tensor to be resident. Quantify either change
with paired `qaic-opstats` runs on representative inputs; the benefit depends on
tile size, reuse, copy overhead, access pattern, contention, and the rest of the
graph.

Budget the complete concurrent VTCM working set across the op, including whole
resident tensors, active tiles, scratch, and double buffers. A workload with
separate input/output buffers for several tensors can approach the ~8 MB limit;
larger logical tensors can still work when their active slices are bounded, while
larger tiles or more buffers may require some data to remain in `CacheableDDR`.

#### `requiredFor`

Leave `requiredFor` empty to allow the compiler to fall back if the placement
constraint cannot be satisfied (e.g. VTCM is full).  Set it to `[AIC]` to make
the placement a hard requirement — compilation fails with a clear error rather
than silently falling back to a slower tier.
