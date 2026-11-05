# HVX Vectorization for Custom Ops

## Contents

- HVX model, headers, and wrappers
- Intrinsic tables and arithmetic patterns
- Alignment, tails, and vectorization
- Performance considerations

## HVX vectorization

Hexagon v68 HVX vectors are 128 bytes. For detailed vector register
specifications, element types, and lane counts, see the `cloud-ai-architecture`
skill's Hexagon v68 Programming Reference.

Key points for custom ops:
- 128 bytes → 128 uint8, 64 int16, 32 float32 lanes per vector
- Use `HVX_VectorUB`/`HVX_VectorB` for byte operations, `HVX_VectorSF` for float32

### Header files

HVX intrinsics and types are accessed via these headers:

| Header | Contents |
|--------|----------|
| `QAicHexagonHVX.h` | `LoadHVX`, `StoreHVX`, `HVX_Vector_UN`, alignment helpers (SDK wrappers) |
| `hvx_hexagon_protos.h` | All `Q6_*` intrinsic declarations (primary reference) |

For the complete header reference including paths and detailed usage, see the
`cloud-ai-architecture` skill's Hexagon v68 Programming Reference.

In kernel source, include:

```cpp
#include "CustomOpAICInterface.h"
#include "jit/QAicHexagonHVX.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
```

### SDK load/store wrappers (`QAicHexagonHVX.h`)

These wrappers enforce alignment and are preferred over raw pointer casts:

| Wrapper | Signature | Notes |
|---------|-----------|-------|
| `LoadHVX(ptr)` | `HVX_Vector LoadHVX(const T* ptr)` | Aligned 128-byte load; asserts alignment in debug builds |
| `StoreHVX(ptr, v)` | `void StoreHVX(T* ptr, HVX_Vector v)` | Aligned 128-byte store |
| `StoreHVX(ptr, v, n)` | `void StoreHVX(T* ptr, HVX_Vector v, int n)` | Predicated store: writes only the first `n` bytes; safe for tail vectors |
| `*(HVX_Vector_UN*)(ptr)` | unaligned load/store | Use when stride is not a multiple of 128; emits `vmemu` instruction |

`HVX_Vector_UN` is `__attribute__((aligned(1)))` — assigning through it emits
an unaligned store (`vmemu`) rather than the aligned `vmem`. Use it for any
row pointer where `row_offset % 128 != 0` or whenever `stride % 128 != 0`.
Using `LoadHVX` or `StoreHVX` on an unaligned address reads from the aligned-down
address and can silently produce wrong data; use the unaligned type for all
loads and stores in that row.

### Intrinsics table

All `Q6_*` intrinsics are declared in `hvx_hexagon_protos.h`. For the complete
naming convention (`Q6_<dst-type>_<op>_<src-types>`) and type codes, see the
`cloud-ai-architecture` skill's Hexagon v68 Programming Reference.

Key type codes: `V`=vector, `W`=vector-pair (256 bytes), `Q`=predicate, `R`=scalar register, `b`=byte, `h`=halfword (16-bit), `ub`=unsigned byte, `uh`=unsigned halfword, `sf`=float32, `_sat`=saturating, `_rnd`=rounding.

#### Arithmetic — uint8 / int8

| Intrinsic | Operation | Useful for |
|-----------|-----------|------------|
| `Q6_Vub_vsub_VubVub_sat(Vu, Vv)` | `max(Vu[i] - Vv[i], 0)` per byte — saturating unsigned subtract | Subtract and clamp in one instruction; no separate `vmax` needed |
| `Q6_Vub_vadd_VubVub_sat(Vu, Vv)` | `min(Vu[i] + Vv[i], 255)` per byte — saturating unsigned add | Accumulating per-element counters or offsets without overflow |
| `Q6_Vub_vavg_VubVub_rnd(Vu, Vv)` | `(Vu[i] + Vv[i] + 1) >> 1` per byte — rounding average | Round-up average of two byte vectors; avoids 16-bit widen for simple blends |
| `Q6_Vub_vabsdiff_VubVub(Vu, Vv)` | `|Vu[i] - Vv[i]|` per byte — unsigned absolute difference | Threshold comparisons, SAD computation, activity detection |
| `Q6_Vub_vmax_VubVub(Vu, Vv)` | `max(Vu[i], Vv[i])` per byte | Element-wise max, clamp lower bound |
| `Q6_Vub_vmin_VubVub(Vu, Vv)` | `min(Vu[i], Vv[i])` per byte | Element-wise min, clamp upper bound |

#### Arithmetic — int16 (intermediate overflow-safe path)

When uint8 arithmetic would overflow (e.g. `a*4 + b - 2*c`), widen to int16
with `Q6_Wuh_vzxt_Vub`, operate in 16-bit, then pack back with `Q6_Vub_vsat_VhVh`.
For expressions such as `(q0 - p0) * 4 + (p1 - q1)`, the intermediate values
can overflow uint8; keep the arithmetic in 16-bit before packing back.

| Intrinsic | Operation | Useful for |
|-----------|-----------|------------|
| `Q6_Wuh_vzxt_Vub(Vu)` | Zero-extend 128 uint8 → 64 uint16 pairs (VectorPair) | Widen before arithmetic that can overflow uint8 |
| `Q6_Vh_vadd_VhVh(Va, Vb)` | 64-wide int16 add | Accumulating multi-term expressions |
| `Q6_Vh_vsub_VhVh(Va, Vb)` | 64-wide int16 subtract | Signed differences, negating a vector |
| `Q6_Vh_vasl_VhR(Va, R)` | 64-wide int16 arithmetic shift left by scalar | Multiply by power of 2 (e.g. `× 4` → `<< 2`) |
| `Q6_Vh_vasr_VhR(Va, R)` | 64-wide int16 arithmetic shift right by scalar | Divide by power of 2 with rounding |
| `Q6_Vh_vmax_VhVh(Va, Vb)` | 64-wide int16 max | `clip3` lower bound: `max(x, -limit)` |
| `Q6_Vh_vmin_VhVh(Va, Vb)` | 64-wide int16 min | `clip3` upper bound: `min(x, limit)` |
| `Q6_Vh_vsplat_R(R)` | Broadcast int16 scalar to all 64 lanes | Constant bias or threshold vectors |
| `Q6_Vub_vsat_VhVh(Vhi, Vlo)` | Pack two int16 vectors → uint8 with saturation | Final `clip1` (clamp to [0,255]) after 16-bit arithmetic |
| `Q6_Vb_vpack_VhVh_sat(Vhi, Vlo)` | Pack two int16 vectors → int8 with saturation | Pack signed 16-bit result to int8 with clamp |
| `Q6_V_lo_W(W)` | Extract low 128 bytes of a VectorPair | Access lower half after `vzxt` widen |
| `Q6_V_hi_W(W)` | Extract high 128 bytes of a VectorPair | Access upper half after `vzxt` widen |

#### Splat (broadcast scalar to all lanes)

| Intrinsic | Operation | Useful for |
|-----------|-----------|------------|
| `Q6_Vb_vsplat_R(R)` | Broadcast int8/uint8 scalar to all 128 byte lanes | Zero vector, constant mask or threshold |
| `Q6_Vh_vsplat_R(R)` | Broadcast int16 scalar to all 64 halfword lanes | Constant bias, rounding offset |
| `Q6_V_vsplat_R(R)` | Broadcast 32-bit scalar to all 32 float lanes (reinterprets bits) | Zero vector, constant scale factor |

#### Predicates and conditional select

| Intrinsic | Operation | Useful for |
|-----------|-----------|------------|
| `Q6_Q_vcmp_gt_VubVub(Va, Vb)` | `Va[i] > Vb[i]` unsigned byte → predicate | Threshold test: `threshold > abs_diff` is equivalent to `abs_diff < threshold` |
| `Q6_Q_vcmp_eq_VbVb(Va, Vb)` | `Va[i] == Vb[i]` byte → predicate | Detect sentinel values (e.g. flag bytes set to `0xff`) |
| `Q6_Q_and_QQ(Qa, Qb)` | Predicate AND | Combine multiple per-element conditions |
| `Q6_Q_and_QQn(Qa, Qb)` | Predicate AND NOT: `Qa & ~Qb` | Exclude a subset from an active mask |
| `Q6_V_vmux_QVV(Q, Va, Vb)` | `Q[i] ? Va[i] : Vb[i]` — predicated select | Apply result only to active lanes; blend transformed and original values |
| `Q6_Q_vsetq2_R(n)` | Predicate with first `n` bits set | Tail predicate for partial last vector when row width is not a multiple of 128 |

For a partial final vector, use `Q6_Q_vsetq2_R(TAIL_BYTES)` and
`Q6_V_vmux_QVV` to blend the filtered result with the original data before
storing. This prevents out-of-bounds writes into adjacent rows.

#### Float32

| Intrinsic | Operation | Useful for |
|-----------|-----------|------------|
| `Q6_Vsf_vsub_VsfVsf(Va, Vb)` | 32-wide float32 subtract | Per-element subtraction in float32 kernels |
| `Q6_Vsf_vmax_VsfVsf(Va, Vb)` | 32-wide float32 max | Clamp to zero: `max(x, 0.0f)`; ReLU activation |

### uint8 / int8 vectorization (index-type tensors)

When using INT8 index-type tensors, the kernel receives raw `uint8_t` bytes.
HVX processes 128 bytes per vector — 4× more lanes than float32.

**Key intrinsics** (from `hvx_hexagon_protos.h`, included via `jit/QAicHexagonHVX.h`):

```cpp
// Saturating unsigned subtract: max(Vu[i] - Vv[i], 0) per byte lane.
Q6_Vub_vsub_VubVub_sat(Vu, Vv)

// Unsigned byte max / min per lane
Q6_Vub_vmax_VubVub(Vu, Vv)
Q6_Vub_vmin_VubVub(Vu, Vv)

// Splat a byte value into all 128 lanes
Q6_Vb_vsplat_R(val)
```

**Pattern: build a constant vector once, apply in the inner loop**

For operations where a coefficient or mask vector is constant across a row
(or depends only on the row index), build it before the inner loop and reuse:

```cpp
// Build once per row (or once total if fully constant)
uint8_t coeff_arr[BYTES_PER_VEC] __attribute__((aligned(128)));
for (int i = 0; i < BYTES_PER_VEC; i++)
    coeff_arr[i] = compute_coeff(i, row_context);
const HVX_Vector vcoeff = LoadHVX(coeff_arr);

// Inner loop: one instruction per 128-byte vector
for (int v = 0; v < vecs_per_row; v++) {
    HVX_Vector values = LoadHVX(vsrc + v);
    HVX_Vector result = Q6_Vub_vsub_VubVub_sat(values, vcoeff);
    StoreHVX(vdst + v, result);
}
```

`Q6_Vub_vsub_VubVub_sat` performs `max(value - coeff, 0)` per lane in a single
instruction — the saturating subtract eliminates the need for a separate
clamp. Prefer saturating operations such as `Q6_Vub_vsat_VhVh` when packing
16-bit results back to uint8 so the clamp and pack happen together.

Stride-spaced edges require non-contiguous gathers. Compare gather cost with
the filter work and measure before vectorizing; contiguous regions are the
safer default for HVX optimization.

### float32 vectorization

```cpp
#include "CustomOpAICInterface.h"
#include "jit/QAicHexagonHVX.h"

// Load/store (ptr must be 128-byte aligned)
HVX_VectorSF v = LoadHVX(ptr);
StoreHVX(ptr, v);

// Arithmetic
HVX_VectorSF result  = values - vmask;                       // element-wise subtract
HVX_VectorSF clamped = Q6_Vsf_vmax_VsfVsf(result, v_zero);  // element-wise max
```

### Alignment requirements

- Buffer pointers from `ctx->inputs[i].data` and `ctx->outputs[i].data` are
  128-byte aligned when placed in `VTCM` or `CacheableDDR`.
- Stack arrays used as HVX operands must be declared with
  `__attribute__((aligned(128)))`; unaligned stack arrays can silently produce
  wrong device results because the alignment assertion is a no-op in device
  builds.
- For uint8: a stride that is a multiple of 128 bytes gives clean aligned
  vector loops; otherwise use the unaligned path and a predicated tail.
- For float32: a stride that is a multiple of 32 floats (128 bytes) gives
  clean aligned vector loops.

### Performance considerations

Use these trends as hypotheses, not guaranteed speedups; measure both device
cycles and end-to-end latency on representative tensors:

| Strategy | Likely effect | Main constraint |
|---|---|---|
| Scalar, one thread | Baseline implementation | Limited parallelism |
| Scalar, four threads | Can improve throughput when work is balanced | Memory bandwidth and synchronization |
| Float32 HVX | Processes 32 lanes per vector | Four-byte elements increase transfer volume |
| Scalar INT8 | Reduces transfer volume | Scalar compute can dominate |
| INT8 HVX | Processes 128 byte lanes per vector | Alignment, tails, and memory placement |

For end-to-end latency, compare host conversion, PCIe transfer, and NSP
execution separately. A smaller INT8 transfer and wider byte-level HVX vectors
can help, but the dominant component depends on tensor shape, placement, and
content. Use the profiling workflow in [Build, test, and profile](build-test-profile.md)
to measure the actual trade-off.
