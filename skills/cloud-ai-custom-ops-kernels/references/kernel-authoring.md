# Custom Op Kernel Authoring

## Contents

- Kernel implementation and context
- Thread parallelism and synchronization
- Kernel constraints and device logging
- Op-design and threading best practices

## Kernel implementation (`myop_aic.cpp`)

### Minimal skeleton — float32

```cpp
#include "CustomOpAICInterface.h"

extern "C" {
void MyOpAIC(const CustomOpContext *ctx, const int32_t threadId) {
    const float  *src    = (const float *)ctx->inputs[0].data;
    float        *dst    = (float *)ctx->outputs[0].data;
    const int32_t rows   = ctx->inputs[0].sizes[0];
    const int32_t stride = ctx->inputs[0].sizes[1];
    const int32_t width  = *(int32_t *)ctx->params[0].data;

    for (int32_t y = threadId; y < rows; y += 4) {
        for (int32_t x = 0; x < width; x++) {
            dst[y * stride + x] = src[y * stride + x];  // passthrough
        }
    }
}
} // extern "C"
```

### Minimal skeleton — INT8 index type

```cpp
#include "CustomOpAICInterface.h"

extern "C" {
void MyOpAIC(const CustomOpContext *ctx, const int32_t threadId) {
    const uint8_t *src    = (const uint8_t *)ctx->inputs[0].data;
    uint8_t       *dst    = (uint8_t *)ctx->outputs[0].data;
    const int32_t  rows   = ctx->inputs[0].sizes[0];
    const int32_t  stride = ctx->inputs[0].sizes[1];
    const int32_t  width  = *(int32_t *)ctx->params[0].data;

    for (int32_t y = threadId; y < rows; y += 4) {
        for (int32_t x = 0; x < width; x++) {
            dst[y * stride + x] = src[y * stride + x];  // passthrough
        }
    }
}
} // extern "C"
```

### `CustomOpContext` fields

```cpp
ctx->inputs[i].data        // pointer to input tensor data
ctx->inputs[i].sizes[d]    // size of dimension d
ctx->outputs[i].data       // pointer to output tensor data
ctx->params[i].data        // pointer to parameter value (cast to int32_t* or float*)
ctx->scratchBuffers[i].buf // scratch memory (if requested in yaml)
ctx->scratchBuffers[i].size
ctx->syncThread(threadId)  // barrier: wait for all 4 threads
```

---

## Thread parallelism

The NSP provides **4 HVX threads** (threadId 0–3) to custom operator kernels. (Note: Hexagon v68 hardware provides 6 threads total; the remaining 2 are reserved for HMX and synchronization. See the `cloud-ai-architecture` skill's Hexagon v68 DSP Architecture reference for the hardware context.)

The kernel is invoked once per thread simultaneously. Distribute work with a row-stride pattern:

```cpp
// Each thread handles rows: threadId, threadId+4, threadId+8, ...
for (int32_t y = threadId; y < total_rows; y += 4) {
    // process row y
}
```

Use a row-stride partition so each thread handles rows `threadId`, `threadId+4`,
`threadId+8`, and so on. This gives each thread a contiguous stripe of rows,
maximizes VTCM locality, and keeps thread ucycles balanced.

Use `syncThread` only when one thread must complete a shared write before
others read. A barrier stalls all four threads; do not add barriers between
independent passes.

```cpp
// Thread 0 writes status; all threads must see it before proceeding
if (threadId == 0) {
    float *status = (float *)ctx->outputs[1].data;
    *status = validate() ? 0.0f : 1.0f;
}
ctx->syncThread(threadId);   // barrier — all threads wait here

const float *status = (const float *)ctx->outputs[1].data;
if (*status != 0.0f) return;
// ... rest of kernel
```

**Do not create threads** — the 4 threads are provided by the runtime. The kernel
cannot call `pthread_create` or any OS service.

---

## Kernel constraints

The AIC backend runs on bare metal — no OS services:

- **No dynamic allocation** — no `malloc`, `new`, `std::vector`, or VLAs.
  Use fixed-size stack arrays or scratch buffers.
- **No file I/O** — no `fopen`, `printf` (use `AIC_PRINT_*` macros instead).
- **No new threads** — 4 threads are provided; do not create more.
- **No exceptions** — C++ exceptions are not supported.
- **Static arrays only** — array sizes must be compile-time constants.

### Logging on device

```cpp
#include "CustomOpLog.h"
AIC_PRINT_INFO(ctx, "y=%d width=%d", y, width);
AIC_PRINT_DEBUG(ctx, "value: %f", some_float);
```

Requires `qaic-monitor-service` running on host:
```bash
sudo /opt/qti-aic/scripts/qaic-monitor-service.sh start
```

---

### Op design

**Dispatch at the largest practical granularity.** PCIe adds fixed overhead to
each invocation. Dispatch once per invocation or batch rather than once per row
or small tile. Choose offload only when the kernel work is large enough to
amortize transfer and dispatch overhead, and measure the break-even point for
the target workload.

**Separate side-channel metadata from primary tensor data.** For ops that need
per-region metadata (strengths, offsets, flags), pass it as a separate `params`
tensor rather than embedding it in the primary tensor. This keeps the primary
tensors contiguous and VTCM-friendly, and lets the metadata live in
`CacheableDDR` (read-once, no VTCM benefit).

**Enforce shape contracts in `customOpVerify`.**  Check rank, dtype, and
dimension bounds at compile time. A clear compile-time error is far easier
to debug than a silent wrong result at runtime. Reject inputs that exceed the
maximum tensor shape the kernel was designed for.

**Keep `status` as `float32[1]`.**  The compiler does not support `int32`
as a custom op output dtype.  Use `float32` for the status output and write
`0.0f` (ok) or `1.0f` (error).
