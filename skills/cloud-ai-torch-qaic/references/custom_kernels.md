# torch_qaic – Custom Hexagon NSP Kernels

Full reference for writing, building, and dispatching JIT custom kernels via
`torch_qaic.custom_ops`. For HVX vectorization and kernel implementation best practices,
see `cloud-ai-architecture`.

---

## Kernel entry point

```c
#include "QAicHexagonUtils.h"  // umbrella: types, HVX ops, qshim runtime, error codes

// args->pointers[0] = __fp16*  input         (fp16 tensor)
// args->pointers[1] = __fp16*  output        (fp16 tensor)
// args->pointers[2] = int32_t* num_elements  (1-element int32 tensor)
extern "C" uint32_t my_kernel(
    const AicJitEntryPointConfig *cfg,
    const AicJitPointerArray     *args)
{
    const __fp16 *input  = static_cast<const __fp16 *>(args->pointers[0]);
          __fp16 *output = static_cast<__fp16 *>(args->pointers[1]);
    uint32_t      n      = *static_cast<const int32_t *>(args->pointers[2]);

    uint32_t total = cfg->numCores * cfg->numThreads;
    uint32_t id    = cfg->coreID * cfg->numThreads + cfg->threadID;
    uint32_t chunk = (n + total - 1) / total;
    uint32_t start = id * chunk, end = start + chunk < n ? start + chunk : n;

    for (uint32_t i = start; i < end; ++i)
        output[i] = input[i] * (__fp16)2.0f;

    return 0;  // 0 = success; use QAIC_OPERATOR_IMPL_ERROR for errors
}
```

---

## Work partitioning across NSPs

The kernel is invoked simultaneously on every dispatched NSP — one instance per core per
thread. Each instance receives its position via `cfg` and independently processes its
slice of the work.

**Standard flat-worker-ID pattern:**

```c
// Total workers = numCores × numThreads (e.g. 16 × 4 = 64)
uint32_t total = cfg->numCores * cfg->numThreads;
uint32_t id    = cfg->coreID * cfg->numThreads + cfg->threadID;  // 0..total-1

uint32_t chunk = (n + total - 1) / total;
uint32_t start = id * chunk;
uint32_t end   = start + chunk < n ? start + chunk : n;
```

See `examples/scale_fp16/scale_kernel.cpp` for a working example.

**When workers need to synchronize or exchange data:**

| Need | API |
|---|---|
| Barrier across threads within one core | `qshimSyncThreads` |
| Barrier across all NSPs | `qshimSyncNSPs` |
| Broadcast a result from one NSP to others (e.g. after a reduction) | `qshimMulticastXfer` / `qshimWaitOnMulticast` |

Threads within a core share VTCM (via `qshimGetBaseVtcmAddr`) and can exchange data
through scratch memory after a `qshimSyncThreads` barrier. NSPs do not share VTCM —
cross-NSP data exchange requires multicast or a DDR staging buffer.

The kernel code is identical regardless of `[nsp_count, thread_count]` — `numCores` and
`numThreads` change at dispatch time and the partitioning math adjusts automatically.

---

## JIT include headers (`/opt/qti-aic/dev/inc/jit`)

| Header | Provides |
|---|---|
| `QAicHexagonUtils.h` | Umbrella include |
| `QAicHexagonTypes.h` | `HVX_VectorSF`, `HVX_VectorHF`, `HVX_VectorH`, `HVX_VectorB`, etc. with operator overloads; `HVX_VectorSize = 128` |
| `QAicHexagonHVX.h` | `LoadHVX`, `alignTo`, `divideCeil`, `ASSERT_VECALIGN` |
| `jit_dev_exe_function.h` | `AicJitEntryPointConfig`, `AicJitPointerArray` |
| `jit_qshim_api.h` | NSP runtime services (see below) |
| `QAicPlatformIntf.h` | `qaicGetSystemTimestampInUstmr()`, `qaicGetPcycleCount()` |
| `jit_dev_status_codes.h` | Error codes |

---

## NSP runtime services (`jit_qshim_api.h`)

| API | Purpose |
|---|---|
| `qshimLinearUdmaSubmit` / `qshim2DUdmaSubmit` | Async uDMA transfers (1D / 2D strided) |
| `qshimUDmaWait` | Wait for uDMA handle |
| `qshimGetBaseVtcmAddr` / `qshimGetVtcmSize` | VTCM scratch memory |
| `qshimSyncThreads` | Barrier across threads within a core |
| `qshimSyncNSPs` | Barrier across NSPs |
| `qshimMulticastXfer` / `qshimWaitOnMulticast` | Multi-NSP VTCM multicast |
| `qshimGetNumCores` / `qshimGetNumThreads` / `qshimGetCoreId` / `qshimGetThreadId` | Runtime topology |
| `qshimProfilingEntryStart` / `qshimProfilingEntryEnd` | Chrome trace events from device |
| `qshimLog` | Device-side logging |
| `qshimGetRandomDataSubmit` / `qshimGetRandomDataWait` | Async RNG |
| `qshimMmap` / `qshimUnmap` | Map buffer handle to virtual address |

---

## Build

```python
# setup.py
from torch_qaic.custom_ops import QAicBuildExt, HexagonKernelExtension
from setuptools import setup

setup(
    ext_modules=[HexagonKernelExtension(sources=["my_kernel.cpp"])],
    cmdclass={"build_ext": QAicBuildExt},
)
```

```bash
# QAicBuildExt does not create the output directory — create it first
mkdir -p build/lib.linux-x86_64-cpython-310   # adjust Python version
python setup.py build_ext --inplace
# produces ./hexagon_kernels.so (name is always fixed)
```

Compiler: `hexagon-clang++ -mv68 -mhvx=v68 -mhmx -mhvx-ieee-fp -ffreestanding -std=c++17`
Toolchain: `/opt/qti-aic/dev/hexagon_tools` (override with `HEXAGON_TOOLS_DIR`)

Only one `HexagonKernelExtension` per distribution. All kernels compile into the single
`hexagon_kernels.so`.

---

## Dispatch from Python

```python
import torch_qaic.custom_ops as ops

# ops.<name> walks up from the calling file's directory to find hexagon_kernels.so,
# then resolves <name> as an exported C symbol.
ops.my_kernel[nsp_count, thread_count](tensor_a, tensor_b, count_tensor)
```

---

## Tensor layout requirements

`_dispatch_kernel` extracts only the raw data pointer from each tensor argument —
no shape, strides, or storage offset are passed to the kernel. The kernel has no way
to detect or handle non-contiguous tensors.

**Callers must ensure contiguous layout before dispatch.** Call `.contiguous()` in the
Python wrapper:

```python
@my_op.register_kernel("qaic")
def _(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()   # ensures flat row-major storage before pointer is extracted
    out = torch.empty_like(x)
    n = torch.tensor([x.numel()], dtype=torch.int32, device=x.device)
    ops.my_kernel[16, 4](x, out, n)
    return out
```

The built-in QAIC aten kernels (`mm`, `conv`, etc.) handle non-contiguous tensors
correctly because they go through the full aten dispatch path with stride metadata.
This does not apply to custom kernels dispatched via `_dispatch_kernel`.

---

## Integrating with aten dispatch

`_dispatch_kernel` bypasses the aten op registry. To make a kernel replace a
`UnimplCPUFallback`, register it via `torch.library`:

```python
import torch.library, torch_qaic.custom_ops as ops

@torch.library.custom_op("mylib::my_op", mutates_args=())
def my_op(x: torch.Tensor) -> torch.Tensor: ...

@my_op.register_kernel("qaic")
def _(x):
    out = torch.empty_like(x)
    ops.my_kernel[16, 4](x, out)
    return out
```

See `examples/scale_fp16/` for a complete working example.
