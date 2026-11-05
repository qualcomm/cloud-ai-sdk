---
name: cloud-ai-torch-qaic
description: Use when fine-tuning transformer models, running PyTorch in eager mode (qaic device string), using AMP/DDP/FSDP on AIC, debugging CPU fallbacks (UnimplCPUFallback), or writing JIT custom NSP kernels dispatched from Python via torch_qaic.custom_ops. Not for ONNX/QPC compilation — use cloud-ai-model-conversion or cloud-ai-qaic-compile. Not for AOT custom kernels — use cloud-ai-custom-ops-kernels.
---

# Skill: Cloud AI 100 – torch_qaic PyTorch Backend

torch_qaic registers `"qaic"` as a first-class PyTorch device type, enabling eager-mode
execution on Cloud AI 100 (AIC) hardware. Primary design goal is fine-tuning of
transformer models. Distributed as a wheel in every Cloud AI SDK image
(https://github.com/qualcomm/cloud-ai-containers).

For detailed API reference see [reference.md](references/reference.md).

For benchmarking latency, throughput, and QAIC/CPU op split, see
[examples/benchmark/bench.py](examples/benchmark/bench.py) and the
[Benchmarking](references/reference.md#benchmarking) section in reference.md.

---

## Setup

**Docker images** from [qualcomm/cloud-ai-containers](https://github.com/qualcomm/cloud-ai-containers) ship
a pre-installed venv at `/opt/torch-qaic-env`:

```bash
source /opt/torch-qaic-env/bin/activate
```

For manual installation outside Docker, see [Setup](references/reference.md#setup) in reference.md.

Requires a **CPU-only PyTorch build** — raises `RuntimeError` if another accelerator is loaded.

```python
import torch_qaic   # before torch, or set TORCH_DEVICE_BACKEND_AUTOLOAD=1
import torch
print(torch.qaic.device_count())
```

---

## Core Device API

```python
torch.qaic.device_count()            # int
torch.qaic.is_available()            # bool
torch.qaic.set_device(0)
torch.qaic.synchronize()
torch.qaic.get_device_info(0)        # .name, .num_cores, .has_bf16
with torch.qaic.device(1): ...

free, total = torch.qaic.mem_get_info(0)
torch.qaic.memory_allocated()
# empty_cache() is a no-op; max_memory_* stats are stubs in v0.1.0
```

---

## Tensor Operations

```python
x = torch.randn(4, 4).qaic()                           # .qaic() shorthand
x = torch.randn(4, 4, device="qaic")                   # device string
x = torch.randn(4, 4, device=torch.device("qaic", 0))  # specific index
x.is_qaic          # bool
x_cpu = x.cpu()
```

| dtype | Support |
|---|---|
| `torch.float32` | Full |
| `torch.float16` | Full (always available) |
| `torch.bfloat16` | Device-dependent — check `torch.qaic.is_bf16_supported()` |
| `torch.int8` / `int32` / `int64` | Supported (int64 treated as int32 internally) |
| `torch.float64` | **Not supported** — will error |
| `torch.complex*` / `torch.quint*` / `torch.float8*` | **Not supported** |

---

## torch.compile

`torch.compile` and graph lowering are **not supported in v0.1.0** — the qaic backends
(`qaic-inference`, `qaic-aot-autograd`, etc.) are not registered in this release. All
inference and fine-tuning runs in eager mode.

---

## NSP Dispatch

In eager mode the runtime uses **all available NSPs** on the device for every op — there
is no per-op or per-session core allocation. Custom JIT kernels control NSP count
explicitly via `ops.my_kernel[nsp_count, thread_count]`.

See [NSP Allocation](references/reference.md#nsp-allocation) in reference.md for details
and contrast with AOT `-aic-num-cores`.

---

## Custom Hexagon NSP Kernels

When an op has no built-in QAIC kernel it falls back to CPU (`UnimplCPUFallback`).
You can move it back onto the NSP by writing a C++/HVX kernel and dispatching it via
`torch_qaic.custom_ops`. This is a **JIT eager-mode** mechanism — kernels are called
directly from Python and are not baked into a QPC binary.

For NSP/HVX architecture background (core topology, VTCM, programming guide, vector intrinsics) see
`cloud-ai-architecture`.

See [custom_kernels.md](references/custom_kernels.md) for the full interface: kernel entry
point signature, work partitioning, JIT headers, NSP runtime services, build instructions,
tensor layout requirements, and a link to the working `examples/scale_fp16/` example.

---

## Known Limitations (v0.1.0)

- `aten::embedding` / `aten::embedding_dense_backward` have no built-in QAIC kernel — token lookups fall back to CPU. Can be addressed with a custom NSP kernel (see above).
- Pooling ops (`avg_pool2d`, `max_pool2d`, `adaptive_avg_pool2d`) fall back to CPU — CNN backbones are most affected.
- FFT/STFT (`torch.fft`, `torch.stft`) fall back to CPU — spectrogram-in-graph models (Whisper, AST) are affected.
- QCCL `send`/`recv` P2P ops not fully supported.
- Memory APIs largely stubs: `empty_cache()` no-op; `max_memory_allocated()` / `memory_stats()` return `None`/`{}`.
- `torch.float64` not supported on any AIC device.
- Requires CPU-only PyTorch — cannot coexist with GPU-accelerated torch or other out-of-tree backends.
