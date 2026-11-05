# torch_qaic Reference

Detailed API reference for the `cloud-ai-torch-qaic` skill.

---

## Setup

Docker images from [qualcomm/cloud-ai-containers](https://github.com/qualcomm/cloud-ai-containers) ship
a pre-configured venv at `/opt/torch-qaic-env` with CPU-only PyTorch and torch_qaic already
installed:

```bash
source /opt/torch-qaic-env/bin/activate
```

For manual installation, create a CPU-only PyTorch venv first (torch_qaic raises `RuntimeError`
if a GPU-accelerated build is present):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Python 3.10
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl
# Python 3.11
pip install /opt/qti-aic/integrations/torch_qaic/py311/torch_qaic-0.1.0-cp311-cp311-linux_x86_64.whl
# Python 3.12
pip install /opt/qti-aic/integrations/torch_qaic/py312/torch_qaic-0.1.0-cp312-cp312-linux_x86_64.whl
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `QAIC_VISIBLE_DEVICES` | Comma-separated device indices to expose |
| `QAIC_DEBUG` | Debug levels to enable (integers 1–9). E.g. `QAIC_DEBUG=8` enables verbose dispatch tracing |
| `QAIC_DUMP_DIR` | Directory for profiling/debug dumps (default: `./qaic-dumps`) |
| `QAIC_SIM_MODE` | Enable simulator mode (no physical device needed) |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | Set to `0` to disable auto-import of torch_qaic on `import torch` |
| `HEXAGON_TOOLS_DIR` | Hexagon toolchain path (default: `/opt/qti-aic/dev/hexagon_tools`) |
| `QAIC_PLATFORM_JIT_INCLUDE_DIR` | JIT include path for custom kernel builds (default: `/opt/qti-aic/dev/inc/jit`) |
| `MASTER_ADDR` / `MASTER_PORT` | Standard distributed training rendezvous |

---

## Registered Compute Kernels

The following aten ops have native QAIC kernels. Ops NOT in this list fall back to CPU automatically.

**Linear algebra**: `mm`, `bmm`, `addmm`

**Activations**: `relu`, `gelu`, `gelu_backward`, `sigmoid`, `silu`, `tanh`, `leaky_relu`

**Normalization**: `native_layer_norm`, `native_batch_norm`, `rms_norm`

**Attention**: `scaled_dot_product_attention` (fused)

**Softmax**: `_softmax`, `_log_softmax`, `_safe_softmax`, `_softmax_backward_data`, `_log_softmax_backward_data`

**Convolution**: `convolution_overrideable`, `convolution_backward_overrideable`

**Elementwise**: `add`, `sub`, `mul`, `div`, `pow`, `abs`, `exp`, `log`, `sqrt`, `rsqrt`, `reciprocal`, `cos`, `sin`, `neg`, `clamp`, `lerp`, `where`, `addcdiv`, `addcmul`

**Comparison/bitwise**: `eq`, `ne`, `gt`, `lt`, `le`, `bitwise_and`, `bitwise_not`, `bitwise_or`, `logical_and`, `logical_not`, `logical_or`

**Reduction**: `mean`, `sum`, `max`, `any`, `all`, `cumsum`

**Indexing**: `index_select`, `index`, `index_copy`

**Dropout**: `native_dropout`, `native_dropout_backward`

**Loss**: `nll_loss_forward`, `nll_loss_backward`

**AMP training**: internal GradScaler ops (called automatically by `GradScaler`)

**Memory/view**: `empty`, `empty_strided`, `arange`, `fill_`, `zero_`, `resize_`, `view`, `as_strided`, `unfold`, `view_as_real`, `view_as_complex`, `_copy_from`, `masked_fill_`, `normal_`, `uniform_`, `tril`, `triu`, `cat`

**Custom**: `qaic::rotate_half` — native rotary position embedding (RoPE) kernel

**Distributed (c10d via QCCL)**: `allreduce_`, `allgather_`, `broadcast_`, `reduce_scatter_`, `alltoall_`, `barrier`, `send`, `recv`, `gather_`, `scatter_`, plus FSDP collective ops

---

## Model Architecture Coverage

| Architecture | On-device coverage | Key CPU fallbacks |
|---|---|---|
| **Transformer inference** (GPT, LLaMA, BERT, T5) | ~90% | `aten::embedding` (token lookups) |
| **Vision Transformer** (ViT, Swin, DeiT) | ~90% | minimal |
| **Text classification** (BERT, RoBERTa, DistilBERT) | ~85% | `aten::embedding`, `aten::embedding_dense_backward` |
| **CNN vision** (ResNet, EfficientNet, MobileNet) | ~65–75% | all pooling: `avg_pool2d`, `max_pool2d`, `adaptive_avg_pool2d` |
| **Audio — conv-based** (wav2vec2, HuBERT) | ~85% | `aten::embedding` if present |
| **Audio — spectrogram-in-graph** (Whisper, AST) | ~60–70% | FFT/STFT (`torch.fft.rfft`, `torch.stft`) + any pooling |

Transformer models are the primary target. The `aten::embedding` gap means token lookups
execute on CPU. CNN backbones lose all pooling layers. Audio models computing spectrograms
in-graph (Whisper) lose FFT — precomputing offline avoids this.

To check fallbacks after a forward pass:
```python
from torch_qaic import stats
stats.get_global_fallback_op_schemas()  # {schema_str: count}
```

---

## torch.compile

Not supported in v0.1.0 — the qaic backends (`qaic-inference`, `qaic-aot-autograd`, etc.)
are not registered. Calling `torch.compile(model)` falls through to Inductor, which does not
support the `qaic` device and will error. All workloads run in eager mode.

---

## AMP / Mixed Precision Training

```python
# Autocast (forward pass)
with torch.autocast(device_type="qaic", dtype=torch.float16):
    output = model(input)

torch.qaic.is_autocast_enabled()
torch.qaic.get_autocast_dtype()           # default: torch.float16
torch.qaic.set_autocast_dtype(torch.bfloat16)  # if device supports it
torch.qaic.get_amp_supported_dtype()

# GradScaler for FP16 training
from torch_qaic.qaic.amp import GradScaler

scaler = GradScaler()
for input, target in dataloader:
    optimizer.zero_grad()
    with torch.autocast(device_type="qaic", dtype=torch.float16):
        output = model(input)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Autocast covers: all conv variants, matmul/mm/bmm/linear/einsum, all loss functions
(nll, l1, mse, kl_div, cross_entropy, etc.), layer/group norm, RNN cells, upsample
variants, and `scaled_dot_product_attention`.

---

## Distributed Training (QCCL)

QCCL (Qualcomm Collective Communications Library) is registered as the `"qccl"` backend
for `torch.distributed`.

```python
import torch.distributed as dist
import torch_qaic  # must import to register qccl backend

dist.init_process_group(
    backend="qccl",        # or "qaic:qccl"
    init_method="env://",
    world_size=...,
    rank=...,
)

from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model.qaic(), device_ids=[local_rank])
```

FSDP is also supported. **Known limitation**: CPU fallback for QCCL is disabled;
`send`/`recv` P2P ops are not fully supported in v0.1.0.

---

## RNG / Reproducibility

```python
torch.qaic.manual_seed(42)
torch.qaic.manual_seed_all(42)

state  = torch.qaic.get_rng_state()
states = torch.qaic.get_rng_state_all()
torch.qaic.set_rng_state(state)
torch.qaic.set_rng_state_all(states)
```

---

## Profiling

```python
from torch_qaic.profile import ProfileForwardWithSampling, ProfileOpsWithSamplingMode

# Profile N forward passes (skip_first_n warmup)
with ProfileForwardWithSampling(
    model, n_samples=3, skip_first_n=2, device_name="qaic:0", dump_dir="./hw_trace"
) as profiled_model:
    for batch in dataloader:
        profiled_model(batch)
trace_files = profiled_model.get_trace_files()  # Chrome-compatible JSON

# Profile exactly N op dispatches
with ProfileOpsWithSamplingMode(num_ops=10, skip_first_n=5, device_name="qaic:0") as p:
    output = model(input)
trace_files = p.get_trace_files()
```

---

## Benchmarking

`examples/benchmark/bench.py` is a self-contained template that measures wall-clock latency,
throughput, and the QAIC/CPU op split for any model:

```bash
source /opt/torch-qaic-env/bin/activate

python bench.py                         # BERT-base, seq_len=128, batch=1
python bench.py --model resnet50        # ResNet50, batch=1
python bench.py --model bert --seq-len 256 --batch 4
python bench.py --warmup 5 --iters 20
```

Measured results (batch=1, eager mode):

| Model | Latency | Throughput | On QAIC |
|---|---|---|---|
| BERT-base (seq=128) | 103 ms | 9.7 seq/sec | ~100% |
| ResNet50 | 4232 ms | 0.2 img/sec | 53% |

ResNet50's poor result is expected: without `torch.compile` (not supported in v0.1.0),
all 53 convolutions dispatch as `aten::conv2d` and fall back to CPU. BERT runs almost
entirely on-device because its core ops (`mm`, `bmm`, `scaled_dot_product_attention`,
`native_layer_norm`) have registered QAIC kernels.

---

## Op Dispatch Debugging

```python
from torch_qaic import stats

stats.get_global_op_stats()
# {"Compute": N, "NonCompute": N, "ImplCPUFallback": N, "UnimplCPUFallback": N}
# ImplCPUFallback  = kernel exists but dtype/layout unsupported
# UnimplCPUFallback = no kernel for this op

stats.get_global_fallback_op_schemas()  # {schema_str: count}
stats.get_recent_op_schemas()           # [(dispatch_type, schema), ...]
stats.reset_recent_stats()
```

```python
from torch_qaic.utils import TraceOpsMode, TraceOpsWithModuleHierarchyMode

with TraceOpsMode(prefix="fwd"):         # logs every op to stderr
    output = model(input)

with TraceOpsWithModuleHierarchyMode():  # nested under module names
    output = model(input)
```

```python
from torch_qaic.utils import save_dispatch, load_dispatch

save_dispatch(torch.ops.aten.mm.default, (a, b), {}, torch.device("qaic"), "mm.pt")
func, args, kwargs, device, config = load_dispatch("mm.pt")
result = func(*args, **kwargs)
```

```python
from torch_qaic.utils import RooflineEstimateMode

config = RooflineEstimateMode.Config.default()
with RooflineEstimateMode("exp", config, dump_file="roofline.xlsx") as r:
    output = model(input)
# Excel: per-op read/write bytes, exec time, theoretical bounds
```

---

## NSP Allocation

### Eager mode — full device, no user control

The torch_qaic runtime dispatches every built-in QAIC kernel across all NSPs on the
device. On a 16-core SKU (AI 100 Pro / Ultra SoC), all 16 NSPs participate in each op.
There is no API to restrict which cores are used or to share cores between sessions in
eager mode.

Each NSP has 4 HVX threads and 1 HMX thread. The kernel partitions its workload across
cores and threads using the `AicJitEntryPointConfig` fields (`coreID`, `numCores`,
`threadID`, `numThreads`) — see the Custom Hexagon Kernels section for the entry point
signature.

### Custom JIT kernels — explicit control

Custom kernels dispatched via `torch_qaic.custom_ops` take an explicit `[nsp_count, thread_count]`
parameter:

```python
ops.my_kernel[nsp_count, thread_count](tensor_a, tensor_b, count_tensor)
```

Use `[1, 4]` for single-NSP dispatch (4 HVX threads), `[16, 4]` to use all cores on a
16-core device.

### Contrast with AOT qaic-compile

`qaic-compile`'s `-aic-num-cores=N` compiles a QPC that occupies exactly N cores,
leaving the remainder free for other QPCs to co-reside via ProgramGroup. There is no
equivalent in torch_qaic eager mode — each op takes the full device, so multiple
concurrent inference sessions are not supported within a single torch_qaic process.

---

## Custom Hexagon Kernels

See [custom_kernels.md](custom_kernels.md) for the full reference: kernel entry point
signature, work partitioning, JIT headers, NSP runtime services, build instructions,
tensor layout requirements, and aten dispatch integration.

---

## Global Configuration

```python
cfg = torch_qaic.global_config    # TorchQAicConfig instance

cfg.enabled_debug_levels          # list of 10 bools, index = debug level
cfg.is_debug_level_enabled(8)
cfg.is_autocast_disabled          # bool
cfg.cpu_fallback_config           # dict of per-op fallback overrides
cfg.qaic_dumps_dir                # str, default "./qaic-dumps"
```

Debug levels are set via `QAIC_DEBUG` env var (e.g. `QAIC_DEBUG=8` enables verbose
dispatch tracing).
