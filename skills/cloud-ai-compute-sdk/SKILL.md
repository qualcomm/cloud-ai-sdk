---
name: cloud-ai-compute-sdk
description: Use when writing standalone C/C++ compute applications for Cloud AI 100 NSPs with the QAic Compute SDK, including `qaic-cc`, `activate()`, program configuration, or inter-NSP multicast. Not for ML-model inference, ONNX custom ops, or PyTorch eager-mode kernels.
---

# Cloud AI 100 — QAic Compute SDK

Repo: `https://github.com/qualcomm/software-kit-for-qualcomm-cloud-ai-100-cc` (BSD-3-Clause-Clear)

## 1. What this skill covers

The QAic Compute SDK is the **third NSP programming model**:

| Programming model | What you write | Where it runs | Use case |
|---|---|---|---|
| Host inference | C++ host app calling `QAicApi.hpp` | x86 host | Load and run compiled QPCs, manage device sessions |
| Custom op (in-graph) | Kernel inside an ONNX node | NSP, dispatched by model | Replace or extend an op inside a compiled ML model |
| **Compute app** | **`extern "C" void activate()` entry point** | **NSP, standalone** | **Fully custom compute, no ML graph** |
| JIT torch_qaic | PyTorch eager-mode kernel | NSP, dispatched from Python | Replace an unimplemented op in PyTorch eager mode |

## 2. Tools at a glance

| Tool / Term | Kind | One-liner |
|---|---|---|
| `qaic-cc` | Host binary | Hexagon cross-compiler driver — replaces `clang` for building NSP binaries; accepts standard clang flags plus `-qaic-program-config`. **Not** `qaic-compile` (which compiles ONNX models). |
| `qaic-runner` | Host binary | Loads and executes any `.qpc` on a physical device — the same tool used for model inference. |
| `qaic.cmake` | CMake toolchain file | Sets `CMAKE_C/CXX_COMPILER` to `qaic-cc`; provides `add_qaic_executable` / `add_qaic_library` helpers. |
| `QAICProgramConfig` (`.json`) | Config file | Host↔device contract: declares number of NSPs, threads per NSP, and all I/O buffers (name, type, shape, memory destination). Passed to `qaic-cc` via `-qaic-program-config`; embedded into the QPC at link time. See §6. |
| `QPC` | Binary format | Qualcomm Program Container — output of the `qaic-cc` link step; identical format to model QPCs. |
| `MCID` | Hardware concept | Multicast ID — substrate for one-to-many NSP-to-NSP writes via the UDMA engine. |
| `VTCM` | Memory tier | Vector Tightly Coupled Memory — 8 MB per NSP fast scratchpad (1–4 cycle latency). |
| `L2TCM` | Memory tier | Level-2 Tightly Coupled Memory — 1 MB per NSP (10–30 cycle latency), default buffer destination. |

## 3. Programming model

### Entry point

```cpp
extern "C" void activate(void *ctx, uint8_t virtualThreadId, uint32_t stid);
```

- `ctx` is `AICExecContext *` from `lib/metadata/metadata-common/inc/AICMetadataExecCtx.h`
- `virtualThreadId` is 0..N-1 within a single NSP; call `ctx->virtualNSPId` to identify the current NSP
- `stid` is the QuRT hardware software-thread-context ID — only useful for PMU thread filtering (see §9)
- No `main()` — the runtime's `_qaic_start.cpp` is the real entry point; it tail-calls `activate()`
- SPMD: same binary on every requested NSP; all threads invoke `activate()` simultaneously — dispatch on `virtualThreadId`

Init/exit flow: **references/runtime-api.md**.

### Bare-metal constraints

- Toolchain enforces `-fno-exceptions` and `-nostdinc++` — no C++ stdlib, no exceptions
- No file I/O, no `printf` — use `NN_LOG`
- All buffers statically declared in the JSON config (see §6) — no dynamic allocation

## 4. Build

### Step 1 — Build and install the SDK

`qaic-cc` is **not pre-installed**. Build and install it from the upstream
repository's README, which lists Hexagon toolchain v15.0.5 as its tested
version.

After install, export:

```bash
export QAIC_COMPUTE_INSTALL_DIR=/path/to/install/qaic-compute-release-assert
export PATH=$QAIC_COMPUTE_INSTALL_DIR/exec:$PATH
export HEXAGON_TOOLS_DIR=/path/to/clang+llvm-15.0.5-cross-hexagon-unknown-linux-musl
```

### Step 2 — Build a user app

```bash
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$QAIC_COMPUTE_INSTALL_DIR/dev/cmake/qaic.cmake \
      -DCMAKE_C_COMPILER_WORKS=TRUE -DCMAKE_CXX_COMPILER_WORKS=TRUE ..
make
```

**`-DCMAKE_C_COMPILER_WORKS=TRUE` is always required** (`qaic-cc` is C++-only; CMake's C-language self-test probe fails against it).

Add `add_qaic_executable(<name> <config.json> <sources...>)` in `CMakeLists.txt` to force `.qpc` output and inject `-qaic-program-config` automatically. See `examples/*/CMakeLists.txt` for the canonical pattern.

## 5. Starting templates

Three examples ship in `examples/` (installed to `examples/compute/`):

| Example | NSPs | What it shows | Start here when… |
|---|---|---|---|
| `barebones_app` | 14 | Minimum viable QPC — `activate()` logs one line, no I/O handshake | Proving the toolchain end-to-end; JSON config skeleton |
| `simple_io` | 1 | Full I/O handshake: `readyForAllInputs` → `waitForBuffer` → compute → `sendAllOutputs` | Writing any single-NSP compute app |
| `multicast_example` | 3 | Fan-out/fan-in across NSPs via internal VTCM buffers; buffer offset overlap trick | Multi-NSP coordination, MCID usage |

Copy an example to your workspace and build using §4 Step 2:

```bash
cp -R $QAIC_COMPUTE_INSTALL_DIR/examples/compute/simple_io /your/workspace/
```

## 6. JSON config — `QAICProgramConfig`

Passed via `-qaic-program-config <file>`. Source of truth: `lib/program/ProgramConfig.proto`.

### Top-level keys

| Key | Meaning |
|---|---|
| `numNSPs` | 1..16; hard ceiling `MAX_NUM_CORES = 16` |
| `hwVersionMajor`, `hwVersionMinor` | **Must be 2 / 0** |
| `numThreads` | Threads per NSP (default 4, max 6) |
| `inputBuffers`, `outputBuffers`, `internalBuffers` | Buffer descriptor arrays |

### Buffer descriptor fields

Full schema in `lib/program/ProgramConfig.proto`. Non-obvious gotchas:

| Field | Gotcha |
|---|---|
| `dataType` | Enum: `FloatTy`, `Float16Ty`, `Int8QTy`, `UInt8QTy`, `Int16QTy`, `Int32QTy`, `Int32ITy`, `Int64ITy`, `Int8Ty` |
| `dest` | Enum: `L2TCM` (default), `VTCM`, `DDR` — see §2 for sizes |
| `baseAddrOffset` | **Must be 4 KiB-aligned for L2TCM/VTCM** — compile error if not |

`internalBuffers` are device-only; used for inter-NSP communication via MCIDs.

## 7. Device-side runtime API

I/O handshake order: `readyForAllInputs` → `waitForBuffer(inputBufferNum(i))` per input → `waitForAllOutputsReady` → write outputs → `sendAllOutputs`. See `examples/simple_io/SimpleIO.cpp` for the canonical implementation.

Logging:
```cpp
NN_LOG(qctx->logFuncPtr, NNC_LOG_MASK_DEBUG, "thread %d", virtualThreadId);
```

**Capturing `NN_LOG` output at the host:** pass `--collect-device-log info` (short: `-u info`) to `qaic-runner`. Without this flag the messages are silently dropped — `qaic-log` does **not** surface them.

```bash
qaic-runner -t . -i input.bin --write-output-dir out --num-iter 1 \
  --collect-device-log info
```

Valid levels: `error`, `warn`, `info`, `verbose`. `info` is sufficient for `NNC_LOG_MASK_DEBUG` and above.

Full API signatures, init/exit flow, UDMA, multicast walkthrough: **references/runtime-api.md**.

## 8. Host runtime interop

The `.qpc` is **byte-format-compatible** with model QPCs:

- `qaic-runner` and `QAicApi.hpp` load either interchangeably
- `AICExecContext` (the `ctx` pointer) is the same struct ML inference workloads see
- `ProgramGroup` can mix a compute QPC and a model QPC on the same SoC

## 9. PMU profiling

Hexagon v68 PMU access is manual and device-side only via two function pointers on the exec context:

```cpp
qctx->setPMUReg(reg_id, value);          // configure (event select, enable, thread filter)
qctx->readPMUCnt(uint32_t counters[8]);  // read all 8 counters
```

Counter values must be exfiltrated manually — write to a DDR output buffer or print with `NN_LOG`.

**`qaic-runner --aic-profiling-type raw_device_stats` does not work for compute SDK QPCs.** `qaic-cc` has no `-stats-level` equivalent and never populates the `Stats` proto, so the stats buffer required by that path is never embedded.

Full register reference, usage pattern, and exfiltration examples: **references/profiling.md**.

## 10. Common pitfalls

- **README references `examples/compute/`** but the repo path is `examples/`. That path is the *installed* layout.
- **`qaic-runner` requires a `programqpc.bin` symlink**, not the original `.qpc` filename.
- **Stack overflow is silent.** Default per-thread stack is 12 288 B; large stack arrays corrupt adjacent memory with no diagnostic.
- **`hwVersionMajor:2 hwVersionMinor:0` is mandatory** in JSON.
- **HMX is not available** in the open Hexagon 15.0.5 toolchain (`-mhmx` not accepted).
- **`/opt/qti-aic/dev/hexagon_tools` is not compatible.** The internal Qualcomm hexagon-link does not support `-z separate-code`, which `qaic-cc` unconditionally adds. Always use the codelinaro community toolchain as `HEXAGON_TOOLS_DIR`.
