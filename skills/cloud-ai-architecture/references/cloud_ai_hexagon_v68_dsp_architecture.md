# Cloud AI 100 Hexagon v68 DSP Architecture

## Overview

Each Cloud AI 100 NSP contains a Qualcomm Hexagon v68 DSP. This reference
covers the directly programmable per-core hardware for custom kernels. HMX is
compiler-controlled for Conv, MatMul, and Gemm; it is not directly exposed to
custom kernels.

## 1. ISA Overview

| Property | Detail |
|---|---|
| Architecture family | QDSP6 (Hexagon), 7th generation scalar core |
| Instruction width | Fixed 32-bit per instruction |
| Dispatch | Up to **4 instructions per clock** (VLIW packet) |
| Threading | **6 hardware threads** per core (dynamic multithreading, DMT) |
| Thread switching | On L2 cache miss, interrupt, or explicit yield instruction |
| Data bus width | 32-bit (GPRs can be paired to 64-bit) |
| General-purpose registers | 32 x 32-bit (pairable to 64-bit) |
| Floating point | FP16 and FP32 (introduced in v5+) |
| Integer | INT8, INT16, INT32 |
| ISA type | Load-store (memory ops separate from compute) |

Hexagon v68 provides a scalar core tightly integrated with vector and tensor engines in Cloud AI 100. Its DMT threading and vector extensions are key to hiding memory latency and accelerating non-tensor operations.

---

## 2. VLIW Execution Model

Hexagon v68 uses a **VLIW (Very Long Instruction Word)** model:

- Up to **4 instructions per clock** can be dispatched as a **packet**.
- Each instruction in a packet is issued to a different execution unit (ALU, load/store, vector, branch, etc.).
- Instructions in the same packet must be independent (no intra-packet data dependencies).
- The compiler (Hexagon LLVM backend) is responsible for packing.

Conceptual view:

```text
Clock N:
  Packet = [ ALU_op | MEM_load | VEC_op | BRANCH ]
              ↓          ↓          ↓        ↓
          EX Unit 0  EX Unit 1  EX Unit 2  EX Unit 3
```

Implications:

- Instruction-level parallelism is extracted at compile time.
- Unused slots are filled with NOPs.
- Good compiler optimization (scheduling, bundling) is important for kernel performance.

---

## 3. Dynamic Multithreading (DMT)

Hexagon v68 supports **Dynamic Multithreading**:

- **6 hardware threads** share the scalar pipeline.
- Thread switch occurs on:
  - L2 cache miss
  - Interrupt
  - Explicit `wait` / yield-like instructions
- Purpose: hide memory latency by switching to another thread while one waits for memory.

**Note on custom ops:** Although Hexagon v68 provides 6 hardware threads, QAic Custom Ops exposes **4 HVX threads** (threadId 0–3) to custom operator kernels. The remaining threads are reserved for HMX (matrix engine operations) and synchronization. See `cloud_ai_hexagon_v68_programming_reference.md` and the `cloud-ai-custom-ops-kernels` skill for custom kernel threading patterns.

For inference workloads:

- Different threads may handle different tiles, layers, or micro-tasks.
- DMT helps sustain utilization when there are L2/DDR misses.

This threading model complements the VLIW execution: VLIW exploits instruction-level parallelism within a thread; DMT exploits concurrency across threads.

---

## 4. Hexagon Vector Extensions (HVX)

HVX is the SIMD/vector extension of Hexagon v68, used by the **Vector Unit** in each AI core.

| Property | Detail |
|---|---|
| Vector register width | 1024-bit (128 bytes) |
| Vector register file | 32 x 1024-bit vector registers |
| Supported types | INT8, INT16, FP16, FP32 |
| Approx. throughput (INT8) | 512 MAC/clock |
| Approx. throughput (FP16) | 256 MAC/clock |
| Instruction count | 536 HVX vector instructions (v19.0.06 toolchain) |
| Typical use cases | Activation functions, normalization, pre/post-processing, element-wise ops |

## 5. Memory Hierarchy per Core

At a high level, the on-core memory hierarchy for a Hexagon v68 NSP is:

```text
  Registers (scalar + vector)
       ↓  (sub-ns)
  VTCM – fast scratchpad (8 MB per core)
       ↓  (~1–4 cycles)
  L2 (L2TCM) – per-core, hardware-prefetched (1 MB per core)
       ↓  (~10–30 cycles)
  → off-core: DDR (external LPDDR4X, ~100+ GB/s)
```

Key points:

- **VTCM** is the primary working memory for HVX and tensor operations. Hot activations and weight tiles live here. Many model-design rules (channel alignment, residual sizing) derive directly from VTCM capacity.
- **L2 (L2TCM)** is per-core, hardware-prefetched, and used for scratch buffers, doorbell/sync slots, and multicast staging. It is shared across the scalar/vector/tensor units within a single core but **not across cores**.
- **DDR** is the off-core spill destination, ~100+ GB/s bandwidth.

---

## 6. Hexagon Toolchain


| Tool | Details |
|---|---|
| Compiler | Clang/LLVM 15 (Hexagon cross-compiler) |
| Toolchain package | `clang+llvm-15.0.5-cross-hexagon-unknown-linux-musl` |
| Toolchain source | <https://github.com/qualcomm/toolchain_for_hexagon> |
| LLVM backend | Hexagon target in upstream LLVM (v5+), HVX v66+ support in LLVM 8.0+ |
| Build flag requirement | **`-fno-rtti`** – LLVM is built without RTTI; client code must match |
| Target triple | `hexagon-unknown-linux-musl` |
| Output format | ELF (`.elf`) packaged into QAic Program Containers (`.qpc`) |

---

## External Reference Manuals

For full architectural details beyond this skill, consult:

- **Hexagon v68 Programmer’s Reference Manual**  
  <https://docs.qualcomm.com/doc/80-N2040-46/80-N2040-46_REV_B_Hexagon_V68_Programmers_Reference_Manual.pdf>
- **Hexagon v68 HVX Reference** (vector extension details)  
  <https://docs.alexrp.com/hexagon/hexagon_v68_hvx.pdf>

These documents are authoritative for the ISA, pipeline, HVX instructions, and
micro-architectural behavior.
