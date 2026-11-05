# Cloud AI 100 SoC Architecture

## 1. Cloud AI 100 Device Overview

Cloud AI 100 cards contain one or more AI SoCs, each with multiple AI cores (NSPs) and attached DDR memory.

### Platform SKUs

| Accelerator SKU | AI 100 Standard | AI 100 Pro | AI 100 Ultra | AI 080 Ultra |
|---|---|---|---|---|
| AI SoC Count | 1 | 1 | 4 | 4 |
| **AI cores per SoC** | 14 | 16 | **16** | 8 |
| **AI cores per card** | 14 | 16 | **64** | 32 |
| DDR Capacity | 16 GB | 32 GB | **128 GB** | 128 GB |
| DDR Bandwidth | 137 GB/s | 137 GB/s | **548 GB/s** | 548 GB/s |
| On-chip SRAM | 126 MB | 144 MB | **576 MB** | 576 MB |
| FP16 Throughput | 110 TFLOPs | 125 TFLOPs | **290 TFLOPs** | 222 TFLOPs |
| INT8 Throughput | 325 TOPs | 375 TOPs | **870 TOPs** | 618 TOPs |
| PCIe Host Link | Gen4 x8 | Gen4 x8 | **Gen4 x16** | Gen4 x16 |
| TDP | 75 W | 75 W | 150 W | 150 W |

**Ultra topology**: The AI 100 Ultra has 4 SoCs connected via an on-card **PCIe Gen4 switch** — the host sees a single Gen4 x16 link, which the switch fans out to 4 × Gen4 x8 links to each SoC.

Each AI SoC features:
- **16 seventh-generation AI cores** (NSPs) — Hexagon v68 DSPs (Pro/Ultra), 14 cores (Standard), 8 cores (080)
- **144 MB total on-chip SRAM** — aggregate of per-core VTCM + L2 across 16 cores; see §4 for the addressable hierarchy
- **External LPDDR4X**: 4 channels × 64-bit = 137 GB/s bandwidth, up to 32 GB capacity per SoC
- **PCIe Gen4 x8** host interface per SoC

From a model and performance perspective, the key concepts are:

- How many cores you use: `-aic-num-cores`
- How weights and activations are partitioned across cores: `-mos`, `-ols`, SDP
- How subgraphs or tensor slices are mapped across devices: MDP

---

## 2. Foundational Performance Tenets

These three principles guide compiler-metric interpretation and optimization candidate selection.

### Tenet 1 — Minimize Repeated VTCM Spill/Fill Traffic

VTCM is 8 MB of fast scratchpad **per core**. Use its capacity for active,
reusable tiles. Performance suffers when poor tile partitioning or reuse causes
repeated VTCM spill/fill traffic to DDR.

**Partition work across cores so each core can retain active reusable tiles in VTCM.** Each SoC has up to 16 NSP cores connected by a NoC; the compiler partitions large tensors across them so each core only needs to hold its slice. Two complementary strategies operate within an SoC:

- **Channel-wise (`-mos`, max output-channel split)** — splits output channels of Conv/MatMul across cores. Each core holds a slice of the weights and computes its share of output channels; NoC multicast distributes weights shared across cores. Shrinks the per-core weight tile in VTCM.
- **Spatial (`-ols`, overlap split)** — splits activations along spatial dimensions with tile overlap across cores; each core processes a spatial region of D32 Conv / Pool / Decimate ops. Useful when activations dominate the target metric. Tradeoff: OLS keeps overlapping tiles resident simultaneously, so sweep empirically and back off only after a real failure signal: compile failure, per-core `DDR traffic write` rising, spill/BAR warnings, runtime `DMAIssue` climbing materially, or ops flipping to a `DDR` suffix. Do not skip solely because baseline `VTCMMaxSize` is high or low.

**At a larger memory scale: when the model itself does not fit in a single SoC's DDR, tensor-slice across SoCs.** This is a separate concern from per-core VTCM fit — here the constraint is total weight footprint exceeding the SKU's per-SoC DDR capacity (see §1 for SKU figures). MDP + TS partitions tensors across multiple SoCs (1, 2, 4, 8, or 16) connected by PCIe P2P DMA, with the compiler emitting the slice/gather communication automatically. See §7.

Levers that reduce repeated VTCM spill/fill traffic: 32-channel alignment (avoids padding waste), BatchNorm and activation fusion (avoids extra VTCM writes/reads), residual tensor budgeting (both branches must fit at the merge), `-multicast-weights` for shared weights, depth-first execution to reuse tiles before eviction.

Diagnostic signals (compile.log + opstats):

- `VTCMMaxSize` near 8 MB → the compiler is using most of the available per-core VTCM capacity for this compile; not evidence of a spill.
- per-core `DDR traffic write`, spill/BAR warnings, DDR-suffix ops, or rising `DMAIssue` → repeated VTCM spill/fill traffic to DDR
- `Not VTCM` multicast count > 0 → weight multicasts spilling
- `aiccopysamevtcm HVX CacheableDDR` in opstats top-ops → explicit DDR↔VTCM staging copies

### Tenet 2 — Keep the Matrix Unit (HMX) Busy

HMX delivers the bulk of compute throughput. Conv and MatMul that miss HMX run on HVX and are significantly slower. Two things keep HMX busy:

- **Eligibility** — HMX accepts FP16 or INT8 only. Conv geometry must satisfy the rules in §8.2; ops that violate them fall back to HVX.
- **Pipelining** — When HMX is immediately followed by HVX (activation, Add, scale/shift), the compiler pipelines them at tile granularity so both units stay busy. Cast / Quantize / Reshape nodes inserted between an HMX op and its HVX consumer break the pipeline; HMX idles while HVX serializes (see §3.1).

Diagnostic: `AIC non-HMX MAC estimate` should be zero. Non-zero means at least one Conv or MatMul missed HMX. High `sync HVX` with low HMX utilization in opstats is the signature of broken HMX→HVX fusion.

### Tenet 3 — Compile a Static Graph

The hardware executes a fully static schedule of tile movements and compute. Dynamic shapes, runtime control flow (Loop / If with runtime conditions), and runtime-computed shape attributes (Reshape `shape`, Slice indices, TopK `k`, OneHot `depth`, Resize `scales`/`sizes`) have no place in that schedule. Bake all such inputs as constants during export. This is a hard compile-time requirement, not a performance lever — a graph with dynamic structure will not compile.

---

## 3. AI Core (NSP) – Compute Units

Each AI core (NSP) consists of three main units, each with a distinct role:

- **HMX (Tensor/MAC unit)** — the matrix engine: convolutions, matrix multiplications, Gemm. **1 HMX thread per core**, delivering the bulk of the core's compute throughput on FP16 / INT8 workloads.
- **HVX (Vector unit)** — the per-core SIMD engine for non-matrix vector work: element-wise math, activation functions, normalization, vectorized reductions, indexing, layout transforms, format conversion. **4 HVX threads per core.**
- **Scalar unit** — orchestrates the core. It dispatches HMX and HVX work, drives DMA submissions for VTCM↔DDR transfers, handles control flow and synchronization barriers, and runs the inherently sequential portions of ops (e.g. the per-class box suppression loop in NMS, scan/prefix accumulation). The scalar unit is *not* a compute-throughput resource — when an op spends meaningful time here, that time is communication and orchestration overhead, not parallelizable work.

Hexagon v68 provides 6 hardware threads per core; the partitioning above (1 HMX + 4 HVX + 1 scalar/sync) reflects how the AIC backend assigns them. For per-core hardware detail (ISA, VLIW execution, dynamic multithreading, HVX register specs), see the Hexagon v68 DSP architecture doc.

### 3.1 HMX/HVX Fusion and the Cost of Broken Fusion

For any op sequence where an HMX op produces output immediately consumed by an HVX op (Conv→Relu, Conv→Add, Conv→BatchNorm, MatMul→Add, or any HMX op followed by an elementwise scale/shift or activation), the compiler can **fuse** them: it pipelines at tile granularity, feeding each HMX output tile directly to HVX while HMX begins the next tile. This keeps both units busy with minimal synchronization overhead.

**When fusion is broken** — by any node inserted between the HMX op and the HVX op in the ONNX graph (Cast, Quantize, Reshape, or any extra elementwise) — the compiler must treat them as separate passes:

1. HMX completes the entire op and writes its output to VTCM/L2.
2. An explicit synchronization barrier fires.
3. The HVX threads run the second op while HMX idles.

This serialization produces a characteristic opstats signature: **high `sync HVX`** (HVX threads accumulating sync-stall cycles at every layer boundary) combined with **low HMX utilization** (HMX idle during every HVX pass). The low HMX utilization is not repeated VTCM spill/fill traffic in this case — HMX is waiting for HVX to drain before it can accept the next tile.

**Common breaking causes:** ONNX transformation scripts that insert Cast or Quantize nodes for FP16 preparation between ops that were originally adjacent; `onnxsim` passes that introduce intermediate Reshape nodes; any graph surgery that places a new node between an HMX op and its immediate HVX consumer.

**Fix:** Remove the inserted nodes or recompile from the pre-transformation ONNX. If a transformation script was applied, try compiling the onnxsim-only ONNX instead and compare opstats — fusion restored → HMX utilization rises and `sync HVX` drops.

---

## 4. Memory Hierarchy

Cloud AI 100 has a three-level on-chip + external DDR memory hierarchy. Understanding which level a tensor lives in is central to performance — Tenet 1 in §2 is precisely the principle that the active working set must stay in VTCM.

### 4.1 Hierarchy at a Glance

| Level | Scope | Capacity | Latency (approx) | Primary use |
|---|---|---|---|---|
| **VTCM** | per-core | 8 MB | ~1–4 cycles | active weight and activation tiles |
| **L2 (L2TCM)** | per-core | 1 MB | ~10–30 cycles | scratch buffers, cache for DDR loads, doorbell/sync slots, multicast staging |
| **DDR** | per-SoC (external) | 16–32 GB per SoC by SKU (see §1) | hundreds of cycles | model weights, host-staged inputs, results |

The §1 SKU table's **"On-chip SRAM"** column is the **aggregate of VTCM + L2 across all cores on the card**: AI 100 Pro 16 cores × (8 + 1) MB = 144 MB; Standard 14 × 9 = 126 MB; Ultra 64 × 9 = 576 MB. It is a card-spec capacity number, not a separately addressed memory level. The compiler exposes only VTCM, L2TCM, and DDR (plus a CacheableDDR variant) as placement targets.

When a live tile cannot be retained in VTCM, it is spilled to L2 and then DDR; each lower tier costs roughly an order of magnitude more cycles. Per-core latency figures and pipeline interactions are detailed in the Hexagon v68 DSP architecture doc §5.

### 4.2 VTCM — Active-Tile Locality

VTCM is the per-core fast scratchpad that holds the active tiles HMX and HVX operate on. **8 MB per core.** Most Cloud AI 100 model-design rules — channel alignment to 32, BatchNorm fusion, residual sizing, tile granularity — preserve VTCM residency and reuse for those tiles. See Tenet 1 in §2 for the parallelism strategies.

**Diagnostics:**

- `VTCMMaxSize` in `qaic-compile -aic-perf-metrics` reports peak allocated VTCM for the compiled configuration. It shows how much per-core capacity the compile uses; it is not a universal cutoff for MOS, OLS, batching, or depth-first.
- `Not VTCM` in multicast counts (compile.log) means multicast destinations are spilling out of VTCM.
- Per-core `DDR traffic write`, spill/BAR warnings, DDR-suffix ops, and rising `DMAIssue` indicate real off-chip activation traffic.
- `aiccopysamevtcm HVX CacheableDDR` in opstats top-ops indicates compiler-inserted DDR↔VTCM staging copies.

For evidence-based tuning of these diagnostics, use `cloud-ai-performance-optimization`.

### 4.3 L2 (L2TCM)

L2 is a per-core, hardware-prefetched cache and scratch sitting between VTCM and DDR — **1 MB per core**. The compiler directly targets L2TCM for some scratch buffers, doorbell/sync slots, and multicast staging, but most tile-placement decisions are between VTCM and DDR. Loads that miss VTCM are served from L2 (or, on miss, from DDR). For per-core L2 detail and cycle costs, see the Hexagon v68 DSP architecture doc §5.

### 4.4 DDR (External Memory)

Each SoC has external LPDDR4X — 16–32 GB capacity per SoC by SKU, ~137 GB/s bandwidth per SoC. This is where the model weights and host-staged inputs live. DDR is the slowest memory; models that continuously stream from it are **DDR-bound**, and adding cores or `-ols` brings limited benefit in that regime.

To reduce DDR traffic:

- Use `-mos` to partition weights across cores so each core's tile shrinks.
- Use `-multicast-weights` (default-on) so shared weights are read once and broadcast over the NoC. See §5.
- Apply VTCM-resident optimization patterns (channel alignment, fusion, residual budgeting) so the working set does not churn back to DDR.
- **Hardware weight compression** (`-mxfp6-matmul`): stores constant MatMul weights in DDR in a compressed format; the hardware decompresses during compute. Reduces DDR footprint and read traffic for MatMul-heavy workloads at the cost of a small additional compute step. Effective when DDR bandwidth is the bottleneck, not compute.

---

## 5. NoC and Multicast

Cloud AI 100 includes multiple high-speed on-chip networks (NoCs):

- A compute NoC connecting cores.
- A memory NoC connecting cores to DDR.
- A configuration/synchronization fabric.

### 5.1 Multicast — The High-Speed Inter-Core Transport

**Multicast** is the NoC mechanism that broadcasts a single source tensor to multiple destination cores in one transaction. It is what makes channel-wise (`-mos`) splitting practical: when output channels are partitioned across cores, all cores typically need the same input activations, and many models reuse the same weight tiles across cores. Multicast issues one read and fans the data out to N cores, replacing N independent DDR reads with one — proportional savings in DDR bandwidth and latency. Multicast replacement is enabled by default; disabling it (or letting it miss opportunities through poor graph shape) is a common cause of avoidable DDR pressure.

**Multicast IDs (MCIDs) are a limited hardware resource.** Each live multicast group consumes one MCID. When the compiler runs out — typically because too many simultaneous multicast destinations are live across the schedule — it falls back to per-core unicast copies, which restore the N× DDR cost the multicast was avoiding. The compiler tries to preallocate small multicast targets to keep MCIDs available for the larger ones.

**Authoring rule:** Shared inputs to multiple branches must be a single graph node. Duplicated Reshape, Identity, or Transpose ops before a branch produce distinct tensors that defeat multicast and force per-core copies — see `cloud-ai-performance-optimization/references/graph-optimization.md` for the full set of graph checks and multicast-friendly patterns.

### 5.2 Implications

- Excessive cross-core transfers (poor partitioning, duplicated pre-branch ops) increase NoC traffic and hurt performance.
- Diagnostic signals: `Not VTCM` multicast count > 0 in compile.log means multicast destinations spilled out of VTCM; `aicmulticastvtcm` in opstats top-ops is the runtime signature of multicast in flight.

---

## 6. Execution Model

From the model’s perspective, execution proceeds in phases:

1. **Download / Load**
   - Host sends QPC binary and weights to device DDR.
2. **Activation Staging**
   - Inputs are transferred from host to device (PCIe DMA) and staged in device memory.
3. **Compute**
   - Cores execute assigned subgraphs, tiling activations and weights into VTCM as needed.
4. **Upload / Results**
   - Outputs are transferred back to host.

### 6.1 Tiling and Splitting

The compiler uses tiling to make tensors fit into VTCM and to distribute work:

- **Activation tiling**: splits spatial or channel dimensions into tiles that fit in VTCM.
- **Weight splitting**: partitions weights across cores, controlled by `-mos` and related options.

**Implication for model design:**

- Channel counts aligned to hardware tile sizes (for example, multiples of 32) reduce padding and tile fragmentation.
- Extremely narrow or highly irregular shapes increase tiling overhead.
- Large residual/skip connections that require both branches resident simultaneously in VTCM can force additional tiling or spills.

---

## 7. MDP — Multi-Device Partitioning and Tensor Slicing

**MDP (Multi-Device Partitioning)** splits a model across multiple SoCs so that each SoC holds and computes a slice of the model. This is the primary mechanism for deploying models too large to fit on a single SoC, and for scaling throughput across multiple SoCs.

**TS (Tensor Slicing)** is the specific parallelism strategy used within MDP: weights and activations are sliced across the participating SoCs, with each SoC computing its slice in parallel. The compiler handles the slice/gather communication automatically via the p2p connections declared in the MDP config.

Within a tensor slice, the per-core weight partitioning principle from Tenet 1 still applies (`-mos`). MDP/TS adds a second axis: `-mdts-mos` partitions the weights of a single op across slices, complementing `-mos` which partitions across cores within a slice. Both default to a compiler heuristic.

When PCIe P2P bandwidth between SoCs is the bottleneck, `-allow-mxint8-mdp-io` compresses inter-SoC tensor traffic. This reduces link utilization at the cost of potential accuracy impact — apply only after confirming inter-device DMA is the limiting factor.

**Scope:** MDP/TS only applies on multi-SoC SKUs (AI 100 Ultra — 4 SoCs per card). The single-SoC SKUs (Standard, Pro) cannot use TS; the model must fit in one SoC's DDR.

### 7.1 Terminology

| Term | Meaning |
|---|---|
| **SoC** | One AI 100 die — the unit of MDP partitioning. Also called a **QID** (device ID) in runtime and config files. |
| **Ultra card** | 4 SoCs on one PCIe card. |
| **TS-N** | Tensor slicing across N SoCs (e.g. TS4 = 4 SoCs = 1 Ultra card). Powers of 2 only: TS1, TS2, TS4, TS8, TS16. |
| **MDP partition config** | JSON file passed to `qaic-compile` via `-mdp-load-partition-config` declaring which SoCs participate and how many cores each uses. |
| **ProgramGroup** | QAIC runtime feature that lets multiple single-device QPCs share the same reserved SoC, running sequentially. **Single-device only** — does not support MDP/TS QPCs. For MDP/TS pipelines there is no NSP-sharing equivalent; device requirements add across QPCs. |

### 7.2 Memory-Driven Sizing Principle

The minimum TS level for a model is set by memory: the model's weights + kv_cache + activation headroom must fit in `num_SoCs × per-SoC DDR` (32 GB on Pro/Ultra, 16 GB on Standard — see §1 SKU table). Higher TS levels can be selected to add compute parallelism for better throughput, at the cost of more devices.

**TS16 (16 SoCs = 4 Ultra cards) is the maximum supported configuration for a single QPC.** A model that does not fit at TS16 with the available weight precision needs additional compression or cannot be deployed.

### 7.3 Multi-Module Pipelines

For pipelines composed of multiple modules (e.g. encoder + prefill + decoder), each module compiles to a separate QPC with its own TS level. TS QPCs cannot share NSPs (there is no equivalent of ProgramGroup for MDP/TS), and QPC load/unload is too slow to amortize over per-inference latency — so all QPCs in a pipeline stay loaded concurrently. SoC requirements therefore **add across modules**:

```
total_SoCs = sum(ts_per_module)
```

The pipeline executes module-by-module in time, but the deployment must reserve the sum of all per-module SoC counts simultaneously.

---

## 8. Operator-to-Hardware Routing Reference

ONNX operator type determines its Cloud AI 100 compute path and the associated
eligibility constraints.

### 8.1 32-Channel Tile Alignment

The hardware processes activations and weights in **32-channel groups**. Several model-design rules follow from this:

- Channels are padded up to multiples of 32. A 33-channel layer becomes 64-channel for compute; design at 32-channel boundaries (or at minimum multiples of 8) to avoid waste.
- **Convolution tile**: 8×8 spatial × 32 channels.
- **MatMul tile** along the contracted dimension: 1×64 (INT8) or 1×32 (FP16), with 32-channel groups.
- Layout-conversion ops (`AICConvertToD32` / `AICConvertFromD32` in opstats) appear at HMX boundaries; most fuse into an adjacent Reshape or boundary Transpose at no cost. Chains of layout churn at every layer boundary do not.

### 8.2 HMX — Matrix Operations

HMX handles Conv (including depthwise and grouped), MatMul, and Gemm. Eligibility:

**Precision** — HMX accepts **FP16 or INT8 only**. The model must run in one of these precisions (via `-convert-to-fp16` or `-convert-to-quantize`). FP32 ops do not map to HMX; a whole-network FP32 compile produces inference latency in seconds per frame.

**Convolution geometry** — Conv falls back from HMX to HVX (significantly slower) when:
- `stride > kernel_size` AND `padding > 0`
- x/y dilation values differ
- dilation is greater than 2
- dilation is greater than 1 and either spatial stride is greater than 1

For performance-critical 2D Conv, keep dilation equal on x/y axes and <=2; if dilation is >1, keep stride exactly 1. 3D Conv dilation is a separate compile-compatibility issue; see the ONNX operator support table.

**Diagnosing HMX misses** — `AIC non-HMX MAC estimate` in `qaic-compile -aic-perf-metrics` should be zero. Non-zero means at least one Conv or MatMul did not map to HMX. A warning like `dilation too large or dilation x/y dims don't match` points to this Conv geometry fallback. Opstats may also show `aicconvhvx` plus `aicinterleaveconv`/`aicdeinterleaveconv`; the interleave/deinterleave pattern means the compiler applied dilation handling, not that the main Conv reached HMX. See `cloud-ai-qaic-compile` skill §6.

### 8.3 HVX — Vector Operations and Performance Principles

HVX is the per-core SIMD vector engine. It handles non-matrix vector work: element-wise math, activation functions, normalization, vectorized reductions, indexing, layout transforms, and format conversion. HVX is efficient when operands fit in VTCM. (Inherently sequential portions of ops — e.g. NMS box-suppression loop, prefix-sum scan — run on the scalar unit; see §8.4.) Three broad principles govern HVX op performance — they apply across many operator types, not only the named examples.

**Last-axis preference.** Reduce-style and sort-style ops (e.g. Softmax, ReduceMean/Sum/Max, ArgMax/ArgMin, TopK) take optimized vectorized paths when operating on the last axis. Operating on a non-last axis degrades to a slower path; the standard fix is to wrap the op as Transpose→Op→Transpose in the export script so it sees the reduced dimension as the last one.

**VTCM-resident indexing.** Indexing ops (Gather, GatherElements, scatter-style ops) are fast when their operands fit in VTCM; the hardware-accelerated gather path additionally requires index count ≤ 64. Larger or off-VTCM operands fall back to an unaccelerated memory copy. If the indexing expresses a sparse or conditional selection, restructure as a masked multiply (element-wise multiply by a 0/1 weight) to stay on the vectorized path.

**Layout-conversion fusion.** A single NCHW↔NHWC Transpose at a model boundary fuses into the adjacent layout conversion at no cost. Chains of back-and-forth transposes do not fuse; they add sync overhead and block fusion of the ops between them. The same applies to Reshape inserts that exist only to satisfy intermediate shape requirements.

### 8.4 Sequential and Compile-Time-Limited Operations

Some ops are inherently sequential or are gated at compile time:

**NonMaxSuppression** — IoU scoring and score filtering use HVX. Per-class box suppression is a sequential loop and does not parallelize across cores; it becomes latency-dominant at high class or box counts. Move off-device or replace with a batched approximation when it dominates.

**CumSum** — prefix sum is sequential. Supported configurations: axis 0 (all modes), and axis 1 / 2 for 2D/3D inputs without `exclusive` or `reverse`. All other combinations fail at compile time.

**Dynamic control flow** (Loop, If with runtime conditions) — not supported. Models must have fully static control flow.

**Shape-affecting ops require constant shape inputs.** Reshape (`shape`), Slice (`starts`/`ends`/`axes`/`steps`), Pad (`pads`), Tile (`repeats`), TopK (`k`), OneHot (`depth`), and Resize (`scales`/`sizes`) all require their shape input to be a Constant at compile time. Runtime-computed shapes fail to compile; bake them as constants during export. The full operator-support table lives in `cloud-ai-qaic-compile/references/cloud_ai_onnx_operator_support.md`.

### 8.5 HMX→HVX Fusion

When an HMX op is immediately followed by an HVX op (activation, Add, scale/shift), the two units pipeline at tile granularity — see §3.1 for the full picture and the opstats signature of broken fusion. Any node inserted between an HMX op and its HVX consumer (Cast, Quantize, Reshape, extra element-wise) breaks the pipeline.
