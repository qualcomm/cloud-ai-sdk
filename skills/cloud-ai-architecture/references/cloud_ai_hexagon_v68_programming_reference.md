# Cloud AI 100 – Hexagon v68 Programming Reference

## Overview

This reference provides practical guidance for **writing custom operators and kernels** for Qualcomm Cloud AI 100 accelerators using the Hexagon v68 instruction set and HVX (Hexagon Vector Extensions). It complements the architectural overview in `cloud_ai_hexagon_v68_dsp_architecture.md` by focusing on the programming model, instruction patterns, and code examples.

**When to use this guide:**
- Implementing custom HVX kernels for activation functions, normalization, or preprocessing
- Optimizing memory-bound operations with vector intrinsics
- Writing assembly-level optimizations for performance-critical paths
- Understanding register conventions and calling conventions for the Hexagon toolchain
- Debugging performance issues at the instruction level

**Prerequisites:**
- C/C++ programming experience
- Basic understanding of SIMD/vector programming concepts
- Familiarity with the Cloud AI 100 architecture (see `cloud_ai_hexagon_v68_dsp_architecture.md`)

---

## 1. Scalar Programming Model

### 1.1 Register Architecture

Hexagon v68 provides a load-store RISC-like architecture with the following register files:

| Register Type | Count | Width | Naming | Usage |
|--------------|-------|-------|--------|-------|
| General Purpose (GPR) | 32 | 32-bit | `R0-R31` | Integer arithmetic, addresses, scalars |
| Predicate | 4 | 8-bit | `P0-P3` | Conditional execution (each bit controls one element) |
| Control | Multiple | 32-bit | `C0-C31` | System control, loop counters, thread state |
| Vector (HVX) | 32 | 1024-bit | `V0-V31` | Vector operations (128 bytes each) |
| Vector Predicate | 4 | 128-bit | `Q0-Q3` | Vector conditional execution (1 bit per byte) |

**64-bit Register Pairs:**
- GPRs can be paired for 64-bit operations: `R1:0`, `R3:2`, ..., `R31:30`
- Even-numbered register is low-order, odd is high-order
- Example: `R1:0 = MEMD(R10+#0)` loads 64 bits

### 1.2 Calling Convention (C ABI)

When interfacing C/C++ with assembly or understanding intrinsics:

| Register(s) | Purpose | Preserved? |
|------------|---------|-----------|
| `R0-R5` | Function arguments 1-6 | No (caller-saved) |
| `R0-R1` | Return value (32-bit in R0, 64-bit in R1:0) | No |
| `R16-R27` | Saved registers | Yes (callee-saved) |
| `R28` | Predicate register save | Yes |
| `R29` | Stack pointer (SP) | Yes |
| `R30` | Frame pointer (FP) | Yes |
| `R31` | Link register (LR) | No |

**HVX Calling Convention:**
- `V0-V15` are caller-saved (temporary)
- `V16-V31` are callee-saved (must preserve if used)
- Pass vector arguments via memory (stack or VTCM)

### 1.3 Core Instruction Categories

#### Arithmetic Instructions
```assembly
# 32-bit addition
R0 = ADD(R1, R2)
R0 = ADD(R1, #imm)

# 64-bit addition with register pair
R1:0 = ADD(R3:2, R5:4)

# Multiply-accumulate (MAC)
R0 += MPY(R1, R2)        # R0 = R0 + (R1 * R2)

# Saturating arithmetic
R0 = ADD(R1, R2):sat     # Saturate to 32-bit range
```

#### Load/Store Instructions
```assembly
# Load word (32-bit)
R0 = MEMW(R10 + #offset)

# Load doubleword (64-bit)
R1:0 = MEMD(R10 + #offset)

# Store word with auto-increment
MEMW(R10++#4) = R0

# Aligned vector load (HVX)
V0 = VMEM(R10 + #0)

# Unaligned vector load
V0 = VMEMU(R10 + #0)

# Conditional store (predicated)
IF (P0) MEMW(R10) = R0
```

#### Predicate and Conditional Execution
```assembly
# Compare and generate predicate
P0 = CMP.GT(R1, R2)      # P0 = (R1 > R2)
P0 = CMP.EQ(R1, #0)      # P0 = (R1 == 0)

# Conditional execution
IF (P0) R0 = ADD(R1, R2)
IF (!P0) R0 = #0

# Predicate operations
P0 = AND(P1, P2)
P0 = OR(P1, !P2)
```

#### Branch and Control Flow
```assembly
# Unconditional jump
JUMP label

# Conditional branch
IF (P0) JUMP:t label     # :t = likely taken hint

# Hardware loop (0-cycle overhead)
LOOP0(label, R0)         # Loop R0 iterations
{
  # Loop body
}:endloop0

# Nested loops (LOOP0, LOOP1)
LOOP1(outer, R1)
  LOOP0(inner, R0)
    # Inner loop body
  :endloop0
:endloop1
```

### 1.4 VLIW Packet Syntax

Instructions are grouped into packets (up to 4 instructions executed in parallel):

```assembly
{
  R0 = ADD(R1, R2)       # ALU
  R3 = MEMW(R10 + #0)    # Load
  V0 = VADD(V1, V2)      # Vector op
  P0 = CMP.GT(R4, R5)    # Compare
}
```

**Packet Rules:**
- Instructions in a packet must be independent (no data dependencies)
- Enclosed in `{ }` braces
- Compiler automatically generates packets; manual packets are rarely needed
- Use `:mem_noshuf` to prevent memory reordering within packets

---

## 2. HVX Vector Programming

### 2.1 HVX Overview

**HVX (Hexagon Vector Extensions)** provides 1024-bit SIMD operations:
- **32 vector registers** (`V0-V31`), each 1024 bits (128 bytes)
- **4 vector predicate registers** (`Q0-Q3`), 128 bits (1 bit per byte)
- Supports byte, half-word (16-bit), and word (32-bit) element types
- Typical throughput: 512 INT8 MACs/cycle, 256 FP16 MACs/cycle

**Threading context:** Hexagon v68 provides 6 hardware threads (see `cloud_ai_hexagon_v68_dsp_architecture.md`), but QAic Custom Ops exposes **4 HVX threads** (threadId 0–3) to custom kernels. The remaining 2 threads handle HMX (matrix engine) operations and synchronization.

### 2.2 Vector Data Types and Layout

| Type | Elements per Vector | Element Size | HVX Type Suffix |
|------|-------------------|--------------|----------------|
| Byte | 128 | 8-bit | `.b` or `.ub` |
| Half-word | 64 | 16-bit | `.h` or `.uh` |
| Word | 32 | 32-bit | `.w` or `.uw` |

**Signed vs. Unsigned:**
- `.ub`, `.uh`, `.uw` = unsigned
- `.b`, `.h`, `.w` = signed
- Matters for comparisons, saturation, and some arithmetic operations

### 2.3 HVX Intrinsics in C/C++

#### Header Files and Includes

HVX intrinsics are accessed via the following headers:

| Header | Path | Contents |
|--------|------|----------|
| **QAic SDK Headers (custom ops)** | | |
| `QAicHexagonTypes.h` | `/opt/qti-aic/dev/inc/jit/QAicHexagonTypes.h` | `HVX_VectorTyped<T>` and all typed aliases; `HVX_Vector_UN`; `HVX_VectorPairTyped<T>` and pair aliases; `HVX_VectorTuple<T,N>` and tuple aliases; C++ operator overloads for arithmetic, bitwise, and comparison on all typed vector types |
| `QAicHexagonHVX.h` | `/opt/qti-aic/dev/inc/jit/QAicHexagonHVX.h` | `LoadHVX()`, `StoreHVX()` (aligned, predicated, pair, tuple overloads), `StoreUnalignedHVX()` (with optional rotate), `LoadHVXPred()`, `StoreHVXPred()`; alignment helpers (`alignTo`, `alignDown`, `divideCeil`); math approximations (`exp_vector_sf`, `tanh_vector_sf`, `erf_vector_sf`, `floor_vector_sf`, `div_vector_sf`, `div_vector_hf`, `element_rsqrt_impl_f`); utility helpers (`splatFirstNBytes`, `splatNBytes`, `element_mul_impl_i32`, `element_neg_impl_*`, `qaic_cvt_sf_value_to_hf_vector`) |
| **Hexagon Toolchain Headers** | | |
| `hvx_hexagon_protos.h` | `/opt/qti-aic/dev/hexagon_tools-19.0.06/target/hexagon/include/` | All `Q6_*` HVX intrinsic declarations (**536 intrinsics**) |
| `hexagon_protos.h` | `/opt/qti-aic/dev/hexagon_tools-19.0.06/target/hexagon/include/` | Scalar Hexagon intrinsics (**927 intrinsics**) |
| `hexagon_types.h` | `/opt/qti-aic/dev/hexagon_tools-19.0.06/target/hexagon/include/` | Basic `HVX_Vector`, `HVX_VectorPair`, `HVX_VectorPred` type definitions |
| `hmx_hexagon_protos.h` | `/opt/qti-aic/dev/hexagon_tools-19.0.06/target/hexagon/include/` | HMX (matrix engine) intrinsics |

**For custom ops development**, include QAic SDK headers:

```cpp
#include "CustomOpAICInterface.h"
#include "jit/QAicHexagonHVX.h"      // Pulls in QAicHexagonTypes.h
#include <hexagon_protos.h>          // For scalar intrinsics
```

#### QAic SDK typed vector types (`QAicHexagonTypes.h`)

`QAicHexagonTypes.h` provides `HVX_VectorTyped<T>` — a union over `HVX_Vector` that exposes the raw vector, a typed C++ ext_vector, and a typed array. All typed aliases implicitly convert to/from `HVX_Vector`, so they can be passed directly to any `Q6_*` intrinsic.

**Single-vector aliases:**

| Alias | Element type | Elements |
|-------|-------------|---------|
| `HVX_VectorB` | `int8_t` | 128 |
| `HVX_VectorUB` | `uint8_t` | 128 |
| `HVX_VectorH` | `int16_t` | 64 |
| `HVX_VectorUH` | `uint16_t` | 64 |
| `HVX_VectorW` | `int32_t` | 32 |
| `HVX_VectorUW` | `uint32_t` | 32 |
| `HVX_VectorSF` | `float` | 32 |
| `HVX_VectorHF` | `__fp16` | 64 |

**Vector pair aliases** (`HVX_VectorPairTyped<T>`, wraps `HVX_VectorPair`):

`HVX_VectorWB`, `HVX_VectorWUB`, `HVX_VectorWH`, `HVX_VectorWUH`, `HVX_VectorWW`, `HVX_VectorWUW`, `HVX_VectorWSF`, `HVX_VectorWHF` — each provides `.lo()` and `.hi()` accessors returning the typed single-vector half.

**Vector tuple aliases** (`HVX_VectorTuple<T,N>`):

`HVX_Vector2Tuple`, `HVX_Vector4Tuple`, `HVX_Vector8Tuple`, `HVX_Vector32Tuple`, `HVX_Vector64Tuple` (raw `HVX_Vector`); typed variants for 2- and 4-element tuples: `HVX_VectorB2Tuple`, `HVX_VectorUB4Tuple`, `HVX_VectorSF4Tuple`, etc.

**C++ operator overloads** — available on all typed vector types (where hardware supports it):

| Operators | Types | Notes |
|-----------|-------|-------|
| `+`, `-`, `*`, `&`, `\|`, `^`, `<<`, `>>` | All integer typed vectors | Maps to compiler ext_vector builtins |
| `+`, `-`, `*` | `HVX_VectorSF`, `HVX_VectorHF` | Maps to explicit `Q6_Vsf_vadd_VsfVsf` / `vsub` / `vmpy` calls (not compiler builtins — avoids QF intermediate) |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | All typed vectors | Returns `HVX_VectorPred` |
| `/`, `%`, `<<`, `>>` (vector-vector) | `HVX_VectorSF`, `HVX_VectorHF` | `= delete` — not supported |

**Splat constructor:** `HVX_VectorUB(val)` broadcasts a scalar to all lanes using the appropriate `Q6_V*_vsplat_R` intrinsic.

**`HVX_Vector_UN`:** `typedef HVX_Vector HVX_Vector_UN __attribute__((aligned(1)))` — cast a pointer to this type to emit unaligned `vmemu` load/store instructions.

#### QAic SDK load/store and math helpers (`QAicHexagonHVX.h`)

**Load/store overloads** — `LoadHVX` and `StoreHVX` are templated and overloaded for `HVX_Vector`, `HVX_VectorPair`, and `HVX_VectorTuple<V,N>`, with optional `numLeadingBytes` or `HVX_VectorPred` for partial/tail stores. All aligned variants assert 128-byte alignment in debug builds.

**Unaligned stores** — `StoreUnalignedHVX` handles arbitrary pointer alignment using two predicated stores with computed masks. Accepts an optional `rotateAmount` to pre-rotate the vector before storing.

**Math approximations** (device-side, `extern "C"` — call from AIC kernel):

| Function | Signature | Notes |
|----------|-----------|-------|
| `exp_vector_sf` | `HVX_VectorSF(HVX_VectorSF)` | Polynomial approximation |
| `tanh_vector_sf` | `HVX_VectorSF(HVX_VectorSF)` | Polynomial approximation |
| `erf_vector_sf` | `HVX_VectorSF(HVX_VectorSF)` | Polynomial approximation |
| `floor_vector_sf` | `HVX_VectorSF(HVX_VectorSF, HVX_VectorW*)` | Optional int output |
| `div_vector_sf` | `HVX_VectorSF(HVX_VectorSF, HVX_VectorSF)` | Reciprocal approximation |
| `div_vector_hf` | `HVX_VectorHF(HVX_VectorHF, HVX_VectorHF)` | Reciprocal approximation |
| `element_rsqrt_impl_f` | `HVX_Vector(HVX_VectorSF)` | Reciprocal sqrt approximation |

**Utility helpers:**

| Helper | Purpose |
|--------|---------|
| `splatFirstNBytes(n, vec)` | Broadcast first `n` bytes of `vec` across all 128 bytes (n must be power of 2) |
| `splatNBytes(n, vec, src)` | Broadcast the `src`-th group of `n` bytes across all 128 bytes |
| `element_mul_impl_i32(a, b)` | 32-bit integer element-wise multiply (uses `vmpye`+`vmpyoacc` pair) |
| `element_neg_impl_hf/f/i32` | Negate all elements (XOR sign bit for float; NOT+1 for int32) |
| `qaic_cvt_sf_value_to_hf_vector(f)` | Broadcast a scalar float32 as float16 to all 64 lanes |
| `alignTo(x, m)` / `alignDown(x, m)` | Round up/down to multiple of `m` |
| `divideCeil(x, m)` | Integer divide rounding up |

#### Basic Vector Load/Store

**Using QAic SDK wrappers** (recommended for custom ops):

```cpp
#include "jit/QAicHexagonHVX.h"

// Aligned load (address must be 128-byte aligned)
HVX_Vector v0 = LoadHVX(aligned_ptr);

// Aligned store
StoreHVX(aligned_ptr, v0);

// Predicated store (writes only first n bytes)
StoreHVX(ptr, v, numBytes);

// Unaligned access (stride not multiple of 128)
HVX_Vector v_unaligned = *((HVX_Vector_UN*)unaligned_ptr);
*((HVX_Vector_UN*)unaligned_ptr) = v_unaligned;

// Or use StoreUnalignedHVX
StoreUnalignedHVX(unaligned_ptr, v, numBytes);
```

**Using raw intrinsics** (direct Hexagon toolchain):

```cpp
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

// Aligned load/store
HVX_Vector v0 = *((HVX_Vector*)aligned_ptr);
*((HVX_Vector*)aligned_ptr) = v0;

// Note: Q6_V_vldu_A and Q6_V_vstu_A are NOT available in Hexagon v68
// Use HVX_Vector_UN from QAicHexagonTypes.h for unaligned access
```

#### Vector Arithmetic
```c
// Vector addition (32 x word elements)
HVX_Vector vadd = Q6_Vw_vadd_VwVw(v1, v2);

// Vector multiply (16-bit elements, return 32-bit products as vector pair)
HVX_VectorPair vmpy = Q6_Ww_vmpy_VhVh(v1, v2);

// Vector multiply-accumulate (pair accumulator)
HVX_VectorPair vmacc = Q6_Ww_vmpyacc_WwVhVh(vmpy, v1, v2);  // vmacc += v1 * v2

// Vector saturating add (byte elements)
HVX_Vector vaddsat = Q6_Vb_vadd_VbVb_sat(v1, v2);

// Vector max/min
HVX_Vector vmax = Q6_Vb_vmax_VbVb(v1, v2);
```

#### Vector Comparison and Predicates
```c
// Compare (generates predicate mask)
HVX_VectorPred q0 = Q6_Q_vcmp_gt_VbVb(v1, v2);  // q0 = (v1 > v2)

// Conditional select (blend)
HVX_Vector vsel = Q6_V_vmux_QVV(q0, vtrue, vfalse);  // if(q0) vtrue else vfalse

// Predicate operations
HVX_VectorPred q_and = Q6_Q_and_QQ(q1, q2);
HVX_VectorPred q_or = Q6_Q_or_QQ(q1, q2);
HVX_VectorPred q_not = Q6_Q_not_Q(q1);
```

#### Vector Permutations and Shuffles

**Shuffle intrinsics that exist in Hexagon v68 (`hvx_hexagon_protos.h`):**

| Intrinsic | Signature | Purpose |
|-----------|-----------|---------|
| `Q6_Vb_vshuffe_VbVb` | `(HVX_Vector Vu, HVX_Vector Vv)` | Extract even bytes from two byte vectors |
| `Q6_Vb_vshuffo_VbVb` | `(HVX_Vector Vu, HVX_Vector Vv)` | Extract odd bytes from two byte vectors |
| `Q6_Vh_vshuffe_VhVh` | `(HVX_Vector Vu, HVX_Vector Vv)` | Extract even halfwords from two halfword vectors |
| `Q6_Vh_vshuffo_VhVh` | `(HVX_Vector Vu, HVX_Vector Vv)` | Extract odd halfwords from two halfword vectors |
| `Q6_Wb_vshuffoe_VbVb` | `(HVX_Vector Vu, HVX_Vector Vv)` | Byte shuffle to vector pair (even+odd in one call) |
| `Q6_Wh_vshuffoe_VhVh` | `(HVX_Vector Vu, HVX_Vector Vv)` | Halfword shuffle to vector pair (even+odd in one call) |
| `Q6_W_vshuff_VVR` | `(HVX_Vector Vu, HVX_Vector Vv, Word32 Rt)` | General interleave with shift — preferred for transpose |
| `Q6_W_vdeal_VVR` | `(HVX_Vector Vu, HVX_Vector Vv, Word32 Rt)` | General deinterleave with shift |

> **Non-existent forms:** `Q6_Vuh_vshuffe_VubVub`, `Q6_Vuh_vshuffo_VubVub`, `Q6_Vuw_vshuffe_VuhVuh`, `Q6_Vuw_vshuffo_VuhVuh`, `Q6_Vud_vshuffe_VuwVuw`, and `Q6_Vud_vshuffo_VuwVuw` do **not** exist in the Hexagon v68 toolchain. Use the element-matched forms above.

```c
// Interleave bytes (zip/unzip operations)
HVX_VectorPair vzip = Q6_W_vshuff_VVR(v1, v2, -1);  // Zip at byte granularity
HVX_Vector v_lo = Q6_V_lo_W(vzip);
HVX_Vector v_hi = Q6_V_hi_W(vzip);

// Even/odd byte extraction (separate vectors)
HVX_Vector v_even = Q6_Vb_vshuffe_VbVb(v1, v2);
HVX_Vector v_odd  = Q6_Vb_vshuffo_VbVb(v1, v2);

// Rotate/shift vector by bytes
HVX_Vector vrot = Q6_V_vror_VR(v0, rot_amt);  // Rotate right

// Vector align (extract 128 bytes from 256-byte concatenation)
HVX_Vector valign = Q6_V_valign_VVR(v_hi, v_lo, offset);

// Table lookup (VLUT)
HVX_Vector vlut = Q6_Vb_vlut32_VbVbR(v_indices, v_table, 0);
```

**16×16 byte transpose using `Q6_W_vshuff_VVR`:**

The standard HVX pattern for a 16×16 byte matrix transpose uses 4 stages of `Q6_W_vshuff_VVR` with shift amounts -1, -2, -4, -8. Each call produces a vector pair (even+odd interleave) in one instruction — more efficient than separate `vshuffe`/`vshuffo` calls:

```c
// Stage 1: interleave at byte granularity (shift = -1)
HVX_VectorPair w0 = Q6_W_vshuff_VVR(v1,  v0,  -1);
HVX_VectorPair w1 = Q6_W_vshuff_VVR(v3,  v2,  -1);
HVX_VectorPair w2 = Q6_W_vshuff_VVR(v5,  v4,  -1);
HVX_VectorPair w3 = Q6_W_vshuff_VVR(v7,  v6,  -1);
HVX_VectorPair w4 = Q6_W_vshuff_VVR(v9,  v8,  -1);
HVX_VectorPair w5 = Q6_W_vshuff_VVR(v11, v10, -1);
HVX_VectorPair w6 = Q6_W_vshuff_VVR(v13, v12, -1);
HVX_VectorPair w7 = Q6_W_vshuff_VVR(v15, v14, -1);

// Stage 2: interleave at 2-byte granularity (shift = -2)
HVX_VectorPair x0 = Q6_W_vshuff_VVR(Q6_V_hi_W(w1), Q6_V_hi_W(w0), -2);
HVX_VectorPair x1 = Q6_W_vshuff_VVR(Q6_V_lo_W(w1), Q6_V_lo_W(w0), -2);
HVX_VectorPair x2 = Q6_W_vshuff_VVR(Q6_V_hi_W(w3), Q6_V_hi_W(w2), -2);
HVX_VectorPair x3 = Q6_W_vshuff_VVR(Q6_V_lo_W(w3), Q6_V_lo_W(w2), -2);
HVX_VectorPair x4 = Q6_W_vshuff_VVR(Q6_V_hi_W(w5), Q6_V_hi_W(w4), -2);
HVX_VectorPair x5 = Q6_W_vshuff_VVR(Q6_V_lo_W(w5), Q6_V_lo_W(w4), -2);
HVX_VectorPair x6 = Q6_W_vshuff_VVR(Q6_V_hi_W(w7), Q6_V_hi_W(w6), -2);
HVX_VectorPair x7 = Q6_W_vshuff_VVR(Q6_V_lo_W(w7), Q6_V_lo_W(w6), -2);

// Stage 3: shift = -4, Stage 4: shift = -8 (same pattern)
// After 4 stages the 16 output vectors hold the transposed rows.
```

#### Horizontal Operations (Reductions)
```c
// Vector reduce: sum all elements (not a single instruction, requires multiple steps)
// Example: sum 32 x word elements
HVX_Vector v_sum = v0;
v_sum = Q6_Vw_vadd_VwVw(v_sum, Q6_V_vror_VR(v_sum, 64));
v_sum = Q6_Vw_vadd_VwVw(v_sum, Q6_V_vror_VR(v_sum, 32));
v_sum = Q6_Vw_vadd_VwVw(v_sum, Q6_V_vror_VR(v_sum, 16));
v_sum = Q6_Vw_vadd_VwVw(v_sum, Q6_V_vror_VR(v_sum, 8));
v_sum = Q6_Vw_vadd_VwVw(v_sum, Q6_V_vror_VR(v_sum, 4));
int32_t result = Q6_R_vextract_VR(v_sum, 0);
```

### 2.4 Intrinsic Naming Convention

HVX intrinsics follow the pattern: `Q6_<return>_<op>_<args>`

**Return types:**
- `V` = HVX_Vector
- `W` = HVX_VectorPair (two vectors)
- `Q` = HVX_VectorPred
- `R` = scalar (32-bit GPR)

**Element types in args:**
- `Vb` = vector of bytes
- `Vh` = vector of half-words
- `Vw` = vector of words
- `Vub`, `Vuh`, `Vuw` = unsigned variants

**Example:** `Q6_Vw_vadd_VwVw(v1, v2)`
- Returns: `Vw` (vector of words)
- Operation: `vadd` (vector add)
- Arguments: `VwVw` (two vectors of words)

---

## 3. Common Programming Patterns

### 3.1 Element-wise Activation (ReLU)

**Scalar C:**
```c
for (int i = 0; i < n; i++) {
  output[i] = (input[i] > 0) ? input[i] : 0;
}
```

**HVX Intrinsics (32-bit float):**
```c
void relu_hvx(float* output, const float* input, int n) {
  HVX_Vector vzero = Q6_V_vzero();
  
  for (int i = 0; i < n; i += 32) {  // 32 floats per vector
    HVX_Vector vin = *((HVX_Vector*)(input + i));
    
    // Compare with zero
    HVX_VectorPred qmask = Q6_Q_vcmp_gt_VsfVsf(vin, vzero);
    
    // Select: if (vin > 0) vin else 0
    HVX_Vector vout = Q6_V_vmux_QVV(qmask, vin, vzero);
    
    *((HVX_Vector*)(output + i)) = vout;
  }
}
```

### 3.2 Vectorized Dot Product (INT16)

```c
int32_t dot_product_hvx(const int16_t* a, const int16_t* b, int n) {
  // Q6_Ww_vmpy_VhVh returns a VectorPair: 64 x int16 → 32 x int32 (lo) + 32 x int32 (hi)
  HVX_VectorPair vacc = {0};

  for (int i = 0; i < n; i += 64) {  // 64 x int16 per vector
    HVX_Vector va = *((HVX_Vector*)(a + i));
    HVX_Vector vb = *((HVX_Vector*)(b + i));

    // Multiply-accumulate: vacc += va * vb (pair accumulator, 32-bit products)
    vacc = Q6_Ww_vmpyacc_WwVhVh(vacc, va, vb);
  }

  // Sum the two halves of the pair, then reduce the resulting vector
  HVX_Vector vsum = Q6_Vw_vadd_VwVw(Q6_V_lo_W(vacc), Q6_V_hi_W(vacc));
  vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 64));
  vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 32));
  vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 16));
  vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 8));
  vsum = Q6_Vw_vadd_VwVw(vsum, Q6_V_vror_VR(vsum, 4));

  return Q6_R_vextract_VR(vsum, 0);
}
```

### 3.3 Memory-bound Vectorized Copy

```c
void memcpy_hvx(void* dst, const void* src, size_t n) {
  HVX_Vector* vdst = (HVX_Vector*)dst;
  const HVX_Vector* vsrc = (const HVX_Vector*)src;
  int nvec = n / 128;  // 128 bytes per vector
  
  for (int i = 0; i < nvec; i++) {
    vdst[i] = vsrc[i];
  }
}
```

**Unaligned variant:**
```c
void memcpy_hvx_unaligned(void* dst, const void* src, size_t n) {
  for (int i = 0; i < n; i += 128) {
    HVX_Vector v = *((HVX_Vector_UN*)(src + i));  // Use HVX_Vector_UN for unaligned
    *((HVX_Vector_UN*)(dst + i)) = v;
  }
}
```

### 3.4 Scatter/Gather Operations

**Gather (load from indexed addresses):**
```c
// Gather 32 x int32 values using 32 x int32 indices
// Note: HVX doesn't have native gather; this is a scalar loop fallback
void gather_scalar(int32_t* output, const int32_t* base, const int32_t* indices, int n) {
  for (int i = 0; i < n; i++) {
    output[i] = base[indices[i]];
  }
}

// For small index ranges, use VLUT (vector lookup table)
// VLUT supports 32-byte tables with byte indices
```

**Scatter (store to indexed addresses):**
```c
// Scatter 32 x int32 values using 32 x int32 indices
// No native HVX scatter; use scalar loop or reorganize data layout
void scatter_scalar(int32_t* base, const int32_t* indices, const int32_t* values, int n) {
  for (int i = 0; i < n; i++) {
    base[indices[i]] = values[i];
  }
}
```

### 3.5 Softmax (Vectorized Max and Exponential)

```c
void softmax_hvx(float* output, const float* input, int n) {
  // Step 1: Find max (for numerical stability)
  HVX_Vector vmax = *((HVX_Vector*)input);
  for (int i = 32; i < n; i += 32) {
    HVX_Vector vin = *((HVX_Vector*)(input + i));
    vmax = Q6_Vsf_vmax_VsfVsf(vmax, vin);
  }
  
  // Horizontal max reduction
  vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vror_VR(vmax, 64));
  vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vror_VR(vmax, 32));
  vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vror_VR(vmax, 16));
  vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vror_VR(vmax, 8));
  vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vror_VR(vmax, 4));
  float max_val = Q6_R_vextract_VR(vmax, 0);
  HVX_Vector vmax_broadcast = Q6_V_vsplat_R(*(int32_t*)&max_val);
  
  // Step 2: Compute exp(x - max) and sum
  HVX_Vector vsum = Q6_V_vzero();
  for (int i = 0; i < n; i += 32) {
    HVX_Vector vin = *((HVX_Vector*)(input + i));
    HVX_Vector vexp = Q6_Vsf_vsub_VsfVsf(vin, vmax_broadcast);
    
    // Use math library or approximation for exp
    // (HVX doesn't have native exp; call scalar exp or use polynomial approx)
    // This is a simplified example
    
    vsum = Q6_Vsf_vadd_VsfVsf(vsum, vexp);
    *((HVX_Vector*)(output + i)) = vexp;
  }
  
  // Step 3: Divide by sum (element-wise)
  // (omitted for brevity; use reciprocal approximation or scalar division)
}
```

---

## 4. Performance Tips and Best Practices

### 4.1 Memory Alignment

**Rule:** Align vector data to 128-byte boundaries for maximum throughput.

```c
// Use aligned allocation
HVX_Vector* data = (HVX_Vector*)memalign(128, n * sizeof(HVX_Vector));

// Or use __attribute__((aligned(128)))
__attribute__((aligned(128))) float data[1024];
```

**Impact:**
- Aligned loads/stores: 1 cycle
- Unaligned loads/stores: 2-3 cycles (uses `VMEMU`/`VSTOREUNALIGNED`)

### 4.2 VTCM Usage

**VTCM (Vector Tightly Coupled Memory)** is the fast scratchpad (1-4 cycle latency):

```c
// Allocate in VTCM (via QAic Compute SDK)
#pragma clang attribute push (__attribute__((section(".vtcm"))), apply_to = variable)
float workspace[1024 * 1024];  // Place in VTCM
#pragma clang attribute pop
```

**Guidelines:**
- Use VTCM for hot loops (activations, intermediate buffers)
- Cloud AI 100 (AIC) NSP cores: 8 MB VTCM per core
- Partition work so hot working sets fit in VTCM, minimizing DDR round trips.

### 4.3 Loop Optimization

**Hardware loops** have zero-cycle overhead:

```c
// Prefer HVX intrinsics with loop-invariant loads hoisted
for (int i = 0; i < n; i += 128) {
  HVX_Vector v = *((HVX_Vector*)(input + i));
  // Process v
  *((HVX_Vector*)(output + i)) = v;
}
```

**Assembly equivalent (compiler-generated):**
```assembly
LOOP0(.L_loop, R_count) {
  V0 = VMEM(R_input++#1)
  # Process V0
  VMEM(R_output++#1) = V0
}:endloop0
```

### 4.4 Avoiding Pipeline Stalls

**Data dependency stalls:**
```c
// BAD: Immediate use after load
HVX_Vector v0 = *((HVX_Vector*)ptr);
HVX_Vector v1 = Q6_Vw_vadd_VwVw(v0, v0);  // Stall: v0 not ready

// GOOD: Interleave independent operations
HVX_Vector v0 = *((HVX_Vector*)ptr1);
HVX_Vector v2 = *((HVX_Vector*)ptr2);  // Independent load
HVX_Vector v1 = Q6_Vw_vadd_VwVw(v0, v0);  // v0 ready by now
```

**Load latency:** ~2-4 cycles from L2, ~1 cycle from VTCM. Interleave 2-3 independent operations to hide latency.

### 4.5 Predication vs. Branching

**Use predicates instead of branches for conditional operations:**

```c
// BAD: Branch inside hot loop
for (int i = 0; i < n; i++) {
  if (data[i] > threshold) {
    output[i] = data[i];
  } else {
    output[i] = 0;
  }
}

// GOOD: Predicated vector select
for (int i = 0; i < n; i += 32) {
  HVX_Vector vin = *((HVX_Vector*)(data + i));
  HVX_Vector vthresh = Q6_V_vsplat_R(*(int32_t*)&threshold);
  HVX_VectorPred qmask = Q6_Q_vcmp_gt_VwVw(vin, vthresh);
  HVX_Vector vout = Q6_V_vmux_QVV(qmask, vin, Q6_V_vzero());
  *((HVX_Vector*)(output + i)) = vout;
}
```

### 4.6 Quantization-aware Operations

For INT8 inference, use saturating arithmetic to prevent overflow:

```c
// Saturating add (byte elements)
HVX_Vector vsum_sat = Q6_Vb_vadd_VbVb_sat(v1, v2);

// Multiply with shift and saturation back to int16 (common in quantized inference)
HVX_Vector vmpy_sat = Q6_Vh_vmpy_VhVh_s1_rnd_sat(v1, v2);
```

### 4.7 Avoid Scalar Fallbacks in Hot Paths

**Minimize scalar operations inside vectorized loops:**

```c
// BAD: Scalar operation per iteration
for (int i = 0; i < n; i += 32) {
  HVX_Vector v = *((HVX_Vector*)(input + i));
  float scalar_val = some_scalar_func();  // Scalar call per iteration
  HVX_Vector vout = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vsplat_R(*(int*)&scalar_val));
  *((HVX_Vector*)(output + i)) = vout;
}

// GOOD: Hoist scalar computation outside loop
float scalar_val = some_scalar_func();
HVX_Vector vscalar = Q6_V_vsplat_R(*(int*)&scalar_val);
for (int i = 0; i < n; i += 32) {
  HVX_Vector v = *((HVX_Vector*)(input + i));
  HVX_Vector vout = Q6_Vsf_vadd_VsfVsf(v, vscalar);
  *((HVX_Vector*)(output + i)) = vout;
}
```

---

## 5. Assembly Language Reference

### 5.1 Common Assembly Patterns

**Vector accumulation loop:**
```assembly
  R0 = #128               # Vector stride
  R1 = #num_iters
  V0 = VMEM(R_input+#0)   # Initial load
  
  LOOP0(.L_acc, R1) {
    V1 = VMEM(R_input+#1)
    V0 = VADD(V0, V1)
    R_input = ADD(R_input, R0)
  }:endloop0
  
  VMEM(R_output+#0) = V0
```

**Conditional vector store:**
```assembly
  V0 = VMEM(R_input+#0)
  V1 = VMEM(R_thresh+#0)
  Q0 = VCMP.GT(V0.W, V1.W)     # Compare words
  IF (Q0) VMEM(R_output+#0) = V0
```

**Unaligned vector processing:**
```assembly
  V0 = VMEMU(R_unaligned+#0)   # Unaligned load
  # Process V0
  VSTOREUNALIGNED(R_out, V0)   # Unaligned store
```

### 5.2 Inline Assembly in C

For fine-grained control, embed assembly in C:

```c
void my_hvx_kernel(HVX_Vector* output, const HVX_Vector* input, int n) {
  __asm__ __volatile__ (
    "V0 = vmem(%1+#0)\n\t"
    "V1 = vmem(%1+#1)\n\t"
    "V2 = vadd(V0.w, V1.w)\n\t"
    "vmem(%0+#0) = V2\n\t"
    : /* outputs */
    : "r"(output), "r"(input)  /* inputs */
    : "v0", "v1", "v2"         /* clobbers */
  );
}
```

**Use sparingly:** Intrinsics are preferred for maintainability; inline assembly is for rare micro-optimizations.

---

## 6. Debugging and Profiling

### 6.1 Hexagon Simulator

For development without hardware:

```bash
# Compile for simulator
hexagon-clang++ -mv68 -O3 -mhvx -mhvx-length=128B kernel.cpp -o kernel.elf

# Run in simulator
hexagon-sim kernel.elf
```

### 6.2 Performance Counters

Use Cloud AI 100 profiling tools to measure HVX utilization:

```bash
# Profile QPC with per-op stats
/opt/qti-aic/exec/qaic-runner \
  -t model.qpc \
  -a input.raw \
  -write-output-dir outputs/ \
  -enable-op-stats
```

See the `cloud-ai-performance-optimization` skill for detailed profiling workflow.

### 6.3 Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Unaligned loads | Low HVX utilization, 2-3x slower | Align data to 128 bytes |
| VTCM overflow | Sudden perf drop, L2 thrashing | Reduce working set size |
| Scalar fallback | Poor vectorization | Check compiler output (`-Rpass=loop-vectorize`) |
| Data dependency stalls | Low IPC (instructions per cycle) | Interleave independent ops |
| Branch misprediction | Irregular perf | Use predication instead |

---

## 7. Related Skills and References

### Internal References

- **`cloud_ai_hexagon_v68_dsp_architecture.md`** — Architectural overview (VLIW, DMT, memory hierarchy, VTCM)
- **`cloud-ai-custom-ops-kernels` skill** — Custom operator development workflow using QAic Compute SDK
- **`cloud-ai-performance-optimization` skill** — Profiling, compiler flags, and optimization methodology
- **`cloud-ai-qaic-compile` skill** — Compiler options and graph-level optimizations

### External References

- **Hexagon v68 Programmer's Reference Manual**  
  <https://docs.qualcomm.com/doc/80-N2040-46/80-N2040-46_REV_B_Hexagon_V68_Programmers_Reference_Manual.pdf>  
  Comprehensive ISA reference (1000+ pages)

- **Hexagon v68 HVX Reference**  
  <https://docs.alexrp.com/hexagon/hexagon_v68_hvx.pdf>  
  Vector extension instruction catalog and usage patterns

- **Hexagon Toolchain Repository** — <https://github.com/qualcomm/toolchain_for_hexagon> — LLVM-based cross-compiler for Hexagon (clang 15.0.5)

### Skills

Use these Claude Code skills for end-to-end workflows:

- **`/cloud-ai-model-onboarding`** — Full model onboarding pipeline (export → compile → run)
- **`/cloud-ai-performance-optimization`** — Optimize inference latency and throughput
- **`/cloud-ai-accuracy-tuning`** — Debug accuracy issues and apply quantization

---

## Quick Reference Card

### HVX Intrinsic Cheat Sheet

| Operation | Intrinsic Example | Notes |
|-----------|------------------|-------|
| Load (aligned) | `*((HVX_Vector*)ptr)` | 128-byte aligned |
| Load (unaligned) | `*((HVX_Vector_UN*)ptr)` | Any address (use QAic SDK type) |
| Store (aligned) | `*((HVX_Vector*)ptr) = v` | 128-byte aligned |
| Store (unaligned) | `*((HVX_Vector_UN*)ptr) = v` | Any address (use QAic SDK type) |
| Add | `Q6_Vw_vadd_VwVw(v1, v2)` | Word elements |
| Multiply | `Q6_Ww_vmpy_VhVh(v1, v2)` | Half → word pair (`HVX_VectorPair`); no single-vector form exists |
| MAC | `Q6_Ww_vmpyacc_WwVhVh(vacc, v1, v2)` | vacc (pair) += v1 * v2 |
| Max | `Q6_Vb_vmax_VbVb(v1, v2)` | Byte elements |
| Compare | `Q6_Q_vcmp_gt_VbVb(v1, v2)` | Returns predicate |
| Select | `Q6_V_vmux_QVV(q, vtrue, vfalse)` | Conditional blend |
| Splat | `Q6_V_vsplat_R(scalar)` | Broadcast scalar |

### Scalar Register Quick Ref

| Registers | Use |
|-----------|-----|
| R0-R5 | Function args / temporaries |
| R0 / R1:0 | Return value |
| R16-R27 | Callee-saved |
| R29 | Stack pointer |
| R31 | Link register |
| P0-P3 | Predicate registers |

### Vector Register Quick Ref

| Registers | Use |
|-----------|-----|
| V0-V15 | Caller-saved (temporaries) |
| V16-V31 | Callee-saved |
| Q0-Q3 | Vector predicates |

---

**This reference is maintained as part of the Cloud AI 100 developer resources. For questions or contributions, refer to the main repository documentation.**
