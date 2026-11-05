// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear

#include "QAicHexagonUtils.h"

// Elementwise FP16 scale using HVX vectorization.
// 64 fp16 elements fit in one 128-byte HVX vector.
//
// args->pointers[0] = __fp16*   input          (fp16 tensor)
// args->pointers[1] = __fp16*   output         (fp16 tensor)
// args->pointers[2] = int32_t*  num_elements   (1-element int32 tensor)
// args->pointers[3] = __fp16*   scale          (1-element fp16 tensor)

static constexpr uint32_t VEC_ELEMS = HVX_VectorSize / sizeof(__fp16);  // 64

extern "C" uint32_t scale_fp16(
    const AicJitEntryPointConfig *cfg,
    const AicJitPointerArray     *args)
{
    const __fp16 *input  = static_cast<const __fp16 *>(args->pointers[0]);
          __fp16 *output = static_cast<__fp16 *>(args->pointers[1]);
    uint32_t      n      = *static_cast<const int32_t *>(args->pointers[2]);
    const __fp16  scale  = *static_cast<const __fp16 *>(args->pointers[3]);

    // Partition by vector-aligned chunks across all cores × threads.
    uint32_t n_vecs     = divideCeil(n, VEC_ELEMS);
    uint32_t total      = cfg->numCores * cfg->numThreads;
    uint32_t id         = cfg->coreID * cfg->numThreads + cfg->threadID;
    uint32_t vec_chunk  = divideCeil(n_vecs, total);
    uint32_t vec_start  = id * vec_chunk;
    uint32_t vec_end    = vec_start + vec_chunk < n_vecs ? vec_start + vec_chunk : n_vecs;

    // Broadcast scalar to all 64 lanes.
    HVX_VectorHF scale_vec(scale);

    for (uint32_t v = vec_start; v < vec_end; ++v) {
        uint32_t i   = v * VEC_ELEMS;
        uint32_t rem = n - i;

        if (rem >= VEC_ELEMS) {
            // Full 128-byte vector load/multiply/store.
            HVX_VectorHF in = *(const HVX_VectorHF *)(input + i);
            *(HVX_VectorHF *)(output + i) = in * scale_vec;
        } else {
            // Scalar tail for the last partial vector.
            for (uint32_t j = i; j < n; ++j)
                output[j] = input[j] * scale;
        }
    }

    return 0;  // 0 = success; use QAIC_OPERATOR_IMPL_ERROR for errors
}
