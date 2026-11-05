# QAic Compute SDK — PMU Profiling

## Overview

The Compute SDK exposes the Hexagon v68 PMU (Performance Monitoring Unit) via two function pointers on the exec context. There is **no compiler-assisted stats collection** — unlike `qaic-compile`'s `-stats-level` flag, `qaic-cc` has no equivalent and never populates the `Stats` proto field in the network descriptor. `qaic-runner --aic-profiling-type raw_device_stats` will fail for compute SDK QPCs.

All PMU access is manual, device-side only.

## API

Declared in `lib/metadata/metadata-common/inc/nnc_pmu_inf.h` and `AICMetadataExecCtx.h`:

```cpp
// Set a PMU register
ctx->setPMUReg(int reg_id, unsigned int reg_value);

// Read all 8 counter values into caller-supplied array
ctx->readPMUCnt(uint32_t counters[MAX_PMU_COUNTERS]);  // MAX_PMU_COUNTERS = 8
```

## PMU Register Reference

| Register | ID | Purpose |
|---|---|---|
| `NNC_PMUCNT0` | 0 | Event counter 0 |
| `NNC_PMUCNT1` | 1 | Event counter 1 |
| `NNC_PMUCNT2` | 2 | Event counter 2 |
| `NNC_PMUCNT3` | 3 | Event counter 3 |
| `NNC_PMUCFG` | 4 | Enable (1) / disable (0) the PMU |
| `NNC_PMUEVTCFG` | 5 | Event selection for counters 0–3 |
| `NNC_PMUCNT4` | 6 | Event counter 4 |
| `NNC_PMUCNT5` | 7 | Event counter 5 |
| `NNC_PMUCNT6` | 8 | Event counter 6 |
| `NNC_PMUCNT7` | 9 | Event counter 7 |
| `NNC_PMUEVTCFG1` | 10 | Event selection for counters 4–7 |
| `NNC_PMUSTID0` | 11 | HW thread filter — write `stid` to count only this thread |
| `NNC_PMUSTID1` | 12 | Second HW thread filter (inclusive OR with PMUSTID0) |

## `stid` — Hardware Thread Context ID

`stid` is the third parameter of `activate()`:

```cpp
extern "C" void activate(void *ctx, uint8_t virtualThreadId, uint32_t stid);
```

It is the Hexagon QuRT hardware software-thread-context ID — the raw HW context number assigned by the OS. Writing `stid` to `NNC_PMUSTID0` tells the PMU to only increment counters for events from that specific hardware execution context. Without this filter, counters aggregate across all threads running on the NSP, making per-thread analysis impossible.

`stid` is not useful for anything else in typical user code.

## Usage Pattern

```cpp
#include "AICMetadataExecCtx.h"
#include "nnc_pmu_inf.h"

void activate(void *ctx, uint8_t virtualThreadId, uint32_t stid) {
    AICExecContext *qctx = (AICExecContext *)ctx;

    // 1. Filter counters to this hardware thread
    qctx->setPMUReg(NNC_PMUSTID0, stid);

    // 2. Select events to count (event codes are hardware-specific;
    //    refer to the Hexagon v68 PMU event reference)
    qctx->setPMUReg(NNC_PMUEVTCFG, <event_code>);

    // 3. Enable PMU
    qctx->setPMUReg(NNC_PMUCFG, 1);

    // --- work ---

    // 4. Read counters
    uint32_t counters[MAX_PMU_COUNTERS];
    qctx->readPMUCnt(counters);

    // 5. Disable PMU
    qctx->setPMUReg(NNC_PMUCFG, 0);

    // 6. Exfiltrate (see below)
}
```

## Exfiltrating Counter Values

`readPMUCnt` fills a `uint32_t[8]` array on the NSP — it does not transmit anything to the host. Two options:

### Option A — Output buffer (structured, preferred)

Declare a DDR output buffer in the JSON config (e.g. `"dims":[8], "dataType":"Int32ITy", "dest":"DDR"`), then write the counters and send:

```cpp
if (virtualThreadId == 0) {
    uint32_t counters[MAX_PMU_COUNTERS];
    qctx->readPMUCnt(counters);

    uint32_t *out = (uint32_t *)qaic::getBufferAddr(qaic::outputBufferNum(N));
    for (int i = 0; i < MAX_PMU_COUNTERS; i++)
        out[i] = counters[i];

    qaic::sendAllOutputs(true, true);
}
```

The host reads the output buffer via `qaic-runner --write-output-dir`. Interpret each `counters[i]` based on the event code written to `NNC_PMUEVTCFG`/`NNC_PMUEVTCFG1`.

### Option B — Logging (quick/debug)

```cpp
NN_LOG(qctx->logFuncPtr, NNC_LOG_MASK_INFO,
       "PMU cnt0=%u cnt1=%u cnt2=%u cnt3=%u",
       counters[0], counters[1], counters[2], counters[3]);
```

Appears in the device log stream captured by the runtime.

## What Does NOT Work

| Approach | Why it fails |
|---|---|
| `qaic-compile -stats-level=100` | `qaic-compile` flag for ML model QPCs only — not applicable to compute SDK |
| `qaic-cc` stats flag | `qaic-cc` has no `-stats-level` or PMU flag (`DriverOptions.td` confirmed) |
| `qaic-runner --aic-profiling-type raw_device_stats` | Requires `Stats.buff_num` in the network descriptor; `qaic-cc` never populates the `Stats` proto — fails with "Program does not support the requested profiling type" |

The `Stats` message in `AICNetworkDesc.proto` (including `pmu_events`, `pmu_stats_level`) exists in the shared schema but is only populated by `qaic-compile`. The compute SDK's driver, linker, and QPCBuilder never write to it.
