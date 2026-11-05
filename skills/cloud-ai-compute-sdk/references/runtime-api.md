# QAic Compute SDK — Runtime Reference

## Init/exit flow (`runtime/lib/_qaic_start.cpp`)

```
thread 0:                     other threads:
  _nspContextInit             spin-wait on _initsDone
  _programDescInit
  _initsDone = true
  ─────────────── all threads converge ───────────────
  logActivate(tid)
  _udmaContextInit(tid)
  activate(qctx, tid, stid)   ← user code runs here
  flush log (4× NN_LOG "\n")
  _udmaContextCleanup(tid)
  logDeactivate(tid)
  spin on exitDB != 0          ← host signals when safe to exit
  exitThread()
```

Only thread 0 runs the privileged init path. After `activate()` returns every thread spins on a host-side volatile doorbell (`exitDB`) before calling `exitThread()` — the host controls teardown timing.

---

## Multi-NSP multicast (MCID)

`broadcastToBuffer(internalBufferNum(N), dstOffset, size, src, tid, waitForDone)` fans a UDMA write to every NSP owning that internal buffer via `CoreInfo::mcAddresses[]`.

The `multicast_example` (3 NSPs) pattern:

- NSP 0 thread 0: receives inputs → `broadcastToBuffer` to NSPs 1 and 2
- NSPs 1/2: process partial results → `broadcastToBuffer` back to NSP 0
- NSP 0 thread 1: waits for both internal buffers → `sendAllOutputs()`

**Buffer offset overlap trick:** the example's output buffer (`devOffset:128`) and internal buffer 0 on NSP 0 (`devOffset:128`) share the same VTCM address — NSP 0 can call `sendAllOutputs()` directly without an extra copy. See `examples/multicast_example/multicast.json`.

---

## `qaic-cc` driver flags

`tools/qaic-cc/qaic-cc.cpp` wraps the Hexagon clang toolchain. QAIC-specific flags:

| Flag | Purpose |
|---|---|
| `-qaic-program-config <file>` | **Required for executables** — links the JSON config into the QPC |
| `-no-defaultlib-paths` | Skip default lib search paths |
| `-with-stdlib` | Link standard library |
| `-debug` | Enable debug output |
| `-save-temps` | Keep intermediate files |

Standard clang flags work: `-c`, `-S`, `-E`, `-I`, `-D`, `-O0..-O3`, `-Wl,`, `-l`, `-L`.

Omitting `-qaic-program-config` on an executable link produces a QPC with no buffer descriptor — `qaic-runner` loads it but I/O calls hang immediately.

---

## Key constants

For hardware values (VTCM 8 MB, L2TCM 1 MB, 6 threads, 128-byte cache line) see `cloud-ai-architecture`.

**JSON config limits** (`runtime/lib/AICDefsInternal.h`):

| Constant | Value | JSON field |
|---|---|---|
| `MAX_NUM_CORES` | 16 | `numNSPs` upper bound |
| `MAX_NUM_THREADS` | 6 | `numThreads` upper bound (default 4) |

**User app constraints:**

| Constant | Value | Where / impact |
|---|---|---|
| `DEFAULT_STACK_SIZE` | 12 288 B | `lib/toolchain/Linker.h` — set by linker automatically; stack overflow is **silent**, all per-thread stack arrays must fit |
| `UDMAMaxSize` | (1<<24)−1 ≈ 16 MB | `runtime/lib/libdev/libdev_udma.cpp` — `broadcastToBuffer` splits automatically; only relevant calling UDMA descriptors directly |

**Debugging:**

| Constant | Value | Meaning |
|---|---|---|
| `UTimerFreqMS` | 19 200 ticks/ms | `AICDefsInternal.h` — divide raw hardware timer ticks by this for milliseconds |
| `waitTimeoutLogMS` | 1 000 ms | `runtime/lib/NSPContext.h` — hang-notify fires after this; not user-configurable, but explains periodic hang log messages |
