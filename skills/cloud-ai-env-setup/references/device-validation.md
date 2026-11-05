# Device Validation and Smoke Test

## Verify Devices Inside the Container

Confirm that the container can access the devices and record their NSP count:

```bash
docker exec "$CONTAINER" /opt/qti-aic/tools/qaic-util -q \
  | grep -E "^QID|Status:|Nsp Total:|Nsp Free:|SoC temp|Board power"
```

You should see one entry per device in `Ready` state with an `Nsp Total` line:

```
QID 0
        Status:Ready
        Nsp Total:16
        Nsp Free:16
QID 1
        Status:Ready
        Nsp Total:16
        Nsp Free:16
```

**Record the `Nsp Total` value** — this is the maximum `-aic-num-cores` value you can pass to `qaic-compile` for that device. The model onboarding core count sweep (1, 2, 4, 8, 16) stops at this number.

**Check `Nsp Free` before running any workload** — a model compiled with `N` cores requires `N` free NSPs. If `Nsp Free < N` on the intended device, scan other QIDs and select a Ready QID where `Nsp Free ≥ N`. Do not assume QID 0. See the `cloud-ai-qaic-runner` skill §2.1 for the full device selection procedure.

**Troubleshooting:**
- No devices: check that `--device /dev/accel/` was passed correctly to `docker run`
- Status other than `Ready`: ask the server admin to reload the driver, reboot the host, or re-insert the card
- `Status:Error` with log lines `Permission denied … Error no 13` for every QID: the container is running as a non-root user who is not in the `qaic` group — see the [non-root container guidance](container-workflow.md#start-the-docker-container).

### Run the bundled ResNet-50 smoke test

Run the smoke-test block inside the selected SDK container. If you are using a
host shell, execute the block with `docker exec "$CONTAINER" sh -lc`.

Inference containers include validated QPC fixtures under
`/opt/qti-aic/test-data/aic100/v2/<N>nsp/`. Use a fixture whose `<N>nsp`
requirement fits the selected QID's `Nsp Free` value. To find the bundled
ResNet-50 fixtures:

```bash
TEST_ROOT=/opt/qti-aic/test-data/aic100/v2
TEST_DATA=$(find "$TEST_ROOT" -maxdepth 3 -type d \
  -name "4nsp-quant-resnet50" -print -quit)
if [ -z "$TEST_DATA" ]; then
  TEST_DATA=$(find "$TEST_ROOT" -maxdepth 3 -type d \
    -name "*nsp-quant-resnet50" -print | sort -V | head -n 1)
fi
test -n "$TEST_DATA"
find "$TEST_ROOT" -maxdepth 3 -type d -name "*quant-resnet50" -print
```

Prefer the 4-NSP ResNet-50 fixture when it is present. Multiple activations of
that fixture can exercise most available NSPs without assuming a particular
device size. If it is unavailable, the command selects another fixture; verify
that its `<N>nsp` requirement fits the selected QID's `Nsp Free` value.

The fixture is for validating container, device, QPC, and input/output access.
It is not a fresh model conversion or a production performance benchmark. Copy
the fixture before running it so the bundled files remain unchanged:

```bash
WORK_DIR=/workspace/resnet50-smoke
mkdir -p "$WORK_DIR"
cp -a "$TEST_DATA"/. "$WORK_DIR"/
EXPECTED_OUTPUT="$WORK_DIR/expected_ArgMax.bin"
cp "$WORK_DIR/user_idx1_ArgMax.bin" "$EXPECTED_OUTPUT"
REQUIRED_NSP=${TEST_DATA%/*}
REQUIRED_NSP=${REQUIRED_NSP##*/}
REQUIRED_NSP=${REQUIRED_NSP%nsp}
QID=$(
  /opt/qti-aic/tools/qaic-util -q |
    awk -v need="$REQUIRED_NSP" '
      /^QID / { qid=$2; ready=0 }
      /Status:Ready/ { ready=1 }
      /Nsp Free:/ {
        free=$NF; sub(/^.*:/, "", free)
        if (ready && free + 0 >= need + 0) { print qid; exit }
      }'
)
test -n "$QID"

/opt/qti-aic/exec/qaic-runner -t "$WORK_DIR" -q
/opt/qti-aic/exec/qaic-runner \
  -t "$WORK_DIR" \
  --aic-batch-json-input "$WORK_DIR/ios.json" \
  -n 1 -a 1 -d "$QID"
cmp -s "$EXPECTED_OUTPUT" "$WORK_DIR/user_idx1_ArgMax.bin"
```

The `-q` output must show the fixture's required cores and the selected QID
must have at least that many free NSPs. A successful `cmp` confirms the
bundled expected `ArgMax` output. For the preferred 4-NSP fixture, run a
device-fill check with multiple activations:

```bash
NSP_FREE=<value recorded from qaic-util>
ACTIVATIONS=$((NSP_FREE / 4))
/opt/qti-aic/exec/qaic-runner \
  -t "$WORK_DIR" \
  -i "$WORK_DIR/user_idx0_input_tensor_1.bin" \
  -a "$ACTIVATIONS" -S 1 --time 30 \
  -l -r 1000 -s -d "$QID"
```

Use the resulting throughput only as a device/container smoke result. Use
`cloud-ai-qaic-runner` and the performance-optimization skill for formal
throughput or latency measurements.
