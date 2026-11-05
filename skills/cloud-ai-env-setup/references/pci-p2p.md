# PCIe P2P for Ultra MDP/TS Workloads

## Enable PCIe P2P for Ultra MDP/TS workloads

This procedure applies only to `Pcie Ultra` devices. Do not change ACS for
Non-Ultra SKUs. Stop active workloads before inspecting or changing PCIe
routing.

### VM preflight: stop if P2P is already enabled

Before changing ACS in a VM, inspect each intended QID's immediate guest-visible parent bridge:

```bash
for endpoint in $(lspci -Dnnd 17cb:a100 | awk '{print $1}'); do
  bridge=$(basename "$(dirname "$(readlink -f "/sys/bus/pci/devices/$endpoint")")")
  printf '%s via %s: ' "$endpoint" "$bridge"
  sudo lspci -vvv -s "$bridge" | sed -n '/ACSCtl:/p'
done
```

Use `ACSCtl`, not `ACSCap`: `ACSCap` reports supported controls, while
`ACSCtl` reports their state. If every intended route reports `ReqRedir-` and
`CmpltRedir-` (the usual working form has every control disabled), report
`P2P+` and stop. Do not run `qaic-util -a`, Switchtec writes, or `2c-p2p-bw`.
If either control is enabled, or the state cannot be read, report `P2P-` and
choose the applicable enablement path below. Do not use a separately exposed
PFX/Switchtec management BDF to decide a VM QID route.

| Environment | ACS path |
|---|---|
| Bare metal with host `qaic-util` | Use `qaic-util -a` first, then Switchtec if it cannot enable the path. |
| Host `qaic-util` absent | Use Switchtec directly for each mapped Ultra card. |
| VM with passed-through Switchtec | Use Switchtec directly for TS4. |
| VM without passed-through Switchtec | The guest cannot enable the card-internal ACS path. |

Do not use a standard workload container as a substitute for host
`qaic-util`: it normally has read-only PCI sysfs and no `/dev/switchtec*`.
When host `qaic-util` is absent, use Switchtec directly for the mapped Ultra
card.

### Bare metal with host `qaic-util`: primary ACS path

On bare metal, `qaic-util -a` discovers the real PCIe hierarchy and can limit
ACS changes to selected QIDs or a bridge. Inspect the topology first:

```bash
sudo /opt/qti-aic/tools/qaic-util -q
sudo /opt/qti-aic/tools/qaic-util -r -v
lspci -Dtv
test -x /usr/local/bin/switchtec && /usr/local/bin/switchtec list
for DEVICE in /dev/switchtec*; do
  [ -e "$DEVICE" ] || continue
  sudo /usr/local/bin/switchtec info "$DEVICE"
  sudo /usr/local/bin/switchtec status "$DEVICE" -v
done
```

Map each QID to its endpoint BDF from `qaic-util -q`. For an Ultra card,
`switchtec status -v` shows each DSP's `Bus-Dev-Func`, `Bus-Dev-Func Path`,
and `Device:` endpoint. Compare `Device:` with `lspci -Dnn` and the QID's PCI
address. QIDs whose endpoints are in the same Switchtec subtree belong to the
same card. `Board serial` provides a second card-level check.

Before a command with `-a` / `--disable-acs`, ask for explicit permission for
the exact QID or bridge scope. `qaic-util -a` without `-d` or `-b` is
system-wide and can disable ACS through riser switches. Devices downstream of
an affected bridge can then see one another, reducing PCIe isolation for other
devices and workloads.

After approval, use the narrowest scope that enables the intended P2P path:

```bash
# Between selected QAIC devices
sudo /opt/qti-aic/tools/qaic-util -a -d <qid> -d <qid>

# For all QAIC devices below a selected PCIe bridge
sudo /opt/qti-aic/tools/qaic-util -a \
  --pci-sbdf-for-disable-acs <bridge-sbdf>
```

Validate with `qaic-util -r -v`. If the scoped command returns
`QS_UNSUPPORTED` or the mapped Ultra card still lacks the required P2P path,
use the Switchtec fallback below. Do not widen ACS changes to unrelated
bridges just to force a larger tensor slice.

### Switchtec fallback for an Ultra card

Bare-metal fallback: use Switchtec on the mapped Ultra card. Clearing ACS P2P
redirection only on its internal switch enables at least TS4; it does not
enable TS8 or TS16 across cards. The Switchtec userspace utility is separate
from the Cloud AI SDK:

```bash
test -x /usr/local/bin/switchtec
/usr/local/bin/switchtec list
```

On bare metal, select only the Switchtec device mapped above to the affected
Ultra card. All four fixed ACS GAS registers must read successfully before
writing that device. If a read fails, do not write it.

| Switchtec devices | DSP ACS GAS registers | Written value |
|---|---|---|
| Mapped Ultra-card Switchtec devices | `0x1361D0`, `0x1371D0`, `0x1381D0`, `0x1391D0` | `0x1d` |

After explicit permission, run the fixed read-write-read procedure from a
root shell. Pass only mapped devices as arguments:

```bash
# Replace this path with the mapped Switchtec device for the selected card.
sudo bash -s -- /dev/switchtec0 <<'EOF'
set -euo pipefail
SWITCHTEC=/usr/local/bin/switchtec
ACS_REGS=(0x1361D0 0x1371D0 0x1381D0 0x1391D0)
VALIDATED=()

for DEVICE in "$@"; do
  for ADDR in "${ACS_REGS[@]}"; do
    "$SWITCHTEC" gas read "$DEVICE" -a "$ADDR" -n 0x1
  done
  VALIDATED+=("$DEVICE")
done

for DEVICE in "${VALIDATED[@]}"; do
  for ADDR in "${ACS_REGS[@]}"; do
    yes | "$SWITCHTEC" gas write "$DEVICE" -a "$ADDR" -v 0x1d
    "$SWITCHTEC" gas read "$DEVICE" -a "$ADDR" -n 0x1
  done
done
EOF
```

### VM path

Do not use `qaic-util -a` from a VM: it cannot see the host's full ACS
hierarchy. A cloud VM can still enable TS4 when an Ultra card's internal
Switchtec device is passed through. In that case, use Switchtec directly. For
a cloud VM with `Pcie Ultra` QIDs, every passed-through Switchtec device is an
internal-switch candidate. Apply the same fixed procedure to every
passed-through Switchtec device after validating its four reads and obtaining
explicit permission.

Do not infer a specific QID-to-DSP mapping from the port count: per-QID guest BDF
mapping is not required for this TS4 path, and four DSP ports with one upstream port
are diagnostic evidence, not a blocker and not an association proof. When
available, use `switchtec status -v`'s `Device:` endpoint and `lspci -nn` to
correlate the topology. `qaic-util -r -v` should report `P2P+` for the
resulting path.

## Test P2P

The QAIC kernel driver requires a longer control-response timeout for P2P
workloads. Set it before running a P2P test:

```bash
sudo sh -c 'echo 2600 > /sys/module/qaic/parameters/control_resp_timeout_s'
```

Synthetic P2P workloads are included in:
`/opt/qti-aic/test-data/aic100/v2/qaic-compute-networks/factory-workload-bin`.
Use the `2c-p2p-bw` fixture after confirming the QID and PCIe-switch topology:

```bash
P2P_ROOT=/opt/qti-aic/test-data/aic100/v2/qaic-compute-networks/factory-workload-bin

# Example: P2P between QID 0 and QID 1 on the same Ultra card.
sudo /opt/qti-aic/exec/qaic-runner \
  -t "$P2P_ROOT/2c-p2p-bw" -n 10 -a 1 -l -D 0:1

# Example: P2P between QID 0 and QID 4 on different cards.
# Choose a pair of cards connected to the same PCIe switch.
sudo /opt/qti-aic/exec/qaic-runner \
  -t "$P2P_ROOT/2c-p2p-bw" -n 10 -a 1 -l -D 0:4
```

Replace the example QIDs with the tested pair. In a VM, use the
passed-through Switchtec topology to choose the pair; the guest's PCIe tree
may not show the real switch connections.
