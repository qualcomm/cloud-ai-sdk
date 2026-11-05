# QAIC sysfs and debugfs telemetry

Use these nodes after locating the accelerator's dynamic sysfs paths. Do not
assume the accelerator or `hwmon` numbers:

```text
/sys/class/accel/accelN/device/
/sys/class/accel/accelN/device/hwmon/hwmon*/
```

## Sysfs nodes

| Node | Meaning | Units |
|---|---|---|
| `device/uevent` | Driver, PCI ID, and PCI address | text |
| `device/current_link_speed`, `current_link_width` | Current PCIe link | GT/s, lanes |
| `device/max_link_speed`, `max_link_width` | Maximum PCIe link | GT/s, lanes |
| `hwmonN/temp1_input`, `temp1_highest` | Board temperature and maximum | °C × 1000 |
| `hwmonN/temp2_input`, `temp2_highest` | SoC temperature and maximum | °C × 1000 |
| `hwmonN/temp2_crit`, `temp2_emergency` | Warning and shutdown thresholds | °C × 1000 |
| `hwmonN/temp2_floor`, `temp2_ceiling` | SoC thermal bounds | °C × 1000 |
| `hwmonN/temp3_alarm` | DDR thermal alarm/status | driver/firmware-defined; do not treat as °C |
| `hwmonN/temp_pmicA_input`, `temp_pmicC_input`, `temp_pmicE_input` | PMIC temperatures | °C × 1000 |
| `hwmonN/power1_input`, `power1_max` | Board power and TDP | W × 1,000,000 |
| `hwmonN/power2_input`, `power2_max` | SoC power and TDP | W × 1,000,000 |
| `hwmonN/throttle_percent` | Performance currently lost to throttling | percent |
| `hwmonN/throttle_time` | Time spent throttled | seconds |
| `hwmonN/uptime` | Device uptime | seconds |
| `device/aer_dev_correctable`, `aer_dev_nonfatal`, `aer_dev_fatal` | PCIe error counters | counter text |
| `/sys/class/accel/accelN/dbc*_state` | DBC state | state code |

Temperature values are reported in degrees Celsius multiplied by 1,000.
Power values are reported in watts multiplied by 1,000,000. `temp3_alarm` is
an alarm/status node, not a Celsius temperature.

## MSI/MSI-X vectors

Use this as an informational check when troubleshooting performance. The QAIC
kernel driver requires 32 MSIs for optimal performance; it can operate with a
single MSI with reduced performance.

`lspci` is the authoritative MSI check.

Check the PCI capability with:

```bash
device=/sys/class/accel/accel0/device
bdf=$(basename "$(readlink -f "$device")")
sudo lspci -vvv -s "$bdf" | grep -i -A4 'MSI\|MSI-X'
```

Look for `MSI: Enable+ Count=32/32`. `Enable+` means MSI is active and
`Count=32/32` means 32 interrupts are supported and enabled. Only a
single-MSI result is a performance finding, not a driver setting to change in
the guest.

If `sudo` is unavailable, use `msi_irqs` only as a fallback to rule out
single-MSI:

```bash
device=/sys/class/accel/accel0/device
find "$device/msi_irqs" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l
```

A result greater than 1 means MSI configuration is fine. A count of 1 means
single-MSI.

On bare metal, if `lspci` does not show `Enable+ Count=32/32` or the fallback
returns a count of 1, suggest enabling MSI in BIOS: VT-d on Intel platforms or
IOMMU on AMD platforms. In a VM, the hypervisor may expose only one MSI; there
is no guest-side remedy. Record the limitation and continue with other
performance checks.

## DBC states

A DMA bridge channel (DBC) is a per-device channel used to submit work. The
`dbc*_state` nodes are read-only:

| Value | State |
|---:|---|
| `0` | idle; available for activation |
| `1` | assigned; a workload is running |
| `2` | before shutdown |
| `3` | after shutdown |
| `4` | before power-up |
| `5` | after power-up |

The `32` value in the QAIC MHI configuration is `num_elements`, the ring depth
for most channels, not the number of MHI channels or MSI vectors. Control
channels use `128` in the SDK 1.22 driver. This is driver configuration, not a
dedicated sysfs setting.

## Optional debugfs diagnostics

Debugfs must be mounted at `/sys/kernel/debug`. Locate the kernel-created
QAIC and MHI directories because their roots can vary with the kernel
integration.

- QAIC `bootlog` captures firmware boot output.
- Per-DBC `fifo_size` shows channel capacity; `queued` shows queued work.
- MHI `states`, `channels`, `devices`, and `regdump` show transport state,
  channels, registered devices, and registers.

Missing nodes and read errors such as `EOPNOTSUPP` or `ENODEV` mean the value
is unavailable; they are not zero readings. Do not write device power, TDP, or
thermal settings during routine diagnosis.
