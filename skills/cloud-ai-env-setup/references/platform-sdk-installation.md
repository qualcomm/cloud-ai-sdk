# Platform SDK Installation and Components

## SDK Components

Cloud AI consists of two SDKs:

| SDK | Purpose | Install Location |
|---|---|---|
| **Platform SDK** | Drivers, firmware, runtime libraries for executing compiled models | `/opt/qti-aic/tools/`, `/opt/qti-aic/exec/` |
| **Apps SDK** | Compiler (`qaic-compile`) and development libraries for building QPC artifacts | `/opt/qti-aic/exec/` |

The SDK installer does **not** add `/opt/qti-aic` directories to `PATH` — always use full paths.

Commonly used tools:

**Platform SDK:**
- `/opt/qti-aic/tools/qaic-util` — card information and device status
- `/opt/qti-aic/exec/qaic-runner` — run compiled QPC binaries

**Apps SDK:**
- `/opt/qti-aic/exec/qaic-compile` — compile ONNX/TF/PyTorch models into QPC binaries

## Bare-Metal SDK Installation

Open the [Qualcomm Software Center](https://softwarecenter.qualcomm.com/catalog/item/aic_platform),
select the current release, and read the displayed **Download URL**. Download and extract the ZIP, then enter the
x86_64 package directory for the host's package format:

```bash
SDK_URL='<Download URL copied from Software Center>'
SDK_ZIP=platform-sdk.zip
curl --fail --location --output "$SDK_ZIP" "$SDK_URL"
unzip "$SDK_ZIP"
cd qaic-platform-sdk-<version>/x86_64/<deb-or-rpm>
```

### Full host runtime or container-only driver

For a Debian/Ubuntu host that needs Platform SDK tools locally, first check the
downloaded `qaic-rt` package's glibc requirement before running `install.sh`:

```bash
# Run from qaic-platform-sdk-<version>/x86_64/deb
HOST_GLIBC=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
MIN_GLIBC=$(dpkg-deb -f deb/qaic-rt_*.deb Depends \
  | sed -n 's/.*libc6:amd64 (>= \([0-9.]*\)).*/\1/p')
printf 'host glibc: %s; qaic-rt minimum: %s\n' "$HOST_GLIBC" "$MIN_GLIBC"
dpkg --compare-versions "$HOST_GLIBC" ge "$MIN_GLIBC"
```

If the comparison fails, use the container-only path below. Do not run the
full installer.

For a compatible host that needs host-side runtime tools, run:

```bash
sudo ./uninstall.sh
sudo ./install.sh
```

For a Debian/Ubuntu container-only host, install only firmware and the kernel
driver. This provides devices to SDK containers without installing `qaic-rt` on
the host:

```bash
# Run from qaic-platform-sdk-<version>/x86_64/deb
sudo apt-get install -y dkms "linux-headers-$(uname -r)"
sudo dpkg -i deb/qaic-fw_*.deb deb/qaic-kmd_*.deb
```

The KMD installation builds against the running kernel. Stop if matching
headers are unavailable or DKMS fails. Do not run `install.sh` for this path.

For a DNF/YUM container-only host, install the matching kernel development
package and DKMS, then install the local RPMs:

```bash
# Run from qaic-platform-sdk-<version>/x86_64/rpm
if command -v dnf >/dev/null; then
  PKG_MANAGER=dnf
else
  PKG_MANAGER=yum
fi
sudo "$PKG_MANAGER" install -y "kernel-devel-$(uname -r)" dkms
sudo "$PKG_MANAGER" install -y ./rpm/qaic-fw_*.rpm ./rpm/qaic-kmd_*.rpm
```

Stop if the matching kernel development package is unavailable or DKMS fails.
Do not run `install.sh` for this path.

Install the [Apps SDK](https://softwarecenter.qualcomm.com/catalog/item/aic_apps)
separately when `qaic-compile` is required.
