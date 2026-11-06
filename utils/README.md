# Basic Commands/Utilities for Cloud AI 100 devices 

## Containers with `qaic-container`

`qaic-container` runs Docker Compose with `HOST_UID`, `HOST_GID`, and
`QAIC_GID` populated from your host account and `qaic` group. Compose files
choose how to use these values to configure their users and groups. Docker
with the Compose plugin and the Platform SDK driver are required.

Add `utils/` to your shell's PATH from the repository root:

```bash
export PATH="$PWD/utils:$PATH"
```

Use normal Compose commands from the directory containing your Compose file:

```bash
qaic-container build
qaic-container run --rm SERVICE
```

Compose options such as `-f` and `--project-directory` pass through unchanged.
Follow each recipe's instructions to prepare directories and select services.

## Add your host account to `qaic` for direct device access

For host device access without `sudo`, add yourself to the `qaic` group:

```bash
sudo usermod -aG qaic "$USER"
```

Log out and log back in to apply the membership.

## Check device health 
Monitor the health of all AI 100 devices (SoCs) using the `qaic-util` utility. 

```
/opt/qti-aic/tools/qaic-util -q | grep -e Status -e QID
```

## Monitoring of AI 100 devices (SoCs)
Continuously monitor the health, telemetry (temperature, power etc) and resources (compute, DRAM etc) of the AI 100 devices (SoCs) using the `qaic-util` utility. 

```
/opt/qti-aic/tools/qaic-util -t 1
```

## Reset AI 100 devices (SoCs)
To reset **all** AI 100 devices (SoCs), run
```
sudo /opt/qti-aic/tools/qaic-util -s
```

To reset **individual** AI 100 devices (SoCs), run 
```
sudo /opt/qti-aic/tools/qaic-util -s -p <PCIe address SSSS:BB:DD.F>
```
where,
    - SSSS = 4 digits segment number
    - BB = 2 digits bus number
    - DD = 2 digits device number
    - F = 1 digit function number

For example, 
```
sudo /opt/qti-aic/tools/qaic-util -s -p 0000:83:00.0

Resetting 0000:83:00.0:
        0000:83:00.0 success
```
