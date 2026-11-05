# Multi Device 

This guide provides setup instructions for multi-device enablement. PCIe peer-to-peer P2P communication must be enabled to allow efficient tensor slicing across multiple Cloud AI devices (SoCs and Cards).

Refer to [Model Sharding](https://qualcomm.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Features/model_sharding/) for more information on recommended PCIe topologies for tensor slicing (P2P).

## Pre-requisites 

- Server with Platform and Apps SDK versions >= 1.17 installed.
- PCIe switch for inter-card P2P communication
- python3 -m pip install pyudev
- [switchtec](https://github.com/Microsemi/switchtec-user) utility installed

## Setup instructions 

Platform SDK 1.18 and later offers an option (`--setup_mdp all`) to enable P2P for the multi-device partitioning tensor slicing feature during installation.

Example:

```
cd <platform sdk installer>/x86_64/deb
sudo bash install.sh --setup_mdp all
```

> [!IMPORTANT] 
> If P2P is enabled via the Platform SDK installer then skip to the [Testing P2P](#testing-p2p) section.
> 
> The remaining steps in this section show manual steps for enabling P2P.
>
> For VM instances, use the manual steps to enable P2P.

### Disable PCIe ACS for P2P communication between cards 

1. Run `QAicChangeAcs.py` without any flags to display a hierarchial view of PCI bridges and AI 100 devices. 

```
$ python3 QAicChangeAcs.py
Found the following AIC100 devices:
Root
----0000:30:01.1   <-- Host system PCIe switch, script will disable ACS here
--------0000:31:00.0     <-- Ultra AI 100 onboard PCIe switch, script will disable ACS here
------------0000:32:03.0
----------------0000:36:00.0 [Qualcomm AIC100]
------------0000:32:02.0
----------------0000:35:00.0 [Qualcomm AIC100]
------------0000:32:00.0
----------------0000:38:00.0 [Qualcomm AIC100]
------------0000:32:01.0
----------------0000:39:00.0 [Qualcomm AIC100]
--------0000:21:00.0    <-- Ultra AI 100 onboard PCIe switch, script will disable ACS here
------------0000:22:00.0
----------------0000:23:00.0 [Qualcomm AIC100]
------------0000:22:02.0
----------------0000:25:00.0 [Qualcomm AIC100]
------------0000:22:01.0
----------------0000:27:00.0 [Qualcomm AIC100]
------------0000:22:03.0
----------------0000:28:00.0 [Qualcomm AIC100]
```

2. Run `QAicChangeAcs.py all` to disable ACS on all the downstream ports (on the PCIe switch) that connect to AI 100 devices as well as PCIe switch downstream ports that connect to the PCIe switch onboard the AI 100 cards. This command will enable P2P between the AI 100 devices (SoCs) on the same card as well card to card. 

3. Users optionally can selectively disable ACS by running `QAicChangeAcs.py <SSSS:BB:DD.F>`, where 
    - SSSS = 4 digits segment number
    - BB = 2 digits bus number
    - DD = 2 digits device number
    - F = 1 digit function number

    of the nearest common ancestor PCI bridge under which ACS needs to be disabled. 
    
    Examples: 

    `$ python3 QAicChangeAcs.py 0000:31:00.0` will disable ACS on the first set of AI 100 devices (0000:36:00.0, 0000:35:00.0, 0000:38:00.0 and 0000:39:00.0). <br> 
    `$ python3 QAicChangeAcs.py 0000:30:01.1` will disable ACS across both the AI 100 Ultra cards as well as the 4 devices in each AI 100 card <br>

4. Above steps need to be repeated on every server power cycle. 


### Enable multi-device partitioning (MDP)

This step is required everytime a new version of the Platform SDK is installed. 

First, check that the Qaic Monitor service is running
```
sudo systemctl status qmonitor-proxy
```

If not active(running) then start it with:
```
sudo systemd-run --unit=qmonitor-proxy /opt/qti-aic/tools/qaic-monitor-grpc-server
```

Next, enable MDP across all Cloud AI devices in the server.
``` 
sudo /opt/qti-aic/tools/qaic-monitor-json -i enable_mdp.json
```

Reset Cloud AI devices for changes to take effect:
``` 
sudo /opt/qti-aic/tools/qaic-util -s
```

## Testing P2P

The Qaic Kernel driver requires a longer response timeout for P2P workloads. Use the following command to increase the timeout:
```
sudo sh -c 'echo 2600 > /sys/module/qaic/parameters/control_resp_timeout_s'
```

Synthetic P2P workloads are available in `/opt/qti-aic/test-data/aic100/v2/qaic-compute-networks/factory-workload-bin`.

### Multi-SoC Accelerators (Ultra) P2P tests

```
# P2P between 2 SoCs with QID 0 and 1 on the same card
sudo /opt/qti-aic/exec/qaic-runner -t /opt/qti-aic/test-data/aic100/v2/qaic-compute-networks/factory-workload-bin/2c-p2p-bw -n 10 -a 1 -l -D 0:1

# P2P between 2 SoCs with QID 0 and 4 on different cards.  Choose cards that are on the same PCie switch.
sudo /opt/qti-aic/exec/qaic-runner -t /opt/qti-aic/test-data/aic100/v2/qaic-compute-networks/factory-workload-bin/2c-p2p-bw -n 10 -a 1 -l -D 0:4
```

### Single-SoC Accelerators (Standard/Pro) P2P tests

```
# P2P between 2 SoCs with QID 0 and 4 on different cards. Choose cards that are on the same PCie switch.
sudo /opt/qti-aic/exec/qaic-runner -t /opt/qti-aic/test-data/aic100/v2/qaic-compute-networks/factory-workload-bin/2c-p2p-bw -n 10 -a 1 -l -D 0:1
```

### Troubleshooting
If a `Failed to access P2P device` error occurs, check the following:
1. Re-check enablement instructions above
2. Review the PCIe topology from the QAicChangeAcs.py script to make sure that a host PCIe switch is present
