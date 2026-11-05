# qaic-bench

Benchmarking script for Cloud AI Inference accelerators.

## Installation for x86_64

Download Cloud AI Docker Image:

```
docker pull ghcr.io/qualcomm/cloud_ai_inference_ubuntu22:1.23.2.0
```

Start container. For QPC generation, choose a `/cache` location with 1TB or more of free space to store model weights, ONNX files, and QPC model binaries.

Note: Run `docker container rm qaic-bench` to clean up after exiting the container.

```
cd utils/qaic-bench

docker run -it \
  --workdir /app \
  --name qaic-bench \
  --network host \
  --mount type=bind,source=${PWD},target=/app \
  --mount type=bind,source=${HOME}/.cache,target=/cache \
  --env HF_HOME='/cache/huggingface' \
  --env QEFF_HOME='/cache/qeff_models' \
  --env XDG_CACHE_HOME='/cache' \
  --device /dev/accel/ \
  ghcr.io/qualcomm/cloud_ai_inference_ubuntu22:1.23.2.0
```

Activate vLLM environment:<br>

```
source /opt/vllm-env/bin/activate
```

## Installation for AArch64

Follow instructions [here](https://qualcomm.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/vLLM/vLLM/index.html#installing-from-source) to setup the vLLM environment for Cloud AI from source for AArch64.

Activate vLLM environment:<br>

```
source qaic-vllm-venv/bin/activate
```

## KV-Heads Replication

Download KV-Heads Replication script from Efficient Transformers. This is needed to efficiently tensor-slice large models across 16 SoCs.

```
wget https://github.com/qualcomm/efficient-transformers/raw/refs/heads/release/v1.21.0/scripts/replicate_kv_head/replicate_kv_heads.py
```

## Multi-Device Operation

To run models across multiple AI 100 devices, make sure tensor slicing is enabled. Run these commands outside the container on the host system.

```
sudo /opt/qti-aic/tools/qaic-util -a
```

The control response timeout must also be extended:

```
sudo sh -c 'echo 2600 > /sys/module/qaic/parameters/control_resp_timeout_s'
```

More details at: https://github.com/qualcomm/cloud-ai-sdk/tree/main/utils/multi-device

## Hugging Face Access Token

Some models on Hugging Face are access protected. Add your access token with the `--hf_token` script argument or set the `HF_TOKEN` environment variable. Learn more about Authentication here: https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication.

## Usage

Example:

```
python3 qaic_bench.py config/config_llama_3_1_8b.json
```

Details:

```
usage: qaic_bench.py [-h] [--devices DEVICES] [--compile-only] config

positional arguments:
  config             JSON file with model configurations

options:
  -h, --help         show this help message and exit
  --devices DEVICES  List of comma separated device IDs to use for inferencing
  --compile-only     Generate QPCs and skip benchmarking
  --hf_token         Hugging Face access token
```

## Configuration

### Example

```
{
    "vllm_root": "/opt/qti-aic/integrations/vllm",

    "models": [
        {
            "name": "Meta-Llama-3.1-8B-Instruct",
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "configs": [
                {
                    "batch_size": 1,
                    "devices": 4,
                    "prompt_len": 4096,
                    "generation_len": 4096
                }
            ]
        }
    ]
}
```

### JSON Reference

| Property       | Description                               |
| -------------- | ----------------------------------------- |
| vllm_root      | Path to full vLLM installation            |
| models         | List of models to benchmark               |

### Model Properties

| Property       | Description                               |
| -------------- | ----------------------------------------- |
| name           | Model friendly name                       |
| model          | Hugging Face model path                   |
| configs        | List of model configurations to benchmark |

### Config Properties

| Property         | Description                               |
| ---------------- | ----------------------------------------- |
| batch_size       | Model batch size.                         |
| devices          | Number of Cloud AI SoCs for tensor-sliced execution. Set to 1 for single-SoC execution. |
| cores (optional) | Number of AI Cores for compilation. Default 16. |
| prompt_len       | Prompt input length                       |
| generation_len   | Max number of output tokens to generate   |
| qpc (optional)   | Local path or URL for pre-generated QPC binary. If not specified, QPC will be generated. |
