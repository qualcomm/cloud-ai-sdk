# Connecting Cloud AI models to Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) is a self-hosted web interface for AI use-cases like Chat, Image Generation and RAG.
By starting OpenAI-compatible endpoints with vLLM, we can connect Open WebUI to AI models running on Qualcomm Cloud AI accelerators.

<p align="center">
  <img src="open_webui_screen_2.png" alt="Open WebUI Chat" width="900"/>
</p>

## Pre-requisites

* Cloud AI Platform and Apps SDKs [Installation](https://qualcomm.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/index.html)
* Cloud AI 100 Ultra accelerator card
* Python 3.12
* Docker

To run language models on multiple SoCs, make sure [tensor slicing](https://qualcomm.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Features/model_sharding/index.html) is enabled and disable ACS:

```
sudo /opt/qti-aic/tools/qaic-util -a
```

Increase the response timeout:
```
sudo sh -c "echo 2600 > /sys/module/qaic/parameters/control_resp_timeout_s"
```

Preface all docker commands with `sudo`, or add yourself to the docker group:
```
sudo usermod -aG docker $USER
```

Launch a new shell or `newgrp docker` to apply the changes.

## Prepare the model

Use [Efficient Transformers](https://github.com/qualcomm/efficient-transformers) to prepare popular models like Llama-3.3-70B-Instruct, Qwen2.5-Coder and Phi4, or download pre-generated model binaries at http://qualcom-qpc-models.s3-website-us-east-1.amazonaws.com/QPC/.  Note the location of the 'programqpc.bin' files as you'll need these to start vLLM. Efficient-transformers stores model binaries in [~/.cache/qeff_cache](https://qualcomm.github.io/efficient-transformers/source/quick_start.html#transformed-models-and-qpc-storage) by default.

## Cloud AI Inference Container

[Cloud AI Inference containers](https://github.com/qualcomm/cloud-ai-containers/pkgs/container/cloud_ai_inference_ubuntu24) include everything needed to compile and serve models with vLLM on Cloud AI accelerators.
The Docker command below supplies `serve.sh` as the
entrypoint.

Download the Docker image:

```
docker pull ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:1.23.2.0
```

## Start vLLM endpoint

Prepare a script to launch vLLM with the pre-generated model binary inside the container.

Customize the Hugging Face model name (`model` in `serve.sh`), context length
(`--max-model-len`), and concurrent sequence limit (`--max-num-seqs`) to
match the pre-compiled QPC from the 'Prepare the Model' step above. The
`qpc_path` entry identifies the mounted QPC, so vLLM loads it instead of
compiling at startup.

```
$ cat <<'EOF' > serve.sh
#!/bin/bash
model=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4

exec /opt/vllm-env/bin/vllm serve "$model" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --additional-config '{
    "device_group": [0, 1, 2, 3],
    "override_qaic_config": {
      "qpc_path": "/model/qpc"
    }
  }'
EOF

# Script must have execute permission
$ chmod +x serve.sh
```

Set `/path/to/qpc` to the QPC directory from the 'Prepare the model' step. It
must contain `programqpc.bin`. The `device_group` selects the QIDs used by the
QPC. If the QPC was built with a non-default prefill length or CCL context
buckets, add the matching values under `override_qaic_config`. They are
recorded in the QPC build configuration, such as `qconfig.json` and
`specializations.json`.
If your system has multiple Ultra cards, adjust the Docker `--device` mapping
and the `device_group` QIDs together.
This example creates a `qaic-vllm` Docker volume to hold persistent data (namely the tokenizer weights downloaded from Hugging face).

```
docker run -dit \
  --workdir /model \
  --name qaic-vllm \
  --network host \
  --mount type=bind,source=${PWD}/serve.sh,target=/model/serve.sh \
  --mount type=bind,source=/path/to/qpc,target=/model/qpc \
  -v qaic-vllm:/model/data \
  --env VLLM_QAIC_MAX_CPU_THREADS=8 \
  --env HF_HOME=/model/data/huggingface \
  --env QEFF_HOME=/model/data/qeff_models \
  --device=/dev/accel/ \
  --entrypoint=/model/serve.sh \
  ghcr.io/qualcomm/cloud_ai_inference_ubuntu24:1.23.2.0
```

## Test the endpoint

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful AI assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

## Start Open WebUI

Download Open WebUI Docker image:

```
docker pull ghcr.io/open-webui/open-webui:main
```

Refer to [setup instructions](https://docs.openwebui.com/getting-started/quick-start/#quick-start-with-docker-) for more details.

Run the Open WebUI container:

```
docker run \
  -d \
  --network host \
  -e OPENAI_API_KEY=test-key \
  -e OPENAI_API_BASE_URL="http://localhost:8000/v1" \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

In web browser, open http://<your_ip_or_server_name>:8080

Setup:
* For first time startup, create a default user.  This user will have admin access.
* Click Profile icon in upper right and open Admin Panel -> Settings -> Connections.
* Click Configure icon for Manage OpenAI API Connections.
* Make sure URL is http://localhost:8000/v1.  Key can be any value
* Click Verify Connection icon to test the connection.
* You should see a "Server Connection Verified" pop-up
* If it fails, double-check that the server.py script is running
* Back on the Open WebUI home page, select the model name from the 'Prepare the model' step above. 

<img src="open_webui_screen_1.png" alt="Open WebUI Setup" width="400"/>

You can now use the Chat interface in Open WebUI.
