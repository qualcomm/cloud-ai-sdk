# Instructions to run SDXL-Turbo on Cloud AI 100

The instructions below are to run the [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) model on Cloud AI 100. Compile time parameters may need to be adjusted for different cards and different SDKs.

## Containerized workflow (recommended)

Needs Docker with the Compose plugin and about 50 GB of free disk in this
folder (ONNX models, QPCs, and downloaded weights persist here).
The Platform SDK driver must be installed on the host.

From the repository root, add [qaic-container](../../../../utils/README.md) to
PATH, then prepare this recipe's output directories:
```bash
export PATH="$PWD/utils:$PATH"
cd models/multimodal/text_to_image/sdxl_turbo
mkdir -p cache onnx_files qpc compile_logs images
qaic-container build
qaic-container run --rm export
qaic-container run --rm inference --prompt "<your prompt>"
```

The containers run as your host user and receive the host's `qaic` GID for
device access. Existing output directories must be writable by your account.
Re-run `qaic-container build` after changing the recipe or switching hosts or
users.

`export` downloads the model weights and runs the ONNX generation and QPC
compilation of step 1. `inference` writes one image on device 0 (`--prompt`
optional; `run_config_inference.sh` supplies a default if omitted); add
`--num-warmup-iters 3 --num-repeat-iters 3` for the full benchmark. Extra
flags after `inference` pass through to `main.py` (later flags win). See
`compose.yaml` for details.

The remaining sections describe the manual workflow for hosts without
containers, including the REST endpoint of step 3, which the containers do
not cover.

## Pre-requisites

Use the [SDK 1.23.2.0](https://github.com/qualcomm/cloud-ai-containers/pkgs/container/cloud_ai_inference_ubuntu22?tag=1.23.2.0) container to compile the sdxl-turbo models.

```
sudo docker run \
  -it \
  --workdir /cloud-ai-sdk \
  --entrypoint /bin/bash \
  --network=host \
  --mount type=bind,source=<path to cloud-ai-sdk root>,target=/cloud-ai-sdk \
  --device=/dev/accel/accel0 \
  --device=/dev/accel/accel1 \
  --device=/dev/accel/accel2 \
  --device=/dev/accel/accel3 \
  ghcr.io/qualcomm/cloud_ai_inference_ubuntu22:1.23.2.0

cd models/multimodal/text_to_image/sdxl_turbo
```

Install the moreutils package for the `ts` timestamp tool:
```
sudo apt update
sudo apt-get install moreutils
```

Install Git Large File System (LFS) support

```
sudo apt update
sudo apt-get install git-lfs
```

## 1. Generate onnx files and compile for binaries

1.  Set up a virtual environment for ONNX generation and compilation
```
python3.12 -m venv env_onnx
source ./env_onnx/bin/activate
pip install -r requirements.txt
```

2.  Create a folder for caching Hugging Face model downloads, and export the environment variable HF_HOME
```
mkdir cache
mkdir compile_logs
mkdir qpc
touch run.sh
export HF_HOME=${PWD}/cache
```

3. Install diffusers from source after patching for ONNX file generation
```
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-onnx
cd diffusers-onnx
git apply --reject --whitespace=fix ../patches/attention_patch.patch
pip install .
cd ..
```

4. Prepare VAE Decoder
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/stabilityai/sdxl-turbo cache/stabilityai/sdxl_turbo
cd cache/stabilityai/sdxl_turbo
git lfs pull -I vae_decoder/model.onnx
rm -rf .git/lfs # optional to save space
cd ../../../
```

5. Generate ONNX files and compile for binaries
```
bash run_config_gen.sh
```

## 2. Run the end-to-end SDXL-Turbo inference

1. Set up a separate virtual environment for running SDXL Turbo
```
python3.12 -m venv env_pipeline
source ./env_pipeline/bin/activate
pip install -r requirements.txt
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2.  Re-install diffusers from source after patching the SDXL Turbo pipeline for inference
```
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-pipeline
cd diffusers-pipeline
git apply --reject --whitespace=fix ../patches/pipeline_patch_separate.patch
pip install .
cd ..
```

3. Run the SDXL-Turbo inference on AI 100 devices.

For host inference without `sudo`, add yourself to the `qaic` group:

```bash
sudo usermod -aG qaic "$USER"
```

Log out and log back in to apply the membership.

```
bash run_config_inference.sh $(which python3)
```

## 3. Start an OpenAI-compatible REST endpoint

```
source ./env_pipeline/bin/activate
python3 server.py
```

Test the endpoint:

```
curl http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-key' \
  -d '{
    "model": "sdxl-turbo",
    "prompt": "photo of 8k ultra realistic harbour, port, boats, sunset, beautiful light, golden hour, full of colour, cinematic lighting, battered, trending on artstation, 4k, hyperrealistic, focused, extreme details, cinematic, masterpiece",
    "n": 1,
    "size": "512x512",
    "response_format": "b64_json"
  }'
```
