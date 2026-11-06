
### Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
### SPDX-License-Identifier: BSD-3-Clause-Clear


# Instructions to run SDXL on Cloud AI 100

The instructions below are to run [the Stable Diffusion XL model](stabilityai/stable-diffusion-xl-base-1.0) on Cloud AI 100. For Cloud AI Standard Accelerators (AWS DL2q EC2 instances), the block size in attention_processor.py will need to be tuned. Other compile time parameters may also need to be adjusted (e.g., -aic-num-cores)


## Containerized workflow (recommended)

Needs Docker with the Compose plugin and about 50 GB of free disk in this
folder (ONNX models, QPCs, and downloaded weights persist here).
The Platform SDK driver must be installed on the host.

From the repository root, add [qaic-container](../../../../utils/README.md) to
PATH, then prepare this recipe's output directories:
```bash
export PATH="$PWD/utils:$PATH"
cd models/multimodal/text_to_image/stable-diffusion-xl-base-1.0
mkdir -p hf_home HF_repo onnx_files qpc compile_logs images
qaic-container build
qaic-container run --rm export
qaic-container run --rm inference --prompt "<your prompt>" --negative_prompt "<your negative prompt>"
```

The containers run as your host user and receive the host's `qaic` GID for
device access. Existing output directories must be writable by your account.
Re-run `qaic-container build` after changing the recipe or switching hosts or
users.

`export` covers parts 2 and 3 below: ONNX generation (fetching the fixed VAE
decoder file if absent) and all five QPC compiles. `inference` runs part 4 on
device 0 with the batchsize-2 UNet QPC; `--prompt` and `--negative_prompt` are
both required. It writes one image; add
`--num_warmup_iters 3 --num_repeat_iters 3` for the full benchmark. Extra
flags after `inference` pass through to `run_sdxl.py` (later flags win). See
`compose.yaml` for details.

The remaining sections describe the manual workflow for hosts without
containers.

## Folder Structure

```
├── README.md                               # This file
├── onnx_gen_utils.py                       # Utility module for ONNX generation
├── generate_onnx_files.py                  # Python script to generate ONNX files
├── fix_vae_decoder_onnx.py                 # Python script to fix VAE decoder precision issues
├── compile_models.sh                       # Script to compile all the model ONNX files
├── attention_patch.patch                   # Diffusers patch for ONNX generation
├── pipeline_patch.patch                    # Diffusers patch for SDXL inference
├── run_sdxl.py                             # End-to-end SDXL inference script running on AI 100
```

## Pre-requisites
---
Install `moreutils` package for the `ts` timestamp tool and Git Large File System (LFS) support.

```
sudo apt update
sudo apt-get install moreutils git-lfs
```

## 1. Install Platform and Apps SDK

Choose one of the following deployment environments:

- **Container:** Use the
  [cloud_ai_inference_ubuntu24](https://github.com/qualcomm/cloud-ai-containers/pkgs/container/cloud_ai_inference_ubuntu24)
  container. Run the remaining SDK, export, compile, and inference commands
  inside the container.
- **Bare-metal host:** Install the Platform and Apps SDKs directly on the host
  by following the [Cloud AI installation
  documentation](https://qualcomm.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/).


## 2. Generate ONNX files (skip this step if ONNX files are already available)

1.  Set up a virtual environment for ONNX generation
```
python3.12 -m venv env_sdxl_onnxgen
source ./env_sdxl_onnxgen/bin/activate
pip install -r requirements.txt
```

2.  Create a folder for caching HuggingFace model downloads, and export the environment variable HF_HOME
```
mkdir hf_home
export HF_HOME=$(pwd)/hf_home
```

3. Install diffusers from source after patching for ONNX file generation
```
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-dev
cd diffusers-dev
git apply --reject --whitespace=fix ../attention_patch.patch
pip install .
cd ..
```

4. Generate ONNX files for text encoder and text encoder 2
```
python generate_onnx_files.py onnx_files/ --export_text_encoder --export_text_encoder_2
```

5. Generate ONNX files for UNet for batchsize = 2
```
python generate_onnx_files.py onnx_files/unet_bs2/ --export_unet
```

6. Modify attention_processor.py with blocksize = 256 directly in the installed diffusers library
```
sed -i 's/query_block_size = 128/query_block_size = 256/g' ./env_sdxl_onnxgen/lib/python3.12/site-packages/diffusers/models/attention_processor.py
```

7. Generate ONNX files for UNet for batchsize = 1
```
python generate_onnx_files.py onnx_files/unet_bs1/ --export_unet
```

8. Clone SDXL repository from HuggingFace to get the VAE Decoder ONNX file
```
export GIT_LFS_SKIP_SMUDGE=1
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 HF_repo/stabilityai/stable-diffusion-xl-base-1.0
cd HF_repo/stabilityai/stable-diffusion-xl-base-1.0
git lfs pull -I vae_decoder/model.onnx
rm -rf .git/lfs # optional to save space
cd ../../../
```

9. Fix the FP16 overflow error in `vae_decoder` part of the model
```
python fix_vae_decoder_onnx.py --scaling-factor 128 --model-path ./HF_repo/stabilityai/stable-diffusion-xl-base-1.0/vae_decoder/model.onnx
mkdir onnx_files/vae_decoder/
mv ./HF_repo/stabilityai/stable-diffusion-xl-base-1.0/vae_decoder/model_fixed_128.onnx ./onnx_files/vae_decoder/
```


## 3. Compile the models (skip this step if QPCs are already available)
```
bash -x compile_models.sh
```

## 4. Run the end-to-end SDXL inference script

1. Set up a separate virtual environment for running SDXL (this would conflict with code changes for ONNX generation - hence separate environment)

```
python3.12 -m venv env_sdxl_infer
source ./env_sdxl_infer/bin/activate
pip install -r requirements.txt
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2. Export HF_HOME environment variable again
```
export HF_HOME=$(pwd)/hf_home
```

3.  Re-install diffusers from source after patching the SDXL pipeline for inference
```
rm -rf diffusers-dev
git clone --depth 1 --branch v0.24.0 https://github.com/huggingface/diffusers.git diffusers-dev
cd diffusers-dev
git apply --reject --whitespace=fix ../pipeline_patch.patch
pip install .
cd ..
```

4. Run the SDXL inference on AI 100 devices.

For host inference without `sudo`, add yourself to the `qaic` group:

```
sudo usermod -aG qaic "$USER"
```

Log out and log back in to apply the membership. The `text_encoder`,
`text_encoder_2`, and `vae_decoder` binaries are the same whether running on
1 device or 2 devices. You can pass in `--use_latents latents.pt` argument to
use a pre-defined latents PyTorch file.


For running on 1 device, pass in `device_id` argument, and `unet` argument point to the batchsize=2 QPC
```
./env_sdxl_infer/bin/python run_sdxl.py \
    --prompt <prompt_in_quotes> \
    --negative_prompt <negative_prompt_in_quotes> \
    --device_id 0 \
    --text_encoder ./qpc/text_encoder/ \
    --text_encoder_2 ./qpc/text_encoder_2/ \
    --unet ./qpc/unet-bs2/ \
    --vae_decoder ./qpc/vae_decoder/
```


For running on 2 devices, pass in `device_id` and `device_id2` arguments, and `unet` argument point to the batchsize=1 QPC
```
./env_sdxl_infer/bin/python run_sdxl.py \
    --prompt <prompt_in_quotes> \
    --negative_prompt <negative_prompt_in_quotes> \
    --device_id 0 \
    --device_id2 1 \
    --text_encoder ./qpc/text_encoder/ \
    --text_encoder_2 ./qpc/text_encoder_2/ \
    --unet ./qpc/unet-bs1/ \
    --vae_decoder ./qpc/vae_decoder/
```

