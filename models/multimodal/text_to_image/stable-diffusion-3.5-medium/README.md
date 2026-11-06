### Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
### SPDX-License-Identifier: BSD-3-Clause-Clear

# Instructions to run SD3.5 on Cloud AI 100

The instructions below are to run the [Stable Diffusion 3.5 model](stabilityai/stable-diffusion-3.5-medium) on Cloud AI 100. Compile time parameters may need to be adjusted for different cards and different SDKs.

## Containerized workflow (recommended)

Needs Docker with the Compose plugin, about 100 GB of free disk in this
folder (ONNX models, QPCs, and downloaded weights persist here), and access
to the gated stabilityai/stable-diffusion-3.5-medium repository: run
`huggingface-cli login` on the host, the containers mount
`~/.cache/huggingface/token` read-only.
The Platform SDK driver must be installed on the host.

From the repository root, add [qaic-container](../../../../utils/README.md) to
PATH, then prepare this recipe's output directories:
```bash
export PATH="$PWD/utils:$PATH"
cd models/multimodal/text_to_image/stable-diffusion-3.5-medium
mkdir -p cache onnx_files qpc compile_logs images
qaic-container build
qaic-container run --rm export
qaic-container run --rm inference --prompt "<your prompt>" --neg_prompt "<your negative prompt>"
```

The containers run as your host user and receive the host's `qaic` GID for
device access. Existing output directories must be writable by your account.
Re-run `qaic-container build` after changing the recipe or switching hosts or
users.

`export` covers steps 1 to 3 below: it downloads the gated weights into
`cache/`, generates the ONNX models for the CLIP and T5 text encoders,
transformer, and VAE decoder, then compiles all five QPCs.
`inference` runs on device 0; `--prompt` and `--neg_prompt` are
both required. It writes one image; add
`--num-warmup-iters 3 --num-repeat-iters 3` for the full benchmark. Extra
flags after `inference` pass through to `main.py` (later flags win). See
`compose.yaml` for details.

The remaining sections describe the manual workflow for hosts without
containers.

## 1. Download model

1. Setup environment varialble
```
mkdir cache
export HF_HOME=cache
export HF_TOKEN=<your-huggingface-auth-token>
```

2. Follow [instructions on HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) to gain access to model.

## 2. Generate onnx files and compile for binaries

1.  Set up a virtual environment for ONNX generation and compilation
```
python3.12 -m venv env_onnx
source ./env_onnx/bin/activate
pip install -r requirements.txt
```

2.  Create a folder for caching HuggingFace model downloads
```
mkdir compile_logs
mkdir qpc
touch run.sh
```

3. Install diffusers from source after patching for ONNX file generation
```
git clone --depth 1 --branch v0.31.0 https://github.com/huggingface/diffusers.git diffusers-onnx
cd diffusers-onnx
git apply --reject --whitespace=fix ../patches/attention_patch.patch
pip install .
cd ..
```

4. Install transformers from source (for T5 text_encoder_3 only)
```
git clone -b v4.41.2 https://github.com/huggingface/transformers.git transformers-dev
cd transformers-dev
git apply --reject --whitespace=fix ../patches/transformer_patch.patch
pip install .
cd ..
```

5. Generate ONNX files and model binaries
```
bash run_config_gen.sh
```

## 3. Run the end-to-end SD3 inference

1. Set up a separate virtual environment
```
python3.12 -m venv env_pipeline
source ./env_pipeline/bin/activate
pip install -r requirements.txt
pip install --force-reinstall /opt/qti-aic/dev/lib/x86_64/qaic-0.0.1-py3-none-any.whl
```

2.  Re-install diffusers from source after patching the SD3 pipeline for inference
```
git clone --depth 1 --branch v0.31.0 https://github.com/huggingface/diffusers.git diffusers-pipeline
cd diffusers-pipeline
git apply --reject --whitespace=fix ../patches/pipeline_patch.patch
pip install .
cd ..
```

3. Run the inference on AI 100 devices.

For host inference without `sudo`, add yourself to the `qaic` group:

```
sudo usermod -aG qaic "$USER"
```

Log out and log back in to apply the membership.

```
bash run_config_inference.sh "<positive prompt>" "<negative prompt>"
```

## 4. Python interface

```
source ./env_pipeline/bin/activate
```

```python
from model import QAICStableDiffusion3

model = QAICStableDiffusion3()
prompt = 'photo of 8k ultra realistic harbour, port, boats, sunset, beautiful light, golden hour, full of colour, cinematic lighting, battered, trending on artstation, 4k, hyperrealistic, focused, extreme details, cinematic, masterpiece'
image = model.generate(prompt, guidance=7.0)[0]
image.save('harbor.png')
```

