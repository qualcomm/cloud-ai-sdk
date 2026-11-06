####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################
import re
import subprocess
import os, sys, time
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler, StableDiffusion3Pipeline
from utils import *


# checks device status and cores available
def check_device(device_id, cores):
    try:
        result = subprocess.run(
            ["/opt/qti-aic/tools/qaic-util", "-d", str(device_id), "-q"],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Unable to query device {device_id}: {exc}") from exc

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Device {device_id} query failed: {detail}")

    free_match = re.search(r"nsp\s+free:\s*(\d+)", result.stdout, re.IGNORECASE)
    status_match = re.search(r"status:\s*([^\r\n]+)", result.stdout, re.IGNORECASE)
    if free_match is None or status_match is None:
        raise RuntimeError(f"Cannot parse readiness for device {device_id}")

    nsp_free = int(free_match.group(1))
    status = status_match.group(1).strip().lower()
    if status != "ready":
        raise RuntimeError(f"Device {device_id} not ready: {status}")
    if nsp_free < cores:
        raise RuntimeError(
            f"Insufficient resources: {cores} NSP cores required, {nsp_free} available"
        )

def map2device(
     binary_dir = "./qpc/",
     log_dir = "./compile_logs/",
     batch_size = 1,
     image_size = 512,
     block_size = 128,
     seq_len = None,
     t5_seq_len = None,
     latent_channels = None,
     latent_size = None,
     vae_type = "vae",
     num_cores = 1,
     transformer_mos = None,
     transformer_ols = None,
     vae_mos = None,
     vae_ols = None,
     compile_text_encoder = False,
     compile_text_encoder_3 = False,
     compile_transformer = False,
     compile_vae = False,
     together = False,
    ): 
    os.makedirs(binary_dir, exist_ok=True)

    if not compile_text_encoder_3:
        t5_seq_len = seq_len # use CLIP encoder sequence length

    if not together:
        cmd_elements = ["/opt/qti-aic/exec/qaic-compile",
                        f"-aic-hw",
                        f"-aic-hw-version=ai100",
                        "-aic-perf-warnings",
                        "-aic-perf-metrics",
                        "-convert-to-fp16",
                        f"-onnx-define-symbol=batch_size,{batch_size}",
                        f"-stats-batchsize={batch_size}",
                        f"-aic-num-cores={num_cores}",
                    ]
        compile_check = False
        if compile_text_encoder:
            name_text_encoder_1 = f"text_encoder_{block_size}b_{image_size}i"
            name_text_encoder_2 = f"text_encoder_2_{block_size}b_{image_size}i"
            cmd_text_encoder_1 = cmd_elements + [
                        f"-onnx-define-symbol=sequence_length,{seq_len}",
                        f"-m=./onnx_files/{name_text_encoder_1}/model.onnx",
                        f"-aic-binary-dir={binary_dir}{name_text_encoder_1}_{num_cores}c_{batch_size}b",
                    ]
            cmd_text_encoder_2 = cmd_elements + [
                        f"-onnx-define-symbol=sequence_length,{seq_len}",
                        f"-m=./onnx_files/{name_text_encoder_2}/model.onnx",
                        f"-aic-binary-dir={binary_dir}{name_text_encoder_2}_{num_cores}c_{batch_size}b",
                    ]
            compile_check = True
            execute(cmd_text_encoder_1, f"{log_dir}{name_text_encoder_1}_{num_cores}c_{batch_size}b.log ", 'a')
            execute(cmd_text_encoder_2, f"{log_dir}{name_text_encoder_2}_{num_cores}c_{batch_size}b.log ", 'a')
        if compile_text_encoder_3:
            name_text_encoder_3 = f"text_encoder_3_{block_size}b_{image_size}i"
            cmd_text_encoder_3 = cmd_elements + [
                        f"-onnx-define-symbol=sequence_length,{t5_seq_len}",
                        f"-m=./onnx_files/{name_text_encoder_3}/model.onnx",
                        f"-aic-binary-dir={binary_dir}{name_text_encoder_3}_{num_cores}c_{batch_size}b",
                    ]
            compile_check = True
            execute(cmd_text_encoder_3, f"{log_dir}{name_text_encoder_3}_{num_cores}c_{batch_size}b.log ", 'a')
        if compile_transformer:
            name_transformer = f"transformer_{block_size}b_{image_size}i"
            cmd_transformer = cmd_elements + [
                        f"-m=./onnx_files/{name_transformer}/model.onnx",
                        "-onnx-define-symbol=steps,1",
                        f"-onnx-define-symbol=sequence_length,{seq_len+t5_seq_len}",
                        f"-onnx-define-symbol=latent_channels,{latent_channels}",
                        f"-onnx-define-symbol=latent_height,{latent_size}",
                        f"-onnx-define-symbol=latent_width,{latent_size}",
                        f"-aic-binary-dir={binary_dir}{name_transformer}_{num_cores}c_{batch_size}b_{transformer_mos}m_{transformer_ols}o",
                    ]
            if transformer_mos != '':
                cmd_transformer.append(f"-mos={transformer_mos}")
            if transformer_ols != '':
                cmd_transformer.append(f"-ols={transformer_ols}")
            execute(cmd_transformer, f"{log_dir}{name_transformer}_{num_cores}c_{batch_size}b_{transformer_mos}m_{transformer_ols}o.log ", 'a')
            compile_check = True
        if compile_vae:
            name_vae = f"vae_decoder_{block_size}b_{image_size}i_{vae_type}"
            cmd_vae = cmd_elements + [
                        f"-m=./onnx_files/{name_vae}/model.onnx",
                        f"-onnx-define-symbol=latent_channels,{latent_channels}",
                        f"-onnx-define-symbol=latent_height,{latent_size}",
                        f"-onnx-define-symbol=latent_width,{latent_size}",
                        "-aic-enable-depth-first -aic-depth-first-mem=32",
                        f"-aic-binary-dir={binary_dir}{name_vae}_{num_cores}c_{batch_size}b_{vae_mos}m_{vae_ols}o",
                    ]
            if vae_mos != '':
                cmd_vae.append(f"-mos={vae_mos}")
            if vae_ols != '':
                cmd_vae.append(f"-ols={vae_ols}")
            execute(cmd_vae, f"{log_dir}{name_vae}_{num_cores}_{batch_size}b_{vae_mos}m_{vae_ols}o.log ", 'a')
            compile_check = True
        if not compile_check:
            raise ValueError("Not compiling anything...")
    else:
        raise NotImplementedError


def benchmark(
        model_name = "",
        prompt = "",
        negative_prompt = "",
        text_encoder = "",
        text_encoder_2 = "",
        text_encoder_3 = "",
        transformer = "",
        sdxl_vae_decoder = "",
        image_size = 1024,
        num_steps = 1,
        guidance = 0.,
        device_id = 0,
        device_id2 = 1,
        vae_type = "vae",
        num_warmup_iters = 0,
        num_repeat_iters = 1,
    ):
    # check the QPCs
    transformer_qpc = transformer if transformer.endswith('programqpc.bin') else os.path.join(transformer,'programqpc.bin')
    assert os.path.isfile(transformer_qpc), f"Could not find binary {transformer_qpc = }!"
    vae_decoder_sdxl_qpc = sdxl_vae_decoder if sdxl_vae_decoder.endswith('programqpc.bin') else os.path.join(sdxl_vae_decoder,'programqpc.bin')
    assert os.path.isfile(vae_decoder_sdxl_qpc), f"Could not find binary {vae_decoder_sdxl_qpc = }!"
    text_encoder_qpc = text_encoder if text_encoder.endswith('programqpc.bin') else os.path.join(text_encoder,'programqpc.bin')
    assert os.path.isfile(text_encoder_qpc), f"Could not find binary {text_encoder_qpc = }!"
    text_encoder_2_qpc = text_encoder_2 if text_encoder_2.endswith('programqpc.bin') else os.path.join(text_encoder_2,'programqpc.bin')
    assert os.path.isfile(text_encoder_2_qpc), f"Could not find binary {text_encoder_2_qpc = }!"

    # load the latents
    latents = None
            
    # load the model pipeline
    if text_encoder_3:
        text_encoder_3_qpc = text_encoder_3 if text_encoder_3.endswith('programqpc.bin') else os.path.join(text_encoder_3,'programqpc.bin')
        assert os.path.isfile(text_encoder_3_qpc), f"Could not find binary {text_encoder_3_qpc = }!"
        pipe = StableDiffusion3Pipeline.from_pretrained(
                                            model_name, 
                                            device_id=device_id, 
                                            device_id2=device_id2, 
                                            transformer_qpc=transformer_qpc,
                                            vae_decoder_qpc=vae_decoder_sdxl_qpc,
                                            text_encoder_qpc=text_encoder_qpc,
                                            text_encoder_2_qpc=text_encoder_2_qpc,
                                            text_encoder_3_qpc=text_encoder_3_qpc,
        )
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
                                            model_name, 
                                            device_id=device_id, 
                                            device_id2=device_id2, 
                                            transformer_qpc=transformer_qpc,
                                            vae_decoder_qpc=vae_decoder_sdxl_qpc,
                                            text_encoder_qpc=text_encoder_qpc,
                                            text_encoder_2_qpc=text_encoder_2_qpc,
                                            text_encoder_3=None,
                                            tokenizer_3=None,
        )
    if vae_type == 'tiny':
        print("Use Tiny VAE...")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3")

    # warmup entire pipeline for num_warmup_iters - ignore these numbers
    print("*"*80)
    print(f"Starting warmup iterations...")
    for i in range(num_warmup_iters):
        print(f"Warmup Iteration {i+1}/{num_warmup_iters}: Starting ...")
        start_time = time.perf_counter()
        images = pipe(prompt=prompt, 
                      negative_prompt=negative_prompt,
                      num_inference_steps=num_steps, 
                      height=image_size,
                      width=image_size,
                      latents=latents,
                      vae_type=vae_type,
                      guidance_scale=guidance).images[0]
        elapsed_time = time.perf_counter() - start_time
        print(f'E2E total time : {1000.*elapsed_time:.6f} ms', end='\n\n')
    print(f"Warmup complete!")
    print("*"*80)
    
    # save image to folder
    output_folder = "./images/"
    os.makedirs(output_folder, exist_ok=True)
    # benchmark entire pipeline for num_repeat_iters
    print("*"*80)
    print(f"Starting benchmark iterations...")
    e2e_avg = 0
    for i in range(num_repeat_iters):
        print(f'Benchmark Iteration {i+1}/{num_repeat_iters}:')
        start_time = time.perf_counter()
        images = pipe(prompt=prompt, 
                      negative_prompt=negative_prompt,
                      num_inference_steps=num_steps, 
                      height=image_size,
                      width=image_size,
                      latents=latents,
                      vae_type=vae_type,
                      guidance_scale=guidance).images[0]
        elapsed_time = time.perf_counter() - start_time
        e2e_avg += elapsed_time
        image_filename = f"{output_folder}output_sdxl_device_{device_id}__iteration{i}.jpg"
        images = images.save(image_filename) 
        print(f'E2E total time : {1000.*elapsed_time:.6f} ms', end='\n\n')
    print(f"Averaged E2E time: {1000.*e2e_avg/num_repeat_iters:.6f} ms")
    print(f"Benchmark complete!")
    print("*"*80)

