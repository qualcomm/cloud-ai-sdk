####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import re
import torch
import subprocess
import os, sys, time
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler, AutoencoderTiny
from DeepCache.sdxl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline as DeepCacheStableDiffusionXLPipeline
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
     block_size_deep = 128,
     block_size_shallow = 128,
     seq_len = None,
     latent_channels = None,
     latent_size = None,
     vae_type = "vae",
     num_cores = 1,
     unet_mos_deep = None,
     unet_ols_deep = None,
     unet_mos_shallow = None,
     unet_ols_shallow = None,
     unet_type = "deep",
     vae_mos = None,
     vae_ols = None,
     compile_text_encoder = False,
     compile_unet = False,
     compile_vae = False,
     precision = None,
     together = False,
    ): 
    os.makedirs(binary_dir, exist_ok=True)

    if not together:
        cmd_elements = ["/opt/qti-aic/exec/qaic-compile",
                        f"-aic-hw",
                        f"-aic-hw-version=ai100",
                        "-aic-perf-warnings",
                        "-aic-perf-metrics",
                        "-convert-to-fp16",
                        f"-onnx-define-symbol=batch_size,{batch_size}",
                        f"-stats-batchsize={batch_size}",
                        f"-onnx-define-symbol=sequence_length,{seq_len}",
                        f"-aic-num-cores={num_cores}",
                    ]
        compile_check = False
        if compile_text_encoder:
            name_text_encoder_1 = f"text_encoder_{block_size_deep}b_{image_size}i"
            name_text_encoder_2 = f"text_encoder_2_{block_size_deep}b_{image_size}i"
            cmd_text_encoder_1 = cmd_elements + [
                        f"-m=./onnx_files/{name_text_encoder_1}/model.onnx",
                        f"-aic-binary-dir={binary_dir}{name_text_encoder_1}_{num_cores}c_{batch_size}b",
                    ]
            cmd_text_encoder_2 = cmd_elements + [
                        f"-m=./onnx_files/{name_text_encoder_2}/model.onnx",
                        f"-aic-binary-dir={binary_dir}{name_text_encoder_2}_{num_cores}c_{batch_size}b",
                    ]
            compile_check = True
            execute(cmd_text_encoder_1, f"{log_dir}{name_text_encoder_1}_{num_cores}c_{batch_size}b.log ", 'a')
            execute(cmd_text_encoder_2, f"{log_dir}{name_text_encoder_2}_{num_cores}c_{batch_size}b.log ", 'a')
        if compile_unet:
            block_size = block_size_deep if unet_type == "deep" else block_size_shallow
            unet_mos = unet_mos_deep if unet_type == "deep" else unet_mos_shallow
            unet_ols = unet_ols_deep if unet_type == "deep" else unet_ols_shallow
            name_unet = f"unet_{block_size}b_{image_size}i"
            cmd_unet = cmd_elements + [
                        f"-m=./onnx_files/{name_unet}_dc_{unet_type}/model.onnx",
                        "-onnx-define-symbol=steps,1",
                        f"-onnx-define-symbol=latent_channels,{latent_channels}",
                        f"-onnx-define-symbol=num_channels,320",
                        f"-onnx-define-symbol=height,{latent_size}",
                        f"-onnx-define-symbol=width,{latent_size}",
                        f"-aic-binary-dir={binary_dir}{name_unet}_{num_cores}c_{batch_size}b_{unet_mos}m_{unet_ols}o_dc_{unet_type}_{precision[2]}",
                        f"-custom-IO-list-file=./onnx_files/custom_io_{unet_type}.yaml",
                        f"-retained-state",
                    ]
            if precision[2] == "mx6":
                cmd_unet.append(f"-mxfp6-matmul")
            if unet_mos != '':
                cmd_unet.append(f"-mos={unet_mos}")
            if unet_ols != '':
                cmd_unet.append(f"-ols={unet_ols}")
            execute(cmd_unet, f"{log_dir}{name_unet}_{num_cores}c_{batch_size}b_{unet_mos}m_{unet_ols}o_dc_{unet_type}_{precision[2]}.log ", 'a')
            compile_check = True
        if compile_vae:
            name_vae = f"vae_decoder_{block_size_deep}b_{image_size}i_{vae_type}"
            cmd_vae = cmd_elements + [
                        f"-m=./onnx_files/{name_vae}/model.onnx",
                        f"-onnx-define-symbol=num_channels_latent,{latent_channels}",
                        f"-onnx-define-symbol=height_latent,{latent_size}",
                        f"-onnx-define-symbol=width_latent,{latent_size}",
                        "-aic-enable-depth-first -aic-depth-first-mem=32",
                        f"-aic-binary-dir={binary_dir}{name_vae}_{num_cores}c_{batch_size}b_{vae_mos}m_{vae_ols}o_{precision[3]}",
                    ]
            if precision[2] == "mx6":
                cmd_vae.append(f"-mxfp6-matmul")
            if vae_mos != '':
                cmd_vae.append(f"-mos={vae_mos}")
            if vae_ols != '':
                cmd_vae.append(f"-ols={vae_ols}")
            execute(cmd_vae, f"{log_dir}{name_vae}_{num_cores}_{batch_size}b_{vae_mos}m_{vae_ols}o_{precision[3]}.log ", 'a')
            compile_check = True
        if not compile_check:
            raise ValueError("Not compiling anything...")
    else:
        raise NotImplementedError


def benchmark(
        model_name = "",
        prompt = "",
        negative_prompt = "",
        use_latents = None,
        text_encoder = "",
        text_encoder_2 = "",
        unet = "",
        unet_2 = "",
        vae_decoder = "",
        image_size = 1024,
        num_steps = 1,
        device_id = 0,
        device_id_2 = 0,
        vae_type = "vae",
        num_warmup_iters = 0,
        num_repeat_iters = 1,
        cache_interval = 1,
    ):
    # check the QPCs
    unet_qpc = unet if unet.endswith('programqpc.bin') else os.path.join(unet,'programqpc.bin')
    assert os.path.isfile(unet_qpc), f"Could not find binary {unet_qpc = }!"
    unet_qpc_2 = unet if unet_2.endswith('programqpc.bin') else os.path.join(unet_2,'programqpc.bin')
    assert os.path.isfile(unet_qpc_2), f"Could not find binary {unet_qpc_2 = }!"
    vae_decoder_qpc = vae_decoder if vae_decoder.endswith('programqpc.bin') else os.path.join(vae_decoder,'programqpc.bin')
    assert os.path.isfile(vae_decoder_qpc), f"Could not find binary {vae_decoder_qpc = }!"
    text_encoder_qpc = text_encoder if text_encoder.endswith('programqpc.bin') else os.path.join(text_encoder,'programqpc.bin')
    assert os.path.isfile(text_encoder_qpc), f"Could not find binary {text_encoder_qpc = }!"
    text_encoder_2_qpc = text_encoder_2 if text_encoder_2.endswith('programqpc.bin') else os.path.join(text_encoder_2,'programqpc.bin')
    assert os.path.isfile(text_encoder_2_qpc), f"Could not find binary {text_encoder_2_qpc = }!"

    # load the latents
    if not use_latents:
        latents = None
    else:
        assert os.path.isfile(use_latents), f"Could not find latents file {use_latents}"
        assert use_latents.endswith('.pt') or use_latents.endswith('.npy'), f"use_latents should be a .pt or .npy file - found {use_latents}!"
        if use_latents.endswith('.pt'):
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch -- latents are tensor data; restricted loading disallows arbitrary globals.
            latents = torch.load(use_latents, weights_only=True)
        elif ause_latents.endswith('.npy'):
            latents = torch.Tensor(np.load(use_latents))
        assert latents.shape == (1,4,128,128), f"Latents should have shape (1,4,128,128), found {latents.shape = }"

    negative_prompt = None if negative_prompt == "" else negative_prompt

    generator = torch.Generator()
            
    # load the model pipeline
    #pipe = AutoPipelineForText2Image.from_pretrained(model_name, 
    pipe = DeepCacheStableDiffusionXLPipeline.from_pretrained(model_name, 
                                             device_id=device_id, 
                                             device_id2=device_id_2, 
                                             unet_qpc=unet_qpc,
                                             unet_qpc_2=unet_qpc_2,
                                             vae_decoder_qpc=vae_decoder_qpc,
                                             text_encoder_qpc=text_encoder_qpc,
                                             text_encoder_2_qpc=text_encoder_2_qpc)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if vae_type == 'tiny':
        print("Use Tiny VAE...")
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl")

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
                      cache_interval=cache_interval,
                      latents=latents).images[0]
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
        generator.manual_seed(i)
        print(f'Benchmark Iteration {i+1}/{num_repeat_iters}:')
        start_time = time.perf_counter()
        images = pipe(prompt=prompt, 
                      negative_prompt=negative_prompt,
                      num_inference_steps=num_steps, 
                      height=image_size,
                      width=image_size,
                      cache_interval=cache_interval,
                      generator=generator,
                      latents=latents).images[0]
        elapsed_time = time.perf_counter() - start_time
        e2e_avg += elapsed_time
        image_filename = f"{output_folder}output_sdxl_device_{device_id}__iteration{i}.jpg"
        images = images.save(image_filename) 
        print(f'E2E total time : {1000.*elapsed_time:.6f} ms', end='\n\n')
    print(f"Averaged E2E time: {1000.*e2e_avg/num_repeat_iters:.6f} ms")
    print(f"Benchmark complete!")
    print("*"*80)

