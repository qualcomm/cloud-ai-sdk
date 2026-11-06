####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import os, sys, time
import numpy as np
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse device IDs.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt')
    parser.add_argument('--negative_prompt', type=str, required=True, help='Negative prompt')
    parser.add_argument('--use_latents', type=str, default=None, help='Latents file (.pt or .npy)')
    parser.add_argument('--device_id', type=int, required=True, help='AI100 Device ID to run inference')
    parser.add_argument('--device_id2', type=int, default=None, help='If 2nd AI100 Device ID is provided, UNet is run in parallel on 2 devices')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=8.0, help='Guidance scale')
    parser.add_argument('--model_name', type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help='Stable Diffusion model ID from HuggingFace')
    parser.add_argument('--num_warmup_iters', type=int, default=3, help='Number of times to repeat the model inference for warmup')
    parser.add_argument('--num_repeat_iters', type=int, default=10, help='Number of times to repeat the model inference for benchmarking')
    parser.add_argument('--text_encoder', type=str, default="./qpc/text_encoder-16c_1.14.0.28", help='Path to Text Encoder QPC')
    parser.add_argument('--text_encoder_2', type=str, default="./qpc/text_encoder_2-16c_1.14.0.28", help='Path to Text Encoder 2 QPC')
    parser.add_argument('--vae_decoder', type=str, default="./qpc/vae_decoder_fixed128_dfs_mos2_ols1_1.14.0.28", help='Path to VAE decoder QPC')
    parser.add_argument('--unet', type=str, required=True, help='Path to UNet QPC (batchsize=2 if 1 device, batchsize=1 if 2 devices)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # check the QPCs
    unet_qpc = args.unet if args.unet.endswith('programqpc.bin') else os.path.join(args.unet,'programqpc.bin')
    assert os.path.isfile(unet_qpc), f"Could not find binary {unet_qpc = }!"
    vae_decoder_qpc = args.vae_decoder if args.vae_decoder.endswith('programqpc.bin') else os.path.join(args.vae_decoder,'programqpc.bin')
    assert os.path.isfile(vae_decoder_qpc), f"Could not find binary {vae_decoder_qpc = }!"
    text_encoder_qpc = args.text_encoder if args.text_encoder.endswith('programqpc.bin') else os.path.join(args.text_encoder,'programqpc.bin')
    assert os.path.isfile(text_encoder_qpc), f"Could not find binary {text_encoder_qpc = }!"
    text_encoder_2_qpc = args.text_encoder_2 if args.text_encoder_2.endswith('programqpc.bin') else os.path.join(args.text_encoder_2,'programqpc.bin')
    assert os.path.isfile(text_encoder_2_qpc), f"Could not find binary {text_encoder_2_qpc = }!"

    # load the latents
    if args.use_latents is None:
        latents = None
    else:
        assert os.path.isfile(args.use_latents), f"Could not find latents file {args.use_latents}"
        assert args.use_latents.endswith('.pt') or args.use_latents.endswith('.npy'), f"use_latents should be a .pt or .npy file - found {args.use_latents}!"
        if args.use_latents.endswith('.pt'):
            # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch -- latents are tensor data; restricted loading disallows arbitrary globals.
            latents = torch.load(args.use_latents, weights_only=True)
        elif args.use_latents.endswith('.npy'):
            latents = torch.Tensor(np.load(args.use_latents))
        assert latents.shape == (1,4,128,128), f"Latents should have shape (1,4,128,128), found {latents.shape = }"
            
    # load the model pipeline
    pipe = DiffusionPipeline.from_pretrained(args.model_name, 
                                             device_id=args.device_id, 
                                             device_id2=args.device_id2,
                                             unet_qpc=unet_qpc,
                                             vae_decoder_qpc=vae_decoder_qpc,
                                             text_encoder_qpc=text_encoder_qpc,
                                             text_encoder_2_qpc=text_encoder_2_qpc)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # warmup entire pipeline for num_warmup_iters - ignore these numbers
    print("*"*80)
    print(f"Starting warmup iterations...")
    for i in range(args.num_warmup_iters):
        print(f"Warmup Iteration {i+1}/{args.num_warmup_iters}: Starting ...")
        start_time = time.perf_counter()
        images = pipe(prompt=args.prompt, 
                      num_inference_steps=args.num_steps, 
                      latents=latents,
                      negative_prompt=args.negative_prompt,
                      guidance_scale=args.guidance_scale).images[0]
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
    for i in range(args.num_repeat_iters):
        print(f'Benchmark Iteration {i+1}/{args.num_repeat_iters}:')
        start_time = time.perf_counter()
        images = pipe(prompt=args.prompt, 
                      num_inference_steps=args.num_steps, 
                      latents=latents,
                      negative_prompt=args.negative_prompt,
                      guidance_scale=args.guidance_scale).images[0]
        elapsed_time = time.perf_counter() - start_time
        if args.device_id2 is None:
            image_filename = f"{output_folder}output_sdxl_device_{args.device_id}__iteration{i}.jpg"
        else:
            image_filename = f"{output_folder}output_sdxl_devices_{args.device_id}_{args.device_id2}__iteration{i}.jpg"
        images = images.save(image_filename) 
        print(f'E2E total time : {1000.*elapsed_time:.6f} ms', end='\n\n')
    print(f"Benchmark complete!")
    print("*"*80)
    
