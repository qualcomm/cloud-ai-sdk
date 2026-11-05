####################################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
####################################################################################################

import os
import sys
import argparse
import torch
import numpy as np
import onnx
import onnxruntime
import pdb
from device_utils import *
from onnx_generation.generate_onnx_files import convert_models


def main(args):
    # check device status and cores available (only needed when running on the
    # accelerator; ONNX export and QPC compilation are host-side operations)
    if args.run_only:
        check_device(args.device_id, args.num_cores)
        check_device(args.device_id_2, args.num_cores)

    # model configs
    batch_size = args.batch_size
    image_size = args.image_size
    block_size_deep = args.block_size_deep
    block_size_shallow = args.block_size_shallow
    precision = args.precision
    latent_size = 128 if image_size == 1024 else 64
    seq_len = 77
    latent_channels = 4
    encoder_hidden_states_dim = 2048
    text_embeds_dim = 1280
    time_ids_size = 6

    ################################# onnx generation ##############################
    if args.generate_onnx:
        onnx_args = {
                "output_path": "onnx_files/",
                "model_path": args.model_path,
                "opset": args.opset,
                "together": args.together,
            }
        if not args.together:
            if args.onnx_text_encoder:
                onnx_args["export_text_encoder"] = True
                onnx_args["export_text_encoder_2"] = True
            if args.onnx_unet:
                onnx_args["export_unet"] = True
                onnx_args["unet_type"] = args.unet_type
            #onnx_vae = True if args.vae_type == "tiny" else False
            if args.onnx_vae:
                onnx_args["export_vae"] = True
        convert_models(
            image_size = image_size,
            block_size_deep = block_size_deep,
            block_size_shallow = block_size_shallow,
            latent_size = latent_size,
            seq_len = seq_len,
            vae_type = args.vae_type,
            latent_channels = latent_channels,
            encoder_hidden_states_dim = encoder_hidden_states_dim,
            text_embeds_dim = text_embeds_dim,
            time_ids_size = time_ids_size,
            **onnx_args
        )
        
    ################################# compilation #################################
    if not args.run_only:
        map2device(
            binary_dir = "./qpc/",
            log_dir = "./compile_logs/",
            batch_size = batch_size,
            image_size = image_size,
            block_size_deep = block_size_deep,
            block_size_shallow = block_size_shallow,
            seq_len = seq_len,
            vae_type = args.vae_type,
            latent_channels = latent_channels,
            latent_size = latent_size,
            num_cores = args.num_cores,
            unet_type = args.unet_type,
            unet_mos_deep = args.unet_mos_deep,
            unet_ols_deep = args.unet_ols_deep,
            unet_mos_shallow = args.unet_mos_shallow,
            unet_ols_shallow = args.unet_ols_shallow,
            vae_mos = args.vae_mos,
            vae_ols = args.vae_ols,
            compile_text_encoder = args.compile_text_encoder,
            compile_unet = args.compile_unet,
            compile_vae = args.compile_vae,
            precision = precision,
            together = args.together,
        )
        sys.exit()

    ################################ Execution ###################################
    suffix_deep = f"{block_size_deep}b_{image_size}i"
    suffix_shallow = f"{block_size_shallow}b_{image_size}i"

    text_encoder_qpc = f"./qpc/text_encoder_{suffix_deep}_{args.num_cores}c_{batch_size}b"
    text_encoder_2_qpc = f"./qpc/text_encoder_2_{suffix_deep}_{args.num_cores}c_{batch_size}b"
    unet_qpc = f"./qpc/unet_{suffix_deep}_{args.num_cores}c_{batch_size}b_{args.unet_mos_deep}m_{args.unet_ols_deep}o_dc_deep_{precision[2]}"
    unet_2_qpc = f"./qpc/unet_{suffix_shallow}_{args.num_cores}c_{batch_size}b_{args.unet_mos_shallow}m_{args.unet_ols_shallow}o_dc_shallow_{precision[2]}"
    vae_qpc = f"./qpc/vae_decoder_{suffix_deep}_{args.vae_type}_{args.num_cores}c_{batch_size}b_{args.vae_mos}m_{args.vae_ols}o_{precision[3]}"

    benchmark(
        model_name = args.model_path,
        prompt = args.prompt,
        use_latents = args.use_latents,
        negative_prompt = args.negative_prompt,
        text_encoder = text_encoder_qpc,
        text_encoder_2 = text_encoder_2_qpc,
        unet = unet_qpc,
        unet_2 = unet_2_qpc,
        vae_decoder = vae_qpc,
        image_size = image_size,
        num_steps = args.num_steps,
        device_id = args.device_id,
        device_id_2 = args.device_id_2,
        vae_type = args.vae_type,
        num_warmup_iters = args.num_warmup_iters,
        num_repeat_iters = args.num_repeat_iters,
        cache_interval = args.cache_interval,
    )

        
def check_positive(arg_in):
    try:
        if int(arg_in) <= 0:
            raise ValueError(f"Expected positive integer, received '{int(arg_in)}'")
    except ValueError:
        raise ValueError(f"Expected integer, received '{arg_in}'")
    return int(arg_in)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Running SDXL Turbo on Qualcomm AIC100")
    parser.add_argument(
        "--model-path", "-m",
        required=True,
        help="Model name to download from Hugging Face. Try bert-base-cased for instance.",
    )
    parser.add_argument(
        "--opset",  type=check_positive,
        default=16,
        help="ONNX opset. Default <16>",
    )
    parser.add_argument(
        "--prompt", type=str,
        help="Prompt.",
    )
    parser.add_argument(
        "--negative-prompt", type=str,
        help="Prompt.",
    )
    parser.add_argument(
        "--use-latents", type=str,
        help="Latents.",
    )
    parser.add_argument(
        "--unet-type", type=str,
        choices=["deep", "shallow"],
        help="Type of Unet for DeepCache.",
    )
    parser.add_argument(
        "--vae-type", type=str,
        choices=["vae", "tiny"],
        help="Which type of VAE to use.",
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--image-size", type=int,
        choices=[512, 1024],
        help="Generated image dimension.",
    )
    parser.add_argument(
        "--block-size-deep", type=int,
        help="Block size for block attention (used for naming only).",
    )
    parser.add_argument(
        "--block-size-shallow", type=int,
        help="Block size for block attention for shallow UNet (used for naming only).",
    )
    parser.add_argument(
        "--num-cores", "-c", type=int,
        help="Number of AIC100 cores to compile the model for. Default <16> ",
    )
    parser.add_argument(
        "--unet-ols-deep", type=int,
        help="Overlap split factor for deep UNet.",
    )
    parser.add_argument(
        "--unet-ols-shallow", type=int,
        help="Overlap split factor for shallow UNet.",
    )
    parser.add_argument(
        "--vae-ols", type=int,
        help="Overlap split factor for VAE.",
    )
    parser.add_argument(
        "--unet-mos-deep", type=str,
        help="Maximum output channel split for deep UNet.",
    )
    parser.add_argument(
        "--unet-mos-shallow", type=str,
        help="Maximum output channel split for shallow UNet.",
    )
    parser.add_argument(
        "--vae-mos", type=str,
        help="Maximum output channel split for VAE.",
    )
    parser.add_argument(
        "--precision",  type=lambda precision: precision.split(","),
        help="Precision to compile for each components.",
    )
    parser.add_argument(
        "--device-id",  type=int,
        help="AIC100 device ID.",
    )
    parser.add_argument(
        "--device-id-2",  type=int,
        help="AIC100 device ID.",
    )
    parser.add_argument(
        "--num-steps",  type=int,
        help="# of inference steps",
    )
    parser.add_argument(
        "--num-warmup-iters",  type=int,
        help="# of warmup iterations",
    )
    parser.add_argument(
        "--num-repeat-iters",  type=int,
        help="# of repeating iterations for benchmarking",
    )
    parser.add_argument(
        "--cache-interval",  type=int,
        help="Cache interval for DeepCache",
    )
    parser.add_argument('--together', 
                        action='store_true',
                        help="Whether to deploy different modules together")
    parser.add_argument('--generate-onnx', 
                        action='store_true',
                        help="Whether to generate onnx files")
    parser.add_argument('--onnx-text-encoder', 
                        action='store_true',
                        help="Generate onnx file for text encoders")
    parser.add_argument('--onnx-unet', 
                        action='store_true',
                        help="Generate onnx file for UNet")
    parser.add_argument('--onnx-vae', 
                        action='store_true',
                        help="Generate onnx file for VAE")
    parser.add_argument('--compile-text-encoder', 
                        action='store_true',
                        help="Compile for text encoders")
    parser.add_argument('--compile-unet', 
                        action='store_true',
                        help="Compile for UNet")
    parser.add_argument('--compile-vae', 
                        action='store_true',
                        help="Compile for VAE")
    parser.add_argument('--run-only', 
                        action='store_true',
                        help="Only run inference")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
