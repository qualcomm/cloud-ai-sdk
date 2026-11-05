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
    # check device status and cores available
    if args.run_only:
        check_device(args.device_id, args.num_cores)

    # model configs
    batch_size = args.batch_size
    image_size = args.image_size
    block_size = args.block_size
    latent_size = 128 if image_size == 1024 else 64
    seq_len = 77
    t5_seq_len = 256
    latent_channels = 16
    joint_attention_dim = 4096
    pooled_projection_dim = 2048
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
            if args.onnx_text_encoder_3:
                onnx_args["export_text_encoder_3"] = True
            if args.onnx_transformer:
                onnx_args["export_transformer"] = True
            if args.onnx_vae:
                onnx_args["export_vae"] = True
        convert_models(
            image_size = image_size,
            block_size = block_size,
            latent_size = latent_size,
            seq_len = seq_len,
            t5_seq_len = t5_seq_len,
            vae_type = args.vae_type,
            latent_channels = latent_channels,
            joint_attention_dim = joint_attention_dim,
            pooled_projection_dim = pooled_projection_dim,
            **onnx_args
        )
        
    ################################# compilation #################################
    if not args.run_only:
        map2device(
            binary_dir = "./qpc/",
            log_dir = "./compile_logs/",
            batch_size = batch_size,
            image_size = image_size,
            block_size = block_size,
            seq_len = seq_len,
            t5_seq_len = t5_seq_len,
            vae_type = args.vae_type,
            latent_channels = latent_channels,
            latent_size = latent_size,
            num_cores = args.num_cores,
            transformer_mos = args.transformer_mos,
            transformer_ols = args.transformer_ols,
            vae_mos = args.vae_mos,
            vae_ols = args.vae_ols,
            compile_text_encoder = args.compile_text_encoder,
            compile_text_encoder_3 = args.compile_text_encoder_3,
            compile_transformer = args.compile_transformer,
            compile_vae = args.compile_vae,
            together = args.together,
        )
        sys.exit()

    ################################ Execution ###################################
    suffix = f"{block_size}b_{image_size}i"
    benchmark(
        model_name = args.model_path,
        prompt = args.prompt,
        negative_prompt = args.neg_prompt,
        text_encoder = f"./qpc/text_encoder_{suffix}_{args.num_cores}c_{batch_size}b",
        text_encoder_2 = f"./qpc/text_encoder_2_{suffix}_{args.num_cores}c_{batch_size}b",
        text_encoder_3 = f"./qpc/text_encoder_3_{suffix}_{args.num_cores}c_{batch_size}b" if args.text_encoder_3 else None,
        transformer = f"./qpc/transformer_{suffix}_{args.num_cores}c_{batch_size}b_{args.transformer_mos}m_{args.transformer_ols}o",
        sdxl_vae_decoder = f"./qpc/vae_decoder_{suffix}_{args.vae_type}_{args.num_cores}c_{batch_size}b_{args.vae_mos}m_{args.vae_ols}o",
        image_size = image_size,
        num_steps = args.num_steps,
        guidance = args.guidance,
        device_id = args.device_id,
        device_id2 = args.device_id2,
        vae_type = args.vae_type,
        num_warmup_iters = args.num_warmup_iters,
        num_repeat_iters = args.num_repeat_iters,
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
        default="",
        help="Prompt.",
    )
    parser.add_argument(
        "--neg_prompt", type=str,
        default="",
        help="Negative prompt.",
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
        "--block-size", type=int,
        help="Block size for block attention (used for naming only).",
    )
    parser.add_argument(
        "--num-cores", "-c", type=int,
        help="Number of AIC100 cores to compile the model for. Default <16> ",
    )
    parser.add_argument(
        "--transformer-ols", type=int,
        help="Overlap split factor for transformer.",
    )
    parser.add_argument(
        "--vae-ols", type=int,
        help="Overlap split factor for VAE.",
    )
    parser.add_argument(
        "--transformer-mos", type=str,
        help="Maximum output channel split for transformer.",
    )
    parser.add_argument(
        "--vae-mos", type=str,
        help="Maximum output channel split for VAE.",
    )
    parser.add_argument(
        "--device-id",  type=int,
        help="AIC100 device ID.",
    )
    parser.add_argument(
        "--device-id2",  type=int,
        help="AIC100 device ID.",
    )
    parser.add_argument(
        "--num-steps",  type=int,
        help="# of inference steps",
    )
    parser.add_argument(
        "--guidance",  type=float,
        default=0.,
        help="scale of classifier-free guidance",
    )
    parser.add_argument(
        "--num-warmup-iters",  type=int,
        help="# of warmup iterations",
    )
    parser.add_argument(
        "--num-repeat-iters",  type=int,
        help="# of repeating iterations for benchmarking",
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
    parser.add_argument('--onnx-text-encoder-3', 
                        action='store_true',
                        help="Generate onnx file for text encoder 3")
    parser.add_argument('--onnx-transformer', 
                        action='store_true',
                        help="Generate onnx file for MMDiT")
    parser.add_argument('--onnx-vae', 
                        action='store_true',
                        help="Generate onnx file for VAE")
    parser.add_argument('--compile-text-encoder', 
                        action='store_true',
                        help="Compile for text encoders")
    parser.add_argument('--compile-text-encoder-3', 
                        action='store_true',
                        help="Compile for text encoder 3")
    parser.add_argument('--text-encoder-3', 
                        action='store_true',
                        help="Run text encoder 3")
    parser.add_argument('--compile-transformer', 
                        action='store_true',
                        help="Compile for MMDiT")
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
