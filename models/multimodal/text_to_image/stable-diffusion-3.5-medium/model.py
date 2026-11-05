# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
import torch

from diffusers import StableDiffusion3Pipeline

class QAICStableDiffusion3:
    def __init__(self, model_id = 'stabilityai/stable-diffusion-3.5-medium', device_id=0, device_id_2=1):
        sdxl_vae_decoder = './qpc/vae_decoder_64b_1024i_vae_16c_1b_2m_1o/programqpc.bin'
        text_encoder = './qpc/text_encoder_64b_1024i_16c_1b/programqpc.bin'
        transformer = './qpc/transformer_64b_1024i_16c_1b_1m_2o/programqpc.bin'
        text_encoder_2 = './qpc/text_encoder_2_64b_1024i_16c_1b/programqpc.bin'

        text_encoder_3 = './qpc/text_encoder_3_64b_1024i_16c_1b/programqpc.bin'

        # check the QPCs
        transformer_qpc = transformer if transformer.endswith('programqpc.bin') else os.path.join(transformer,'programqpc.bin')
        assert os.path.isfile(transformer_qpc), f"Could not find binary {transformer_qpc = }!"
        vae_decoder_sdxl_qpc = sdxl_vae_decoder if sdxl_vae_decoder.endswith('programqpc.bin') else os.path.join(sdxl_vae_decoder,'programqpc.bin')
        assert os.path.isfile(vae_decoder_sdxl_qpc), f"Could not find binary {vae_decoder_sdxl_qpc = }!"
        text_encoder_qpc = text_encoder if text_encoder.endswith('programqpc.bin') else os.path.join(text_encoder,'programqpc.bin')
        assert os.path.isfile(text_encoder_qpc), f"Could not find binary {text_encoder_qpc = }!"
        text_encoder_2_qpc = text_encoder_2 if text_encoder_2.endswith('programqpc.bin') else os.path.join(text_encoder_2,'programqpc.bin')
        assert os.path.isfile(text_encoder_2_qpc), f"Could not find binary {text_encoder_2_qpc = }!"

        self.vae_type = "vae"

        # load the latents
        self.latents = None

        # load the model pipeline
        if text_encoder_3:
            text_encoder_3_qpc = text_encoder_3 if text_encoder_3.endswith('programqpc.bin') else os.path.join(text_encoder_3,'programqpc.bin')
            assert os.path.isfile(text_encoder_3_qpc), f"Could not find binary {text_encoder_3_qpc = }!"
            pipe = StableDiffusion3Pipeline.from_pretrained(
                                                model_id,
                                                device_id=device_id,
                                                device_id2=device_id_2,
                                                transformer_qpc=transformer_qpc,
                                                vae_decoder_qpc=vae_decoder_sdxl_qpc,
                                                text_encoder_qpc=text_encoder_qpc,
                                                text_encoder_2_qpc=text_encoder_2_qpc,
                                                text_encoder_3_qpc=text_encoder_3_qpc,
            )
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                                                model_id,
                                                device_id=device_id,
                                                device_id2=device_id_2,
                                                transformer_qpc=transformer_qpc,
                                                vae_decoder_qpc=vae_decoder_sdxl_qpc,
                                                text_encoder_qpc=text_encoder_qpc,
                                                text_encoder_2_qpc=text_encoder_2_qpc,
                                                text_encoder_3=None,
                                                tokenizer_3=None,
            )

        self.pipe = pipe

    def generate(self, prompt, n=1, image_size=(1024,1024), num_steps=28, guidance=4.5):
        height, width = image_size[0], image_size[1]

        images = self.pipe(prompt=prompt,
                      negative_prompt='',
                      num_inference_steps=num_steps,
                      height=height,
                      width=width,
                      latents=self.latents,
                      vae_type=self.vae_type,
                      guidance_scale=guidance).images

        return images

def main():
    model = QAICStableDiffusion3()
    prompt = 'photo of 8k ultra realistic harbour, port, boats, sunset, beautiful light, golden hour, full of colour, cinematic lighting, battered, trending on artstation, 4k, hyperrealistic, focused, extreme details, cinematic, masterpiece'
    image = model.generate(prompt, guidance=7.0)[0]
    image.save('harbor.png')

if __name__ == "__main__":
    main()

