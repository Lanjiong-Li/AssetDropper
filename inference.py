# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from ip_adapter.ip_adapter import Resampler

import argparse
import logging
import os
import torch.utils.data as data
import torchvision
import json
import accelerate
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
import random
from diffusers.utils.import_utils import is_xformers_available

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.assetdropper_pipeline import StableDiffusionXLInpaintPipeline as AssetDropperPipeline

from dataloader import AssetDataset

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="paras for inference.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="",required=False,)
    parser.add_argument("--width",type=int,default=512,)
    parser.add_argument("--height",type=int,default=512,)
    parser.add_argument("--Pwidth",type=int,default=512,)
    parser.add_argument("--Pheight",type=int,default=512,)
    parser.add_argument("--txt_name",type=str,default=None)
    parser.add_argument("--num_inference_steps",type=int,default=50,)
    parser.add_argument("--output_dir",type=str,default="./output",)
    parser.add_argument("--data_dir",type=str,default="./dataset")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--test_batch_size", type=int, default=2,)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    args = parser.parse_args()

    return args

def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    unet = UNet2DConditionModel.from_pretrained(
        f"{args.pretrained_model_name_or_path}/unet",
        low_cpu_mem_usage=True, 
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        subfolder="unet"
    )
    unet_encoder.config.addition_embed_type = None
    unet_encoder.config["addition_embed_type"] = None

    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet_encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet_encoder.to(accelerator.device, weight_dtype)
    unet.eval()
    unet_encoder.eval()

    conv_new_encoder = torch.nn.Conv2d(
        in_channels=6,
        out_channels=unet_encoder.conv_in.out_channels,
        kernel_size=3,
        padding=1,
    )
    torch.nn.init.kaiming_normal_(conv_new_encoder.weight)  
    conv_new_encoder.weight.data = conv_new_encoder.weight.data * 0.  
    conv_new_encoder.weight.data[:, :4] = unet_encoder.conv_in.weight.data[:, :4]
    conv_new_encoder.bias.data = unet_encoder.conv_in.bias.data  
    unet_encoder.conv_in = conv_new_encoder  # replace conv layer in unet
    unet_encoder.config['in_channels'] = 6  # update config
    unet_encoder.config.in_channels = 6  # update config
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    test_dataset = AssetDataset(
        dataroot_path=args.data_dir,
        phase="test",
        size=(args.height, args.width),
        txt_name=args.txt_name,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )
    
    newpipe = AssetDropperPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = text_encoder_one,
            text_encoder_2 = text_encoder_two,
            tokenizer = tokenizer_one,
            tokenizer_2 = tokenizer_two,
            scheduler = noise_scheduler,
            image_encoder=image_encoder,
            unet_encoder = unet_encoder,
            torch_dtype=torch.float16,
            add_watermarker=False,
            safety_checker=None,
    ).to(accelerator.device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for sample in test_dataloader:
                        
                        masked_image_emb_list = []
                        
                        for i in range(sample['masked_image'].shape[0]):
                            masked_image_emb_list.append(sample['masked_image'][i])

                        masked_image_embeds = torch.cat(masked_image_emb_list, dim=0)

                        prompt = sample["caption_pattern"]
                        num_prompts = sample['image'].shape[0]                                        
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                        if not isinstance(prompt, List):
                            prompt = [prompt] * num_prompts
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * num_prompts

                        with torch.inference_mode():
                            (
                                prompt_embeds,
                                negative_prompt_embeds,
                                pooled_prompt_embeds,
                                negative_pooled_prompt_embeds,
                            ) = newpipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=True,
                                negative_prompt=negative_prompt,
                            )
                            
                        prompt = sample["caption_gen"]
                        num_prompts = sample['image'].shape[0]
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                        if not isinstance(prompt, List):
                            prompt = [prompt] * num_prompts
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * num_prompts


                        with torch.inference_mode():
                            (
                                prompt_embeds_c,
                                _,
                                _,
                                _,
                            ) = newpipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )
                        
                        seed = args.seed
                        generator = torch.Generator(newpipe.device).manual_seed(seed)

                        images = newpipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                            num_inference_steps=args.num_inference_steps,
                            generator=generator,
                            strength = 1.0,
                            reference_image_embed=prompt_embeds_c, #reference_image_embed
                            image = sample["image"].to(accelerator.device),
                            mask = sample['mask'],
                            edgemap = sample['edgemap'],
                            pattern = sample['pattern'],
                            height=args.height,
                            width=args.width,
                            P_height=args.Pheight,
                            P_width=args.Pwidth,
                            guidance_scale=args.guidance_scale,
                            ip_adapter_image = masked_image_embeds,
                        )[0]

                        for i in range(len(images)):
                            x_sample = pil_to_tensor(images[i])
                            save_path = os.path.join(args.output_dir, f"{sample['image_name'][i]}")
                            torchvision.utils.save_image(x_sample, save_path)

        torch.cuda.empty_cache()
                



if __name__ == "__main__":
    main()
