from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
import torch
from diffusers import UNet2DConditionModel, DPMSolverMultistepScheduler
from transformers import CLIPTextModel

import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class Pipeline:

    def __init__(self, save_path, classes_to_generate=List[str]):
        self.model_id = "stabilityai/stable-diffusion-2-1-base"
        self.save_path = save_path
        self.classes_to_generate = classes_to_generate

        self.check_or_create_save_path()
        self.current_img_id = self.get_current_img_id()
        print(self.current_img_id)

        self.pipe = self.get_hf_pipeline()
    
    def check_or_create_save_path(self):

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for class_prompt in self.classes_to_generate:
            class_path = os.path.join(self.save_path, class_prompt)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

    def get_current_img_id(self):
        current_img_id = {}
        for class_prompt in self.classes_to_generate:
            class_path = os.path.join(self.save_path, class_prompt)
            if not os.path.exists(class_path):
                current_img_id[class_prompt] = 0
            else:
                img_ids = [int(img.split(".")[0]) for img in os.listdir(class_path)]
                current_img_id[class_prompt] = max(img_ids, default=0)
        return current_img_id

    def get_hf_pipeline(self):

        pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16, safety_checker=None)
        pipe.unet = UNet2DConditionModel.from_pretrained("flix-k/tsa_v1.3", subfolder="50/unet", torch_dtype=torch.float16)
        pipe.text_encoder = CLIPTextModel.from_pretrained("flix-k/tsa_v1.3", subfolder="50/text_encoder", torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        pipe = pipe.to("cuda")
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.set_progress_bar_config(disable=True)
        return pipe
    
    def generate_batch_of_images(self, prompt: Union[str, List[str]] = None,
                                 height: Optional[int] = None,
                                 width: Optional[int] = None,
                                 num_inference_steps: int = 25,
                                 guidance_scale: float = 5,
                                 num_images_per_prompt: Optional[int] = 1,
                                 eta: float = 0.0,
                                 latents: Optional[torch.FloatTensor] = None,
                                 prompt_embeds: Optional[torch.FloatTensor] = None,
                                 output_type: Optional[str] = "pil",
                                 return_dict: bool = True,
                                 refine: bool = False):

        images = self.pipe(prompt=prompt,
                           height=height,
                           width=width,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=guidance_scale,
                           num_images_per_prompt=num_images_per_prompt,
                           eta=eta,
                           latents=latents,
                           prompt_embeds=prompt_embeds,
                           output_type=output_type,
                           return_dict=return_dict,).images
        
        if self.save_path:
            id_ = self.current_img_id[prompt] + 1
            # save every image in images
            for i, image in enumerate(images):
                image.save(f"{self.save_path}/{prompt}/{id_+i}.jpg")

            self.current_img_id[prompt] += len(images)

    def generate_batch_for_each_class(self, num_images_per_prompt=1):
        for class_prompt in self.classes_to_generate:
            self.generate_batch_of_images(prompt=class_prompt, num_images_per_prompt=num_images_per_prompt)
        
    def run(self, num_runs_per_class=10, num_images_per_prompt=1):
        for _ in tqdm(range(num_runs_per_class)):
            self.generate_batch_for_each_class(num_images_per_prompt=num_images_per_prompt)