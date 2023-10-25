import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler, DDPMScheduler
import random
from diffusers.utils.import_utils import is_xformers_available

class NoVAEDiffusionModel(nn.Module):

    def __init__(self, class_prompts, strategy="default"):
        super(NoVAEDiffusionModel, self).__init__()
        # Setting modules besides unet and text encoder
        self.noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")

        # setting unet and text encoder; freezing them based on strategy
        self.unet, self.text_encoder = self.learning_strategy(strategy)
        
        self.prepare_unet()

        self.unet.enable_gradient_checkpointing()
        self.text_encoder.gradient_checkpointing_enable()


        # other parameters
        self.noise_offset = 0.1
        self.class_prompts = class_prompts

    def forward(self, batch):
        with torch.no_grad():
            latents = batch["latents"] * 0.18215
        noise = torch.randn_like(latents)
        if self.noise_offset:
            noise = torch.randn_like(latents) +  self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # if self.strategy == "gpp":
        #     inputs_strings = [self.general_pretraining_phase(self.class_prompts[i]) for i in batch["classes"]]
        # else:
        #     inputs_strings = [self.multi_modal_strings(self.class_prompts[i]) for i in batch["classes"]]

        inputs_strings = [self.get_string_from_strategy(self.class_prompts[i]) for i in batch["classes"]]
        inputs = self.tokenizer(
            inputs_strings, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        encoder_hidden_states = self.text_encoder(inputs)[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return model_pred, target, timesteps, bsz

    def multi_modal_strings(self, string):
        file_path = "./multi_modal_string_files/general_pretraining_phase.txt"
        predefined_strings = []

        # Read the strings from the file
        with open(file_path, 'r') as f:
            for line in f:
                # Remove the newline character at the end of each line
                predefined_strings.append(line.strip())
                
        # Choose a random string template from the predefined set
        chosen_template = random.choice(predefined_strings)

        # Use the format method to substitute {string} with the desired value
        chosen_string = chosen_template.format(string=string)
        
        return chosen_string

    def general_pretraining_phase(self):
        file_path = "./multi_modal_string_files/class_specific_phase.txt"
        
        predefined_strings = []

        # Read the strings from the file
        with open(file_path, 'r') as f:
            for line in f:
                # Remove the newline character at the end of each line
                predefined_strings.append(line.strip())
                
        # Choose a random string from the predefined set
        chosen_string = random.choice(predefined_strings)
        return chosen_string


    def learning_strategy(self, strat, gpp_model = None):

        if strat == "default":
            self.strategy = strat
            unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
            text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")
            return unet, text_encoder

        elif strat == "gpp":
            self.strategy = strat
            unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
            text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

            # Freeze the text encoder
            text_encoder.requires_grad_(False)
            return unet, text_encoder
        
        elif strat == "text_encoder_full":
            self.strategy = strat
            unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base")
            text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")


            # Freeze the UNet
            unet.requires_grad_(False)
            return unet, text_encoder
        
        elif strat == "textual_inversion":
            self.strategy = strat
            unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet")
            text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")


            # Freeze the UNet
            unet.requires_grad_(False)

            # Freeze all parameters except for the token embeddings in text encoder
            text_encoder.text_model.encoder.requires_grad_(False)
            text_encoder.text_model.final_layer_norm.requires_grad_(False)
            text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
            return unet, text_encoder
        else:
            raise ValueError("Invalid learning strategy")
        
        
    def prepare_unet(self):
        # if is_xformers_available():
            # self.unet.enable_xformers_memory_efficient_attention()
        
        self.unet.half()

        for layer in self.unet.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
            if isinstance(layer, nn.GroupNorm):
                layer.float()
            if isinstance(layer, nn.LayerNorm):
                layer.float()

    def save_components(self, path, epoch):

        self.unet.save_pretrained(f"./{path}/{epoch}/unet/")
        self.text_encoder.save_pretrained(f"./{path}/{epoch}/text_encoder/")

    def get_pipeline(self):
        # define pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            safety_checker=None,
            )
        pipeline.set_progress_bar_config(disable=True)
        scheduler = DPMSolverMultistepScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
        pipeline.scheduler = scheduler
        # if is_xformers_available():
        #     pipeline.enable_xformers_memory_efficient_attention()
        return pipeline
    
    def get_string_from_strategy(self, string):
        if self.strategy == "default":
            return self.multi_modal_strings(string)
        elif self.strategy == "gpp":
            return self.general_pretraining_phase(string)
        elif self.strategy == "text_encoder_full":
            return self.multi_modal_strings(string)
        elif self.strategy == "textual_inversion":
            return self.multi_modal_strings(string)
        else:
            raise ValueError("Invalid learning strategy")

        
        
