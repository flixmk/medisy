import torch
import torch.nn.functional as F

from lightning.pytorch import LightningModule


import shutil
import wandb

from LatentDiffusionModel import NoVAEDiffusionModel

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from utils import compute_snr, AverageMeter
from LatentDataset import PickleFolder

def collate_fn(examples):
    targets = [example["target"] for example in examples]
    pixel_values = [example["latents"].sample() for example in examples]
    pixel_values = torch.stack(pixel_values).squeeze(1)
    
    batch = {
        "latents": pixel_values,
        "classes": targets,
    }
    
    return batch

class LatentDiffusionModel(LightningModule):
    def __init__(self, 
                model_name, 
                class_prompts=["CNV", "DME", "DRUSEN", "NORMAL"],
                strategy_diff_model="default", 
                text_encoder_lr=1e-4, 
                lr_unet=7.5e-6,
                samples_per_class_train=None,
                samples_per_class_val=None,
                total_samples_train=None,
                snr_gamma=None,
                use_linear_lr_scheduler=False,
                batch_size=8,
                train_path="./latents/train",
                val_path="./latents/val"
                ):
        super().__init__()

        # parameters set by train.py
        self.model_name = model_name
        self.class_prompts = class_prompts
        self.snr_gamma = snr_gamma
        self.samples_per_class_train = samples_per_class_train
        self.samples_per_class_val = samples_per_class_val
        self.total_samples_train = total_samples_train
        self.lr_unet = lr_unet
        self.lr_text_encoder = text_encoder_lr
        self.strategy_diff_model = strategy_diff_model
        self.use_linear_lr_scheduler = use_linear_lr_scheduler
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path

        # check parameters
        if self.samples_per_class_train is not None and self.total_samples_train is not None:
            raise ValueError("Cannot specify both samples_per_class and total_samples")

        self.save_hyperparameters()
        
        self.ldm = NoVAEDiffusionModel(self.class_prompts, strategy=self.strategy_diff_model)
    
        self.average_meter_train = AverageMeter()
        self.average_meter_val = AverageMeter()

    def forward(self, x):
        # Implement the forward function if necessary.
        pass
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, train=True)
    
    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, train=False)

    def _common_step(self, batch, batch_idx, train=True):

        model_pred , target, timesteps, bsz = self.ldm(batch)

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            snr = compute_snr(timesteps, self.ldm.noise_scheduler)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        detached_loss = loss.clone().detach_()    
        
        if train:
            for i, param_group in enumerate(self.optimizers().param_groups):
                lr = param_group['lr']
                if i == 0:
                    self.log('lr_unet', lr, logger=True, prog_bar=True)
                else:
                    self.log('lr_text_encoder', lr, logger=True, prog_bar=True)
            self.average_meter_train.update(detached_loss, bsz)
            if self.trainer.global_step % 10 == 0:
                try:
                    wandb.log({"avg_train_loss": self.average_meter_train.avg.item()}, step=self.trainer.global_step)
                except:
                    pass
                self.log("self_logs_avg_train_loss", self.average_meter_train.avg.item(), prog_bar=True, logger=True)
            return loss
        else:
            self.average_meter_val.update(detached_loss, bsz)

    def configure_optimizers(self):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit

        optimizer = optimizer_cls(
            [
                    {"params": self.ldm.unet.parameters(), "lr": self.lr_unet},
                    {"params": self.ldm.text_encoder.parameters(), "lr": self.lr_text_encoder}
            ],
            lr=self.lr_unet,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )
        if self.use_linear_lr_scheduler:
            max_steps = self.trainer.max_steps

            lr_lambda = lambda step: 1 - step / max_steps

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                'interval': 'step',  # 'step' updates after each training step, 'epoch' updates after each epoch
            }
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def on_training_epoch_start(self) -> None:
        pass

    def on_training_epoch_end(self) -> None:
        pass

    def on_validation_epoch_start(self) -> None:
        pass

    def on_validation_epoch_end(self):
        # NOTE: generate images during training here
        self.ldm.prepare_unet()

        self.log("avg_val_loss", self.average_meter_val.avg.item(), logger=True)

    def train_dataloader(self):
        if self.samples_per_class is not None:
            self.pickle_train = PickleFolder(self.train_path, samples_per_class=self.samples_per_class_train)
        else:
            self.pickle_train = PickleFolder(self.train_path, total_samples=self.total_samples_train)
        return DataLoader(self.pickle_train, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        # NOTE: Here we always use N samples per class because we want to validate the model on all classes equally
        self.pickle_val = PickleFolder(self.val_path, samples_per_class=self.samples_per_class_val)
        return DataLoader(self.pickle_val, batch_size=self.batch_size, collate_fn=collate_fn)

