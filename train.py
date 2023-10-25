
from lightning.pytorch import callbacks, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from TwoStageDiffusionTrainingModule import LatentDiffusionModel
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb

class MyLightningCLI_TI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        self.model_name = "TI"
        parser.set_defaults({
            "model.strategy_diff_model": "textual_inversion",
            "model.model_name": self.model_name,
            "model.text_encoder_lr": 1e-3,
            "model.samples_per_class": 500,
            "model.total_samples": None,
            "model.snr_gamma": None,
            "model.use_linear_lr_scheduler": True,
            "model.batch_size": 1,
            "model.train_path": "/home/flix/Desktop/Masterthesis/medisy/medisy/diffusion_modules/latents/train",
            "model.val_path": "/home/flix/Desktop/Masterthesis/medisy/medisy/diffusion_modules/latents/val",
        })
    def instantiate_trainer(self, **kwargs):
        trainer = super().instantiate_trainer()
        # trainer.logger = WandbLogger(name=self.model_name) # if you want to use wandb, you need to set your project name here
        trainer.devices = 1
        return trainer
    
class MyLightningCLI_GPP(LightningCLI):
    def add_arguments_to_parser(self, parser):
        self.model_name = "GPP"
        parser.set_defaults({
            "model.strategy_diff_model": "gpp",
            "model.model_name": self.model_name,
        })
    def instantiate_trainer(self, **kwargs):
        trainer = super().instantiate_trainer()
        trainer.logger = WandbLogger(name=self.model_name) # if you want to use wandb, you need to set your project name here
        trainer.devices = 1
        return trainer
    
def run_training(cli_class):
    cli = cli_class(
        LatentDiffusionModel,
        seed_everything_default=1234,
        run=False,  # used to de-activate automatic fitting.
        trainer_defaults={"max_epochs": 1000, 
                          "max_steps": 300001,
                          "val_check_interval": 1000, 
                          "check_val_every_n_epoch":None,
                          "accelerator": 'gpu',
                          "precision": 'bf16-mixed',},
        save_config_kwargs={"overwrite": True},
    )
    print(ModelSummary(cli.model, max_depth=2))
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    wandb.finish()
        

def cli_main():

    experiments = [MyLightningCLI_GPP, MyLightningCLI_TI]

    for run in experiments:
        run_training(run)


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
