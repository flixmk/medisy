from network import CustomResNet18
import pytorch_lightning as pl
from torch import nn, optim
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.utils.data import ConcatDataset
from dataset import BalancedImageFolder
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
import numpy as np
import wandb

class LitResNet18(pl.LightningModule):
    def __init__(self, 
                 num_classes=1000, 
                 learning_rate=1e-3,
                 train_dataset=None,
                 val_dataset=None,
                 batch_size=16):
        super(LitResNet18, self).__init__()

        # Store the datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Store the batch size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize our previously defined CustomResNet18 model
        self.model = CustomResNet18(num_classes=num_classes)
        
        # Store hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        # torch.cuda.empty_cache()
        return loss
    
    def on_validation_epoch_start(self):
        self.val_losses = []  # Reset the losses list every epoch
        self.val_labels = []  # To store actual labels
        self.val_preds = []  # To store predictions

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.val_losses.append(loss)
        
        # For accuracy calculation
        preds = torch.argmax(y_hat, dim=1)
        self.val_labels.extend(y.cpu().numpy())
        self.val_preds.extend(preds.cpu().numpy())

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()  # Compute the average loss
        self.log('val_loss', avg_loss, prog_bar=True)  # Log the average validation loss
        
        # Calculate accuracy
        val_labels = np.array(self.val_labels)
        val_preds = np.array(self.val_preds)
        accuracy = np.sum(val_labels == val_preds) / len(val_labels)
        self.log('val_accuracy', torch.tensor(accuracy), prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        lambda_func = lambda epoch: 1 - epoch / self.trainer.max_epochs
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'epoch',  # or 'step'
            'frequency': 1,
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=20)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=20)



def train(n_images_per_class_1, n_images_per_class_2):
    # Example usage:
    

    # Define your transform
    transform_original = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.19196127, 0.19196127, 0.19196127],
                            std=[0.19903535, 0.19903535, 0.19903535])
    ])

    transform_synth = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.16797477, 0.16927803, 0.16864358],
                            std=[0.16557804, 0.1659938, 0.16580833])
    ])


    # Initialize your individual custom datasets
    root_dir1 = '/home/flix/Documents/Datasets/OCT_Dataset_Masterthesis/Splits/good_split_8k/train'
    root_dir2 = '/home/flix/Downloads/slow_dataset_v2 (2)/slow_dataset_v2'
    n_images_per_class_1 = n_images_per_class_1  # Set this to the number of images per class you want
    n_images_per_class_2 = n_images_per_class_2 # Set this to the number of images per class you want

    dataset1 = BalancedImageFolder(root_dir1, transform=transform_original, n_images_per_class=n_images_per_class_1)
    dataset2 = BalancedImageFolder(root_dir2, transform=transform_synth, n_images_per_class=n_images_per_class_2)

    # Concatenate them into a single dataset
    train_dataset = ConcatDataset([dataset1, dataset2])

    print(f"Total number of images in the training dataset: {len(train_dataset)}")

    # Initialize a validation dataset
    val_dataset = BalancedImageFolder('/home/flix/Documents/Datasets/OCT_Dataset_Masterthesis/Splits/good_split_8k/test', transform=transform_original, n_images_per_class=10000)

    # Example usage:
    num_classes = 4
    learning_rate = 1e-4
    lit_model = LitResNet18(
        num_classes=num_classes, 
        learning_rate=learning_rate,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=64)
    
    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=15, 
        accelerator='gpu',
        precision='bf16',
        logger=pl.loggers.WandbLogger(project='OCT_Classification', name=f"Original: {n_images_per_class_1}, Synthetic: {n_images_per_class_2}"),
        )

    # Train the model ⚡
    trainer.fit(lit_model)

    wandb.finish()
    
if __name__ == '__main__':
    train(n_images_per_class_1=1000, n_images_per_class_2=0)
    train(n_images_per_class_1=0, n_images_per_class_2=1000)
    train(n_images_per_class_1=0, n_images_per_class_2=5000)
    train(n_images_per_class_1=0, n_images_per_class_2=10000)

