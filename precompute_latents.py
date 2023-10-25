import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from transformers import CLIPTextModel, CLIPTokenizer
import torch

from tqdm import tqdm 
from diffusers import AutoencoderKL
import pickle
import os

PATH_TO_TRAINDATA = "..."
PATH_TO_VALDATA = "..."
PATH_TO_SAVED_DATA = "..."
SIZE = 498

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, samples_per_class=None):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.samples_per_class = samples_per_class
        
        if samples_per_class is not None:
            # New class_to_idx dictionary
            new_class_to_idx = {}
            # New samples list
            new_samples = []
            # New targets list
            new_targets = []
            
            # For each class in the original class_to_idx
            for class_name in self.class_to_idx:
                # Get all the samples for this class
                class_samples = [(s, t) for s, t in self.samples if t == self.class_to_idx[class_name]]
                # If there are more samples than samples_per_class, trim the list
                if len(class_samples) > samples_per_class:
                    class_samples = class_samples[:samples_per_class]
                
                # Append the samples to the new samples and targets list
                new_samples.extend(class_samples)
                new_targets.extend([self.class_to_idx[class_name]] * len(class_samples))
                # Set the class_to_idx for the new class
                new_class_to_idx[class_name] = self.class_to_idx[class_name]
            
            # Set the new class_to_idx, samples, and targets
            self.class_to_idx = new_class_to_idx
            self.samples = new_samples
            self.targets = new_targets
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: {'images': sample, 'ids': target}
        """
        img, target = super(CustomImageFolder, self).__getitem__(index)
        target = self.classes[target] # Get class name
        
        return {'images': img, 'targets': target}

    
def collate_fn(examples):
    input_ids = [example["targets"] for example in examples]
    pixel_values = [example["images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "targets": input_ids,
        "pixel_values": pixel_values,
    }
    return batch

def precompute_latents_pickle(vae, train_dataloader, val_dataloader, classes):


    counter = 0
    # create folder structure
    os.makedirs(PATH_TO_SAVED_DATA, exist_ok=True)
    os.makedirs(f"{PATH_TO_SAVED_DATA}/train", exist_ok=True)
    os.makedirs(f"{PATH_TO_SAVED_DATA}/val", exist_ok=True)

    # create folder for each class in train and val
    for c in classes:
        os.makedirs(f"{PATH_TO_SAVED_DATA}/train/{c}", exist_ok=True)
        os.makedirs(f"{PATH_TO_SAVED_DATA}/val/{c}", exist_ok=True)


    for batch in tqdm(train_dataloader):
        pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)
        with torch.no_grad():
            latent_dist = vae.encode(pixel_values).latent_dist

        
        target = batch["targets"][0]
            # save latent distribution
        with open(f'{PATH_TO_SAVED_DATA}/train/{target}/{target}-({counter}).pkl', 'wb') as output:
            pickle.dump(latent_dist, output, pickle.DEFAULT_PROTOCOL)
        counter += 1

    counter = 0
    for batch in tqdm(val_dataloader):
        pixel_values = batch["pixel_values"].to("cuda", dtype=torch.float16)
        with torch.no_grad():
            latent_dist = vae.encode(pixel_values).latent_dist

        
        target = batch["targets"][0]
            # save latent distribution
        with open(f'{PATH_TO_SAVED_DATA}/val/{target}/{target}-({counter}).pkl', 'wb') as output:
            pickle.dump(latent_dist, output, pickle.DEFAULT_PROTOCOL) # use second to highest protocol

        counter += 1

if __name__ == "__main__":


    size = SIZE
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # dataset = CustomImageFolder(root="/home/flix/Documents/oct-data/CellData/OCT/test/", transform=transform)

    train_dataset = CustomImageFolder(root=PATH_TO_TRAINDATA, transform=transform)
    val_dataset = CustomImageFolder(root=PATH_TO_VALDATA, transform=transform)

    # Now you can create DataLoaders for your training and validation datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)



    device = "cuda"
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").to(device, dtype=torch.float16)
    vae.eval()

    precompute_latents_pickle(vae, train_loader, val_loader, train_dataset.classes)

