import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from collections import defaultdict

class BalancedImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, n_images_per_class=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load initial dataset
        self.full_dataset = ImageFolder(root=self.root_dir, transform=self.transform)

        # Create a dict to store the indices for each class
        self.class_indices = defaultdict(list)
        for idx, (_, class_label) in enumerate(self.full_dataset.imgs):
            self.class_indices[class_label].append(idx)

        # Balance the dataset
        self.filtered_indices = []
        if n_images_per_class is not None:
            for class_label, indices in self.class_indices.items():
                self.filtered_indices.extend(indices[:n_images_per_class])
        else:
            self.filtered_indices = list(range(len(self.full_dataset)))

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        original_idx = self.filtered_indices[index]
        img, label = self.full_dataset[original_idx]
        return img, label


