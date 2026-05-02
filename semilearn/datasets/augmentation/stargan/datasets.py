import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class EnvTranslationDataset(Dataset):
    def __init__(self, root_dir, domains, transform=None):
        """
        A PyTorch dataset for multi-domain image translation tasks.
        Suitable for use in StarGAN-like models where each image belongs to a domain.

        Args:
            root_dir (str): Path to the data split directory (e.g., 'data/train' or 'data/val').
            domains (List[str]): List of domain names to include (e.g., ['hazy', 'lol', 'normal']).
            transform (callable, optional): Transformations to apply to each image.
        """
        self.root_dir = root_dir
        self.domains = domains
        self.transform = transform

        self.image_paths = []  # Stores tuples of (image_path, domain_label)
        self.domain_to_label = {domain: idx for idx, domain in enumerate(domains)}

        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            for fname in os.listdir(domain_dir):
                if fname.lower().endswith(('.png', '.jpg')):
                    self.image_paths.append(
                        (os.path.join(domain_dir, fname), self.domain_to_label[domain])
                    )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path, label = self.image_paths[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


