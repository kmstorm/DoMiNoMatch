import os
import torch
import numpy as np
from torchvision import transforms
from .cyclegan.networks import define_G

import argparse

class CycleGANAugment():
    def __init__(self, domain_list, loadpath, args):
        """Initialize CycleGAN augmentation class
        Args:
            domain_list: List of domain names
            loadpath: Path to load CycleGAN weights
            args: Arguments including GPU settings
        """
        self.args = args
        self.domain_list = domain_list
        self.loadpath = loadpath
        
        if torch.cuda.is_available():
            self.gpu_id = [0]
        else:
            self.gpu_id = []
        
        self.device = torch.device(f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize generators for each domain pair
        self.netG = {}
        for source_domain in domain_list:
            self.netG[source_domain] = {}
            for target_domain in domain_list:
                if source_domain != target_domain:
                    gan = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                 False, 'normal', 0.02, self.gpu_id)
                    
                    # Load weights
                    weight_path = os.path.join(loadpath, f'{source_domain}2{target_domain}.pth')
                    gan.load_state_dict(torch.load(weight_path, map_location=self.device))
                    gan.eval()
                    self.netG[source_domain][target_domain] = gan.to(self.device)

    def __call__(self, content, domain):
        """Generate augmented content using CycleGAN
        Args:
            content: Content image to be transformed
            styles: Style images (not used in CycleGAN)
            domain: Source domain name
        Returns:
            Dictionary of augmented content for each target domain
        """
        augmented_content = {}
        norm = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

        for target_domain in self.domain_list:
            with torch.no_grad():
                if target_domain != domain:
                    # Apply GAN transformation
                    transformed = self.netG[domain][target_domain](content)
                    # Add residual connection and normalize
                    augmented = content + transformed
                    augmented = norm(augmented)
                    augmented_content[target_domain] = augmented
                else:
                    augmented_content[target_domain] = None

        return augmented_content

if __name__ == "__main__":

    # Example usage
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    loadpath = 'pretrained_weight/cyclegan/weights/PACS'
    args = argparse.Namespace(gpu=0)

    augmentor = CycleGANAugment(domains, loadpath, args)

    # For a batch of images from 'photo' domain
    content = torch.randn(8, 3, 224, 224).cuda()  # Example content
    styles = None  # Not used in CycleGAN
    augmented = augmentor(content, styles, domain='photo')
    for domain, aug_content in augmented.items():
        if aug_content is not None:
            print(f'Augmented content for {domain}: {aug_content.shape}')
        else:
            print(f'No augmentation for {domain}')