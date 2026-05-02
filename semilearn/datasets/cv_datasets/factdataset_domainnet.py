import os
import copy
import json
import random
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, FourierMixAugment
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
import numpy as np
import torchvision

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def get_fact_domainnet(args, alg, name, domain_list=None, num_labels=None, num_classes=None, data_dir='./data', include_lb_to_ulb=True):
    
    if domain_list is None:
        domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    source_domain = args.source_domain
    jitter = getattr(args, 'jitter', 0) 
    
    domain_dir = os.path.join(data_dir, source_domain.lower())
    data_dir = os.path.join(data_dir, name.lower())
        
    # Determine how many samples per class for labeled set
    max_per_class = num_labels // num_classes

    # Initialize data containers
    lb_data, lb_targets = [], []
    ulb_data, ulb_targets = [], []
    test_data, test_targets = [], []
    
    # Get list of domains for training (exclude source domain)
    train_domain_list = []
    try:
        train_domain_list = [args.target_domain]
        print(f"Training on single domain {format(args.target_domain)}")
    except:
        train_domain_list = [domain for domain in domain_list if domain != source_domain]
        print(f"Training on all target domains except source: {train_domain_list}")

    class_list_126_path = 'data/domainnet/domainnet_126.txt'
    class_mapping = {}
    with open(class_list_126_path, 'r') as f:
        for line in f:
            if ':' in line:
                class_name, idx = line.strip().split(':')
                class_mapping[class_name.strip()] = int(idx)
        
    print(f"Training on {num_classes} classes")
    train_on_subset = False
    if num_classes == 126:
        train_on_subset = True

    # Load training data from txt files for all target domains
    all_train_data = []
    all_train_labels = []
    
    for domain in train_domain_list:
        train_file = os.path.join(data_dir, f'{domain}_train.txt')
        if not os.path.exists(train_file):
            print(f"Warning: {train_file} does not exist. Skipping domain {domain}.")
            continue
            
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path, label = parts[0], int(parts[1])
                    class_name = img_path.split('/')[1]
                    if (not train_on_subset) or class_name in class_mapping:  
                        label = class_mapping[class_name] if train_on_subset else label
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(data_dir, img_path)
                        all_train_data.append(img_path)
                        all_train_labels.append(label)
        
    # Organize data for each class
    data_by_class = [[] for _ in range(num_classes)]
    for img_path, label in zip(all_train_data, all_train_labels):
        data_by_class[label].append((img_path, label))
        
    # Split into labeled and unlabeled sets
    for label in range(num_classes):
        if len(data_by_class[label]) > max_per_class:
            # Take first max_per_class for labeled data
            for i in range(max_per_class):
                img_path, _ = data_by_class[label][i]
                lb_data.append(img_path)
                lb_targets.append(label)
            
            # Add remaining to unlabeled if specified
            if include_lb_to_ulb:
                for i in range(max_per_class, len(data_by_class[label])):
                    img_path, _ = data_by_class[label][i]
                    ulb_data.append(img_path)
                    ulb_targets.append(label)
        else:
            # If we don't have enough samples, use all for labeled
            for img_path, _ in data_by_class[label]:
                lb_data.append(img_path)
                lb_targets.append(label)
                
    # Load test data from txt files for all target domains
    for domain in train_domain_list:
        test_file = os.path.join(data_dir, f'{domain}_test.txt')
        if not os.path.exists(test_file):
            print(f"Warning: {test_file} does not exist. Skipping domain {domain}.")
            continue
            
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path, label = parts[0], int(parts[1])
                    class_name = img_path.split('/')[1]
                    if (not train_on_subset) or class_name in class_mapping:  
                        label = class_mapping[class_name] if train_on_subset else label
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(data_dir, img_path)
                        test_data.append(img_path)
                        test_targets.append(label)
    
    # Load source domain data for domain adaptation
    domain_data, domain_targets = [], []
    source_train_file = os.path.join(data_dir, f'{source_domain}_train.txt')
    
    if os.path.exists(source_train_file):
        with open(source_train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path, label = parts[0], int(parts[1])
                    class_name = img_path.split('/')[1]
                    if (not train_on_subset) or class_name in class_mapping:  
                        label = class_mapping[class_name] if train_on_subset else label
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(data_dir, img_path)
                        domain_data.append(img_path)
                        domain_targets.append(label)
    else:
        print(f"Warning: Source domain file {source_train_file} not found!")

    # Print statistics
    lb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    test_count = [0 for _ in range(num_classes)]
    for c in test_targets:
        test_count[c] += 1

    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(len(ulb_data)))
    print("test count: {}".format(test_count))
    print("source domain count: {}".format(len(domain_data)))

    augmentation = args.augmentation if hasattr(args, 'augmentation') else 'factaugment'
    
    if augmentation == 'factaugment':
        # Initialize FourierMixAugment
        fourier_augment = FourierMixAugment(
            data_path=data_dir,
            source_domain=source_domain
        )

        # Set up transformations
        pretransform = [
            transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
        ]
        if jitter > 0:
            pretransform.insert(-1, transforms.ColorJitter(brightness=jitter,
                                                        contrast=jitter,
                                                        saturation=jitter,
                                                        hue=min(0.5, jitter)))
        posttransform = [
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)
        ]
        
        transform_weak = transforms.Compose(pretransform + posttransform)
        pretransform = transforms.Compose(pretransform)
        posttransform = transforms.Compose(posttransform)

        transform_val = transforms.Compose([
            transforms.Resize(math.floor(int(img_size / crop_ratio))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)
        ])

        # Create datasets
        if alg in ['fullysupervised', 'supervised']:
            lb_dset = MixDataset(alg, augmentation, lb_data, source_domain, domain_data, domain_targets, lb_targets, num_classes, transform_weak, 
                                False, None, None, False, fourier_augment, domain_list, pretransform, posttransform)
        else:
            lb_dset = MixDataset(alg, augmentation, lb_data, source_domain, domain_data, domain_targets, lb_targets, num_classes, transform_weak, 
                                False, transform_weak, transform_weak, False, fourier_augment, domain_list, pretransform, posttransform)

        ulb_dset = MixDataset(alg, augmentation, ulb_data, source_domain, domain_data, domain_targets, None, num_classes, transform_weak, 
                            True, transform_weak, transform_weak, False, fourier_augment, domain_list, pretransform, posttransform)
        eval_dset = MixDataset(alg, augmentation, test_data, source_domain, domain_data, domain_targets, test_targets, num_classes, transform_weak, 
                            False, transform_weak, transform_weak, False, fourier_augment, domain_list, pretransform, posttransform)
    elif augmentation == 'randaugment':
        transform_weak_list = [                                           
            transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)
        ]

        transform_medium_list = [
            transforms.Resize(int(math.floor(img_size / crop_ratio))),                                                                                                                                                                              
            RandomResizedCropAndInterpolation((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(1, 7, exclude_color_aug=True),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)    
        ]

        transform_strong_list = [
            transforms.Resize(int(math.floor(img_size / crop_ratio))),                                                                                                                                                                              
            RandomResizedCropAndInterpolation((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 7, exclude_color_aug=True),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)
        ]

        transform_val_list = [
            transforms.Resize(math.floor(int(img_size / crop_ratio))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(imgnet_mean, imgnet_std)
        ]

        transform_weak = transforms.Compose(transform_weak_list)

        transform_medium = transforms.Compose(transform_medium_list)

        transform_strong = transforms.Compose(transform_strong_list)

        transform_val = transforms.Compose(transform_val_list)

        lb_dset = RandAugmentDataset(alg, lb_data, domain_data, domain_targets, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)
        ulb_dset = RandAugmentDataset(alg, ulb_data, domain_data, domain_targets, None, num_classes, transform_weak, True, transform_medium, transform_strong, False)
        eval_dset = RandAugmentDataset(alg, test_data, domain_data, domain_targets, test_targets, num_classes, transform_val, False, None, None, False)


    return lb_dset, ulb_dset, eval_dset


class MixDataset(BasicDataset):
    def __init__(self, alg, augmentation, data, source_domain, domain_data, domain_targets, targets=None, num_classes=None, transform=None, is_ulb=False, 
                 medium_transform=None, strong_transform=None, onehot=False, fourier_augment=None,
                 domain_list=None, pretransform=None, posttransform=None, *args, **kwargs): 
        super().__init__(alg, data, targets, num_classes, transform, is_ulb, 
                        medium_transform, strong_transform, onehot, *args, **kwargs)
        
        self.mix_augment = fourier_augment
        self.domain_list = domain_list if domain_list else []
        self.source_domain = source_domain
        self.domain_data = domain_data
        self.domain_targets = domain_targets
        self.pretransform = pretransform
        self.posttransform = posttransform
        self.augmentation = augmentation

        
    def _get_domain_from_path(self, path):
        """Extract domain name from file path"""
        for domain in self.domain_list:
            if domain in path:
                return domain
        return None
    
    def _get_other_domains(self, current_domain):
        """Get list of domains excluding current domain for Fourier mixing"""
        if current_domain is None:
            return self.domain_list
        return [d for d in self.domain_list if d != current_domain]

    def __sample__(self, idx):
        path = self.data[idx]
        img = Image.open(path).convert('RGB')
        try:
            target = self.targets[idx]
        except:
            target = None
        return img, target, path
    
    def get_domain_image(self):     
        idx = random.choice(range(len(self.domain_data)))
        path = self.domain_data[idx]
        img = Image.open(path).convert('RGB')
        target = self.domain_targets[idx]
        return img, target

    def __getitem__(self, idx):
        """
        Returns augmented images for semi-supervised learning
        For unlabeled data: (weak_aug_view, strong_aug_view, label)
        where weak_aug_view uses Fourier mixing
        """
        img, target, path = self.__sample__(idx)
        domain_img, domain_target = self.get_domain_image()

        if target is not None:
            return {'x_lb': self.transform(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            img_w = self.pretransform(img)
            dm_img = self.pretransform(domain_img)
            
            img_mix = self.mix_augment(img_w, dm_img)
            
            img_w = self.posttransform(img_w)
            dm_img = self.posttransform(dm_img)
            img_mix = self.posttransform(img_mix)
             
            return {
                'idx_ulb': idx,
                'x_ulb_w': img_w,
                'x_ulb_s': img_mix,
                'x_ulb_domain': dm_img,
                'x_ulb_domain_target': domain_target
            }
            
    def __len__(self):
        return len(self.data)
            
class RandAugmentDataset(BasicDataset):
    def __init__(self, alg, data, domain_data, domain_targets, targets=None, num_classes=None, transform=None, is_ulb=False, 
                 medium_transform=None, strong_transform=None, onehot=False, *args, **kwargs): 
        super().__init__(alg, data, targets, num_classes, transform, is_ulb, 
                        medium_transform, strong_transform, onehot, *args, **kwargs)
        self.domain_data = domain_data
        self.domain_targets = domain_targets

        
    def get_domain_image(self):     
        idx = random.choice(range(len(self.domain_data)))
        path = self.domain_data[idx]
        img = Image.open(path).convert('RGB')
        target = self.domain_targets[idx]
        return img, target
        
    def __sample__(self, idx):
        path = self.data[idx]
        img = Image.open(path).convert('RGB')
        try:
            target = self.targets[idx]
        except:
            target = None
        return img, target, path
    
    def __getitem__(self, idx):
        """
        Returns augmented images for semi-supervised learning
        For unlabeled data: (weak_aug_view, strong_aug_view, label)
        """
        img, target, path = self.__sample__(idx)
        domain_img, domain_target = self.get_domain_image()

        # print(f"Image path: {path}, Target: {target}, Domain target: {domain_target}")

        if target is not None:
            return {'x_lb': self.transform(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            img_w = self.transform(img)
            img_s = self.strong_transform(img)
            dm_img = self.transform(domain_img)
                        
            return {
                'idx_ulb': idx,
                'x_ulb_w': img_w,
                'x_ulb_s': img_s,
                'x_ulb_domain': dm_img,
                'x_ulb_domain_target': domain_target
            }
            
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--crop_ratio', type=float, default=0.875)
    parser.add_argument('--source_domain', type=str, default='real')
    parser.add_argument('--jitter', type=float, default=0.4)
    # parser.add_argument('--target_domain', type=str, default='sketch')
    args = parser.parse_args()

    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    lb_dset, ulb_dset, eval_dset = get_fact_domainnet(args, 'fullysupervised', 'domainnet', domain_list=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'], num_labels=3450, num_classes=345)
    print("Labeled dataset size:", len(lb_dset))
    print("Unlabeled dataset size:", len(ulb_dset))
    print("Evaluation dataset size:", len(eval_dset))
    
    # Create DataLoader for unlabeled dataset
    ulb_loader = DataLoader(ulb_dset, batch_size=4, shuffle=True)
    
    # Get a batch of unlabeled data to visualize
    batch = next(iter(ulb_loader))
    
    def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Denormalize tensor for visualization"""
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        return tensor * std + mean
    
    def tensor_to_pil(tensor):
        """Convert tensor to PIL Image for visualization"""
        tensor = torch.clamp(denormalize(tensor), 0, 1)
        return transforms.ToPILImage()(tensor)
    
    # Visualize augmentation results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Unlabeled Data Augmentation Results\nTop: Weak (Fourier), Bottom: Strong (RandAugment)', fontsize=14)
    
    for i in range(4):
        # Weak augmentation (with Fourier mixing)
        weak_img = tensor_to_pil(batch['x_ulb_w'][i])
        axes[0, i].imshow(weak_img)
        axes[0, i].set_title(f'Weak Aug {i+1}')
        axes[0, i].axis('off')
        
        # Strong augmentation (RandAugment)
        strong_img = tensor_to_pil(batch['x_ulb_s'][i])
        axes[1, i].imshow(strong_img)
        axes[1, i].set_title(f'Strong Aug {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    batch_results_path = os.path.join(results_dir, 'ulb_augmentation_results.png')
    plt.savefig(batch_results_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Batch augmentation results saved to: {batch_results_path}")
    
    # Print batch information
    print("\nBatch information:")
    print(f"Weak augmented images shape: {batch['x_ulb_w'].shape}")
    print(f"Strong augmented images shape: {batch['x_ulb_s'].shape}")
    print(f"Indices: {batch['idx_ulb']}")
    
    # Show original images vs augmented for comparison
    if len(ulb_dset) > 0:
        # Get original image without augmentation
        original_img, _, path = ulb_dset.__sample__(batch['idx_ulb'][0].item())
        
        # Create a simple transform for comparison
        simple_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        original_transformed = tensor_to_pil(simple_transform(original_img))
        weak_transformed = tensor_to_pil(batch['x_ulb_w'][0])
        strong_transformed = tensor_to_pil(batch['x_ulb_s'][0])
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'Comparison for image: {os.path.basename(path)}', fontsize=14)
        
        axes[0].imshow(original_transformed)
        axes[0].set_title('Original (normalized)')
        axes[0].axis('off')
        
        axes[1].imshow(weak_transformed)
        axes[1].set_title('Weak (Fourier)')
        axes[1].axis('off')
        
        axes[2].imshow(strong_transformed)
        axes[2].set_title('Strong (RandAugment)')
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(results_dir, 'single_image_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Single image comparison saved to: {comparison_path}")
    
    # Test Fourier augmentation specifically
    print("\nTesting Fourier augmentation...")
    if ulb_dset.fourier_augment and len(ulb_dset.fourier_augment.domain_images) > 0:
        sample_img, _, sample_path = ulb_dset.__sample__(0)
        domain = ulb_dset._get_domain_from_path(sample_path)
        print(f"Sample image domain: {domain}")
        print(f"Available domains for mixing: {list(ulb_dset.fourier_augment.domain_images.keys())}")
        
        # Test single Fourier augmentation
        fourier_result = ulb_dset.fourier_augment.single_augment(sample_img)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(sample_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(fourier_result)
        axes[1].set_title('Fourier Mixed')
        axes[1].axis('off')
        
        plt.suptitle('Fourier Augmentation Test')
        plt.tight_layout()
        fourier_test_path = os.path.join(results_dir, 'fourier_test.png')
        plt.savefig(fourier_test_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Fourier test results saved to: {fourier_test_path}")
    else:
        print("No Fourier augmentation data available")
    
    print(f"\nAll visualization results saved to '{results_dir}' directory:")
    print(f"- {os.path.join(results_dir, 'ulb_augmentation_results.png')}")
    print(f"- {os.path.join(results_dir, 'single_image_comparison.png')}")
    print(f"- {os.path.join(results_dir, 'fourier_test.png')}")