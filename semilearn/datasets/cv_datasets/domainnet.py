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
# from semilearn.datasets.augmentation  FourierMixAugment, FourierMixTransform
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
import numpy as np
import torchvision

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def get_domainnet(args, alg, name, train_on_sources_domain=False, domain_list=None, num_labels=None, num_classes=None, data_dir='./data', include_lb_to_ulb=True):
    
    if domain_list is None:
        domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    source_domain = args.source_domain
    
    domain_dir = os.path.join(data_dir, source_domain.lower())
    data_dir = os.path.join(data_dir, name.lower())

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
    
    # Get list of domains for training (exclude source domain)
    if train_on_sources_domain:
        train_domain_list = [source_domain]
    test_domain_list = train_domain_list.copy()    
        
    train_on_S_T = False
    try:
        if args.train_on_S_T:
            train_on_S_T = True
    except:
        train_on_S_T = False
    if train_on_S_T:
        train_domain_list = [source_domain] + train_domain_list
        print(f"Training on source and target domains: {train_domain_list}")
    
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
    
    # Determine how many samples per class for labeled set
    max_per_class = num_labels // num_classes
    
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
    for domain in test_domain_list:
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

    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(len(ulb_data)))
    print("test count: {}".format(len(test_data)))
    print("source domain count: {}".format(len(domain_data)))

    lb_dset = FactDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)
    ulb_dset = FactDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    eval_dset = FactDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)

    return lb_dset, ulb_dset, eval_dset
    

class FactDataset(BasicDataset):
    def __sample__(self, idx):
        path =  self.data[idx]
        img = Image.open(path).convert('RGB')
        try:
            target = self.targets[idx]
        except:
            target = None
        return img, target
            

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--crop_ratio', type=float, default=0.875)
    parser.add_argument('--source_domain', type=str, default='real')
    parser.add_argument('--jitter', type=float, default=0.4)
    args = parser.parse_args()

    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    lb_dset, ulb_dset, eval_dset = get_domainnet(args, 'fullysupervised', 'domainnet', train_on_sources_domain=False, domain_list=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'], num_labels=3450000, num_classes=345)
    print("Labeled dataset size:", len(lb_dset))
    print("Unlabeled dataset size:", len(ulb_dset))
    print("Evaluation dataset size:", len(eval_dset))
    
    