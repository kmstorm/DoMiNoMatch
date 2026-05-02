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

def get_domainnet_ada(args, alg, name, train_on_sources_domain=False, domain_list=None, num_labels=None, num_classes=None, data_dir='./data', include_lb_to_ulb=True):
    
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
    
    print(f"Labeled data count: {len(lb_data)}")
    
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

    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(len(ulb_data)))
    print("test count: {}".format(len(test_data)))
    print("source domain count: {}".format(len(domain_data)))

    lb_dset = SSDADataset(alg, lb_data, train_on_sources_domain, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)
    ulb_dset = SSDADataset(alg, ulb_data, train_on_sources_domain, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    eval_dset = SSDADataset(alg, test_data, train_on_sources_domain, test_targets, num_classes, transform_val, False, None, None, False)

    return lb_dset, ulb_dset, eval_dset
    

class SSDADataset(BasicDataset):
    def __init__(self, alg, data, is_source_domain, targets=None, num_classes=None, transform=None, is_ulb=False, medium_transform=None, strong_transform=None, onehot=False, *args, **kwargs):
        super().__init__(alg, data, targets, num_classes, transform, is_ulb, medium_transform, strong_transform, onehot, *args, **kwargs)
        self.is_source_domain = is_source_domain

    def __sample__(self, idx):
        path =  self.data[idx]
        img = Image.open(path).convert('RGB')
        try:
            target = self.targets[idx]
        except:
            target = None
        return img, target
    
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target = self.__sample__(idx)

        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                if self.alg in ['adamatch', 'mme', 'st', 'ape', 'ent']:
                    if self.strong_transform:
                        if self.is_source_domain:
                            return {'idx_src_lb': idx, 'x_lb_src_w': img_w, 'x_lb_src_s': self.strong_transform(img), 'y_src': target}
                        else:
                            return {'idx_tgt_lb': idx, 'x_lb_tgt_w': img_w, 'x_lb_tgt_s': self.strong_transform(img), 'y_tgt': target}
                return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target} 
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': idx, 'x_ulb_w':img_w} 
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                # elif self.alg == 'sequencematch' or self.alg == 'somematch':
                elif self.alg == 'sequencematch' or self.alg == 'freesequencematch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_m': self.medium_transform(img), 'x_ulb_s': self.strong_transform(img)} 
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)} 
                elif self.alg == 'envmatch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w}                    
                else:
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)} 
