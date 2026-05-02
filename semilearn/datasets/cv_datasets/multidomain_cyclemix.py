import os
# import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, StarGANAugment
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
import numpy as np
import torchvision

def get_multidomain_cyclemix(args, alg, name, domain_list, num_labels, num_classes, data_dir = './data', include_lb_to_ulb=True):

    if name.lower() in ['acs', 'pcs', 'pas', 'pac']:
        data_dir = os.path.join(data_dir, 'pacs')
    else:
        data_dir = os.path.join(data_dir, name.lower())

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    source_domain = args.source_domain

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

        
    lb_data, lb_targets = [], []
    ulb_data, ulb_targets = [], []

    max_per_class = num_labels // num_classes 
    
    # Get list of domains for training (exclude source domain)
    train_domain_list = []
    try:
        train_domain_list = [args.target_domain]
        print(f"Training on single domain {format(args.target_domain)}")
    except:
        train_domain_list = [domain for domain in domain_list if domain != source_domain]
        print(f"Training on all target domains except source: {train_domain_list}")
    
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

    for env_type in domain_list:
        for label in range(num_classes):
            path = os.path.join(data_dir, env_type, 'train', str(label))
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
            targets = [int(label)]*len(os.listdir(path))    
            
            if len(imagepaths) > max_per_class:
                ulb_imagepaths = imagepaths[max_per_class:]
                ulb_targets_per_class = targets[max_per_class:]
                imagepaths = imagepaths[:max_per_class]
                targets = targets[:max_per_class] 
        
                if include_lb_to_ulb:
                    ulb_data.extend(ulb_imagepaths)  # Add remaining to unlabeled
                    ulb_targets.extend(ulb_targets_per_class)       
        
            lb_data.extend(imagepaths)
            lb_targets.extend(targets)

    
    test_data, test_targets = [], []
    for env_type in test_domain_list:
        for label in range(num_classes):
            path = os.path.join(data_dir, env_type, 'test', str(label))
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
            targets = [int(label)]*len(os.listdir(path))    
            test_data.extend(imagepaths)
            test_targets.extend(targets)
    
    if alg != 'fullysupervised':
        for env_type in domain_list:
            path = os.path.join(data_dir, env_type, 'unlabel')
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]  
            env_ulb_data = imagepaths            
            ulb_data.extend(env_ulb_data)

              
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))
              
    lb_dset = MultiDomainCycleMixDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, transform_strong, False)
    ulb_dset = MultiDomainCycleMixDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_medium, transform_strong, False)
    eval_dset = MultiDomainCycleMixDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, None, False)
    return lb_dset, ulb_dset, eval_dset

class MultiDomainCycleMixDataset(BasicDataset):
    def __sample__(self, idx):
        path =  self.data[idx]
        img = Image.open(path).convert('RGB')
        try:
            target = self.targets[idx]
        except:
            target = None
        return img, target