import os
# import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
import torch
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, StarGANAugment
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
import numpy as np
from torchvision.transforms.functional import to_tensor
from pathlib import Path


def get_multidomain_preaugment(args, alg, name, domain_list, num_labels, num_classes, data_dir = './data', include_lb_to_ulb=True):
    data_dir = os.path.join(data_dir, name.lower())

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

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

        
    lb_data, lb_targets, lb_domains = [], [], []
    ulb_data, ulb_domains = [], []

    max_per_class = num_labels // num_classes 

    for domain in domain_list:
        for label in os.listdir(os.path.join(data_dir, domain, 'train')):
            path = os.path.join(data_dir, domain, 'train', label)
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
            targets = [int(label)] * len(os.listdir(path))  
            domains = [domain] * len(os.listdir(path))    
            
            if len(imagepaths) > max_per_class:
                imagepaths = imagepaths[:max_per_class]
                targets = targets[:max_per_class]
                domains = targets[:max_per_class]
        
                if include_lb_to_ulb:
                    ulb_imagepaths_per_class = imagepaths[max_per_class:]
                    ulb_domains_per_class = domains[max_per_class:]
                    
                    ulb_data.extend(ulb_imagepaths_per_class)  # Add remaining to unlabeled
                    ulb_domains.extend(ulb_domains_per_class)       
        
            lb_data.extend(imagepaths)
            lb_targets.extend(targets)
            lb_domains.extend(domains)



    
    test_data, test_targets, test_domains = [], [], []
    for domain in domain_list:
        for label in os.listdir(os.path.join(data_dir, domain, 'test')):
            path = os.path.join(data_dir, domain, 'test', label)
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
            targets = [int(label)] * len(os.listdir(path))
            domains = [domain] * len(os.listdir(path))     
            test_data.extend(imagepaths)
            test_targets.extend(targets)
            test_domains.extend(domains)
    
    if alg != 'fullysupervised':
        for domain in domain_list:
            path = os.path.join(data_dir, domain, 'unlabel')
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]  
            ulb_data.extend(imagepaths)
            ulb_domains.extend( [domain] * len(imagenames) )

              
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    print("lb count: {}".format(lb_count))
              
    lb_dset = MultiDomainPADataset(alg, lb_data, lb_domains, domain_list, targets=lb_targets, num_classes=num_classes, transform=transform_weak, is_ulb=False, medium_transform=transform_strong, strong_transform=transform_strong, onehot=False)
    ulb_dset = MultiDomainPADataset(alg, ulb_data, ulb_domains, domain_list, num_classes=num_classes, transform=transform_weak, is_ulb=True, medium_transform=transform_medium, strong_transform=transform_strong, onehot=False)
    eval_dset = MultiDomainPADataset(alg, test_data, test_domains, domain_list, targets=test_targets, num_classes=num_classes, transform=transform_val, is_ulb=False, medium_transform=None, strong_transform=None, onehot=False)
    return lb_dset, ulb_dset, eval_dset

class MultiDomainPADataset(BasicDataset):

    def __init__(self, alg, data, domains, domain_list, targets=None, num_classes=None, transform=None, is_ulb=False, medium_transform=None, strong_transform=None, onehot=False, sroot="style", *args, **kwargs ):
        super().__init__(alg, data, targets, num_classes, transform, is_ulb, medium_transform, strong_transform, onehot, *args, **kwargs)
        self.preaugment_path = 'preaugmentcyclegan'
        self.domains = domains
        self.domain_list = domain_list

        assert len(self.domains) == len(self.data), "The domains were not handled correctly"



    def __sample__(self, idx):
        path =  self.data[idx]
        img = Image.open(path).convert('RGB')
        sample_domain = self.domains[idx]

        image_path = Path(path)
        short_path = Path(*image_path.parts[-3:])
        base_name = str(short_path)
        imgs_aug = {}
        for domain in self.domain_list:
            if domain != sample_domain:
                preaugment_path = os.path.join(self.preaugment_path, base_name[:-4] + f"_{domain}.png")
                try:
                    imgs_aug[domain] = Image.open(preaugment_path).convert('RGB')
                except:
                    imgs_aug[domain] = img

        try:
            target = self.targets[idx]
        except:
            target = None
        return img, target, imgs_aug

    def __getitem__(self, idx):

        img, target, imgs_aug = self.__sample__(idx)
        img_w = self.transform(img)


        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            aug_list = []
            
            for domain, img_aug in imgs_aug.items():
                if isinstance(img_aug, np.ndarray):
                    img_aug = Image.fromarray(img_aug)
                imgs_aug[domain] = self.transform(img_aug)
                aug_list.append(imgs_aug[domain])

            if not self.is_ulb:
                return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target} 
            else: 
                return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_aug': aug_list}