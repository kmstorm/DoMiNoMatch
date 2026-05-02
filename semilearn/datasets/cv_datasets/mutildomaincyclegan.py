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

def get_augmentdataset(args, alg, name, domain_list, num_labels, num_classes, data_dir = './data', include_lb_to_ulb=True):
    data_dir = os.path.join(data_dir, name.lower())

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    tsit_img_size = args.tsit_img_size

    sroot = os.path.join(data_dir, 'styles')

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
    
    transforms_style_list = [
        transforms.Resize((tsit_img_size, tsit_img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ]

    transform_weak = transforms.Compose(transform_weak_list)

    transform_medium = transforms.Compose(transform_medium_list)

    transform_strong = transforms.Compose(transform_strong_list)

    transform_val = transforms.Compose(transform_val_list)

    transform_style = transforms.Compose(transforms_style_list)
    
    ulb_data_dict , ulb_domains_dict = {}, {}

    max_per_class = num_labels // num_classes 

    for domain in domain_list:
        ulb_data, ulb_domains = [], []

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

        ulb_data_dict[domain] = ulb_data
        ulb_domains_dict[domain] = ulb_domains
        

    
    if alg != 'fullysupervised':
        for domain in domain_list:
            path = os.path.join(data_dir, domain, 'unlabel')
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]  
            if domain in ulb_data_dict.keys():
                ulb_data_dict[domain].extend(imagepaths)
                ulb_domains_dict[domain].extend( [domain] * len(imagenames) )
            else: 
                ulb_data_dict[domain] = imagepaths
                ulb_domains_dict[domain] = [domain] * len(imagenames) 

    dset_dict = {}
    for domain in domain_list:
        dset_dict[domain] = AugmentDataset(alg, ulb_data_dict[domain], ulb_domains_dict[domain], domain_list, num_classes=num_classes, transform=transform_weak, is_ulb=True, medium_transform=transform_medium, strong_transform=transform_strong, style_transform=transform_style, sroot=sroot, onehot=False)

    return dset_dict

class AugmentDataset(BasicDataset):

    def __init__(self, alg, data, domains, domain_list, targets=None, num_classes=None, transform=None, is_ulb=False, medium_transform=None, strong_transform=None, style_transform=None, onehot=False, sroot="style", *args, **kwargs ):
        super().__init__(alg, data, targets, num_classes, transform, is_ulb, medium_transform, strong_transform, onehot, *args, **kwargs)
        self.domains = domains
        self.domain_list = domain_list
        assert len(self.domains) == len(self.data), "The domains were not handled correctly"

        self.transform_style = style_transform

        self.sroot = sroot
        self.style_lists = {}
        self.load_style()

    def load_style(self):

        for domain in self.domain_list:
            with open(os.path.join(self.sroot, '%s_style.txt' % (domain))) as s_list:
                data = s_list.read().splitlines()
                data = [os.path.join('data', 'pacs', p) for p in data if p != '']
                self.style_lists[domain] = data      

    def __sample__(self, idx):
        path =  self.data[idx]
        img = Image.open(path).convert('RGB')
        style = self.domains[idx]
        try:
            target = self.targets[idx]
        except:
            target = None
        return img, target, style

    def __getitem__(self, idx):

        img, target, sample_style = self.__sample__(idx)

        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target} 
            else: 

        
                # return {'idx_ulb': idx, 'x_ulb_w': img_w, 'style': styles}
                return {'idx_ulb': idx, 'x_ulb_w': img_w}


