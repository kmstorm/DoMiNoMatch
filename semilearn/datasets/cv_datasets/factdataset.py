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
from semilearn.datasets.augmentation import mixup_one_target
import numpy as np
import torchvision

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def get_fact(args, alg, name, domain_list, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    source_domain = args.source_domain
    jitter = getattr(args, 'jitter', 0) 
    
    domain_dir = os.path.join(data_dir, source_domain.lower())
    data_dir = os.path.join(data_dir, name.lower())

    lb_data, lb_targets, lb_domain = [], [], []
    ulb_data, ulb_targets, ulb_domain = [], [], []

    max_per_class = num_labels // num_classes
    
    train_domain_list = [domain for domain in domain_list if domain != source_domain]
    

    # Load labeled data from all domains
    for env_type in train_domain_list:
        for label in range(num_classes):
            path = os.path.join(data_dir, env_type, 'train', str(label))
            if not os.path.exists(path):
                continue
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
            targets = [int(label)] * len(imagepaths)

            if len(imagepaths) > max_per_class:
                # Take first max_per_class for labeled data
                lb_imagepaths = imagepaths[:max_per_class]
                lb_targets_per_class = targets[:max_per_class]
                
                lb_data.extend(lb_imagepaths)
                lb_targets.extend(lb_targets_per_class)

                # Add remaining to unlabeled if specified
                if include_lb_to_ulb:
                    ulb_imagepaths = imagepaths[max_per_class:]
                    ulb_targets_per_class = targets[max_per_class:]
                    
                    ulb_data.extend(ulb_imagepaths)
                    ulb_targets.extend(ulb_targets_per_class)
            else:
                lb_data.extend(imagepaths)
                lb_targets.extend(targets)

    # Load test data from all domains
    test_data, test_targets = [], []
    for env_type in train_domain_list:
        for label in range(num_classes):
            path = os.path.join(data_dir, env_type, 'test', str(label))
            if not os.path.exists(path):
                continue
            imagenames = os.listdir(path)
            imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
            targets = [int(label)] * len(imagepaths)
            test_data.extend(imagepaths)
            test_targets.extend(targets)

    # Add unlabeled data if not fully supervised
    if alg != 'fullysupervised':
        for env_type in train_domain_list:
            unlabel_path = os.path.join(data_dir, env_type, 'unlabel')
            if os.path.exists(unlabel_path):
                imagenames = os.listdir(unlabel_path)
                imagepaths = [os.path.join(unlabel_path, imagename) for imagename in imagenames]
                ulb_data.extend(imagepaths)

    # Load test data from all domains
    domain_data, domain_targets = [], []
    for label in range(num_classes):
        path = os.path.join(domain_dir, 'train', str(label))
        if not os.path.exists(path):
            continue
        imagenames = os.listdir(path)
        imagepaths = [os.path.join(path, imagename) for imagename in imagenames]
        targets = [int(label)] * len(imagepaths)
        domain_data.extend(imagepaths)
        domain_targets.extend(targets)

    # Print statistics
    lb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1

    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(len(ulb_data)))

    augmentation = args.augmentation if hasattr(args, 'augmentation') else 'factaugment'
    
    if augmentation in ['factaugment', 'mixup']:
        # Initialize FourierMixAugment
        mix_augment = None
    
        if augmentation == 'factaugment':
            fourier_augment = FourierMixAugment(
                data_path=data_dir,
                source_domain=source_domain
            )
            mix_augment = fourier_augment
        elif augmentation == 'mixup':
            mix_augment = mixup_one_target
        else:
            raise ValueError(f"Unsupported augmentation: {augmentation}")

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
                                False, None, None, False, mix_augment, domain_list, pretransform, posttransform)
        else:
            lb_dset = MixDataset(alg, augmentation, lb_data, source_domain, domain_data, domain_targets, lb_targets, num_classes, transform_weak, 
                                False, transform_weak, transform_weak, False, mix_augment, domain_list, pretransform, posttransform)
        
        ulb_dset = MixDataset(alg, augmentation, ulb_data, source_domain, domain_data, domain_targets, None, num_classes, transform_weak, 
                            True, transform_weak, transform_weak, False, mix_augment, domain_list, pretransform, posttransform)
        eval_dset = MixDataset(alg, augmentation, test_data, source_domain, domain_data, domain_targets, test_targets, num_classes, transform_weak, 
                            False, transform_weak, transform_weak, False, mix_augment, domain_list, pretransform, posttransform)
    
    elif augmentation == 'cyclemix':
            augment_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            transform_val = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            lb_dset = CycleMixDataset(alg, lb_data, source_domain, domain_data, domain_targets, lb_targets, num_classes, augment_transform, False, augment_transform, augment_transform, False, domain_list)
            ulb_dset = CycleMixDataset(alg, ulb_data, source_domain, domain_data, domain_targets, None, num_classes, augment_transform, True, augment_transform, augment_transform, False, domain_list)
            eval_dset = CycleMixDataset(alg, test_data, source_domain, domain_data, domain_targets, test_targets, num_classes, transform_val, False, None, None, False, domain_list)

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
                 medium_transform=None, strong_transform=None, onehot=False, mix_augment=None, 
                 domain_list=None, pretransform=None, posttransform=None, *args, **kwargs): 
        super().__init__(alg, data, targets, num_classes, transform, is_ulb, 
                        medium_transform, strong_transform, onehot, *args, **kwargs)

        self.mix_augment = mix_augment
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
    
    def get_domain_image(self, domain):     
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
        domain_img, domain_target = self.get_domain_image(self.source_domain)
        # save image img and domain_img to disk for debugging
        # if not os.path.exists('debug_images'):
        #     os.makedirs('debug_images')
        # img.save(os.path.join('debug_images', f'img_{idx}.png'))
        # domain_img.save(os.path.join('debug_images', f'domain_img_{idx}.png'))

        if target is not None:
            return {'x_lb': self.transform(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            img_w = self.pretransform(img)
            dm_img = self.pretransform(domain_img)
            
            if self.augmentation == 'factaugment':
                img_mix = self.mix_augment(img_w, dm_img)
            elif self.augmentation in ['mixup', 'cyclemix']:
                img_mix, _, _ = self.mix_augment(img_w, dm_img, alpha=0.5, is_bias=True)
            
            img_w = self.posttransform(img_w)
            dm_img = self.posttransform(dm_img)
            img_mix = self.posttransform(img_mix)
            
            # # denormalize and save image img_mix and domain_img to disk for debugging
            # if not os.path.exists('debug_images'):
            #     os.makedirs('debug_images')
            # # Denormalize before saving
            # img_mix_denorm = torch.clamp(denormalize(img_mix), 0, 1)
            # dm_img_denorm = torch.clamp(denormalize(dm_img), 0, 1)

            # img_mix_pil = transforms.ToPILImage()(img_mix_denorm)
            # img_mix_pil.save(os.path.join('debug_images', f'img_mix_{idx}.png'))

            # dm_img_pil = transforms.ToPILImage()(dm_img_denorm)
            # dm_img_pil.save(os.path.join('debug_images', f'domain_img_denorm_{idx}.png'))
                        
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

class CycleMixDataset(BasicDataset):
    def __init__(self, alg, data, source_domain, domain_data, domain_targets, targets=None, 
                 num_classes=None, transform=None, is_ulb=False, medium_transform=None, 
                 strong_transform=None, onehot=False, domain_list=None, *args, **kwargs):
        super().__init__(alg, data, targets, num_classes, transform, is_ulb, 
                         medium_transform, strong_transform, onehot, *args, **kwargs)
        
        self.domain_data = domain_data
        self.domain_targets = domain_targets
        self.source_domain = source_domain
        self.domain_list = domain_list if domain_list else []
        print("Domain list: ")
        print(self.domain_list)
        
        self.cycleMixTransform = {}
        for domain in self.domain_list:
            if domain == self.source_domain:
                continue
            

    def _get_domain_from_path(self, path):
        """Extract domain name from file path"""
        for domain in self.domain_list:
            if domain in path:
                return domain
        return None
        
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
        domain = self._get_domain_from_path(path)
        return img, target, domain, path
    
    def __getitem__(self, idx):
        img, target, domain, path = self.__sample__(idx)
        # print(f"Image path: {path}, Target: {target}, Domain: {domain}")
        domain_img, domain_target = self.get_domain_image()
        
        if target is not None:
            # Labeled data
            return {
                'x_lb': self.transform(img), 
                'y_lb': target,
                'dm_lb': domain,
                'x_lb_domain': self.transform(domain_img)
            }
        else:
            # Unlabeled data
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            
            img_w = self.transform(img)
            dm_img = self.transform(domain_img)
            
            return {
                'idx_ulb': idx,
                'x_ulb_w': img_w,
                'dm_ulb': domain,
                'x_ulb_domain': dm_img,
                'x_ulb_domain_target': domain_target
            }
