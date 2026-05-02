# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision import transforms
import numpy as np

# CYCLEGAN Experiments
from .cyclemix.cyclegan.networks import define_G

PRETRAINED_PATH = 'pretrained_weight/cyclegan/weights/PACS'

class CycleMixLayer(nn.Module):
    def __init__(self, sources, device):
        super(CycleMixLayer, self).__init__()

        if torch.cuda.is_available():
            gpu_id = [0]
        else:
            gpu_id = []
        self.device = device
        self.sources = sources
        if len(self.sources) == 3:
            source1, source2, source3 = self.sources

            self.source1 = source1
            self.source2 = source2
            self.source3 = source3

            self.gan1_2 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                   False, 'normal', 0.02, gpu_id)

            self.gan1_2.load_state_dict(
                torch.load(PRETRAINED_PATH + source1 + '2' + source2 + '.pth',
                           map_location=torch.device(self.device)))
            self.gan1_2.eval()

            self.gan1_3 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                   False, 'normal', 0.02, gpu_id)

            self.gan1_3.load_state_dict(
                torch.load(PRETRAINED_PATH + source1 + '2' + source3 + '.pth',
                           map_location=torch.device(self.device)))
            self.gan1_3.eval()

            self.gan2_1 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                   False, 'normal', 0.02, gpu_id)

            self.gan2_1.load_state_dict(
                torch.load(PRETRAINED_PATH + source2 + '2' + source1 + '.pth',
                           map_location=torch.device(self.device)))
            self.gan2_1.eval()

            self.gan2_3 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                   False, 'normal', 0.02, gpu_id)

            self.gan2_3.load_state_dict(
                torch.load(PRETRAINED_PATH + source2 + '2' + source3 + '.pth',
                           map_location=torch.device(self.device)))
            self.gan2_3.eval()

            self.gan3_1 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                   False, 'normal', 0.02, gpu_id)

            self.gan3_1.load_state_dict(
                torch.load(PRETRAINED_PATH + source3 + '2' + source1 + '.pth',
                           map_location=torch.device(self.device)))
            self.gan3_1.eval()

            self.gan3_2 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                   False, 'normal', 0.02, gpu_id)

            self.gan3_2.load_state_dict(
                torch.load(PRETRAINED_PATH + source3 + '2' + source2 + '.pth',
                           map_location=torch.device(self.device)))
            self.gan3_2.eval()
        elif len(self.sources) == 2:
            source1, source2 = self.sources
            self.source1 = source1
            self.source2 = source2

            self.gan1_2 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                   False, 'normal', 0.02, gpu_id)

            self.gan1_2.load_state_dict(
                torch.load(PRETRAINED_PATH + source1 + '2' + source2 + '.pth',
                           map_location=torch.device(self.device)))
            self.gan1_2.eval()

            self.gan2_1 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                   False, 'normal', 0.02, gpu_id)

            self.gan2_1.load_state_dict(
                torch.load(PRETRAINED_PATH + source2 + '2' + source1 + '.pth',
                           map_location=torch.device(self.device)))
            self.gan2_1.eval()
        else:
            raise NotImplementedError

    def forward(self, x: list):
        if len(self.sources) == 3:
            b1, b2, b3 = x
            x_1, y_task_1 = b1
            x_2, y_task_2 = b2
            x_3, y_task_3 = b3

            del b1, b2, b3

            # GAN TRANSFORMATIONS
            norm = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            alpha, beta = np.round(np.random.dirichlet(np.ones(2)), 2)

            x_1_hat = x_1 + (alpha * self.gan1_2(x_1)) + (beta * self.gan1_3(x_1))
            x_1_hat.detach()
            x_1_hat = norm(x_1_hat)

            x_2_hat = (alpha * self.gan2_1(x_2)) + x_2 + (beta * self.gan2_3(x_2))
            x_2_hat.detach()
            x_2_hat = norm(x_2)

            x_3_hat = (alpha * self.gan3_1(x_3)) + (beta * self.gan3_2(x_3)) + x_3
            x_3_hat.detach()
            x_3_hat = norm(x_3)

            return [(x_1, y_task_1), (x_2, y_task_2), (x_3, y_task_3),
                    (x_1_hat, y_task_1), (x_2_hat, y_task_2), (x_3_hat, y_task_3)]

        else:
            b1, b2 = x
            x_1, y_task_1 = b1
            x_2, y_task_2 = b2

            del b1, b2
            # GAN TRANSFORMATIONS
            norm = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            # alpha, beta = np.round(np.random.dirichlet(np.ones(2)), 2)
            alpha = np.round(np.random.random(), 2)

            x_1_hat = x_1 + (alpha * self.gan1_2(x_1))
            x_1_hat.detach()
            x_1_hat = norm(x_1)

            x_2_hat = (alpha * self.gan2_1(x_2)) + x_2
            x_2_hat.detach()
            x_2_hat = norm(x_2)

            return [(x_1, y_task_1), (x_2, y_task_2),
                    (x_1_hat, y_task_1), (x_2_hat, y_task_2)]
            
class CycleMixAugment():
    def __init__(self, source1, source2, device='cuda'):
        super(CycleMixAugment, self).__init__()
        
        if torch.cuda.is_available() and device == 'cuda':
            gpu_id = [0]
        else:
            gpu_id = []
        self.device = device
        
        self.source1 = source1
        self.source2 = source2
        
        # Load the GAN model for transforming from source1 to source2
        self.gan1_2 = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                              False, 'normal', 0.02, gpu_id)
        
        self.gan1_2.load_state_dict(
            torch.load(PRETRAINED_PATH + source1 + '2' + source2 + '.pth',
                      map_location=torch.device(self.device)))
        self.gan1_2.eval()
        
        # Normalization transform to apply after mixing
        self.norm = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        # model to cuda
        self.gan1_2.to(self.device)
    
    def __call__(self, x):
        """
        Mix x with domain transformation
        
        Args:
            x: input image
            x_dm: domain image (not used for transformation, only for reference)
        
        Returns:
            mixed image after applying CycleGAN transformation
        """
                
        # Random mixing coefficient
        alpha = np.round(np.random.random(), 2)
        
        # Apply GAN transformation and mix with original
        x_mixed = x + (alpha * self.gan1_2(x))
        
        # Detach to prevent gradient flow through the GAN
        x_mixed = x_mixed.detach()
        
        # Apply normalization
        x_mixed = self.norm(x_mixed)
        
        # Return randomly mixed image or original 50:50
        if np.random.rand() > 0.5:
            return x_mixed
        else:
            return x