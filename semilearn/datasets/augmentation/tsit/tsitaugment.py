import os
from .models.networks import find_network_using_name, create_network
from .models.networks.generator import TSITGenerator
from semilearn.datasets.augmentation.tsit.util import load_network
import torch


class TSITAugment():


    def __init__(self, domain_list, loadpath, args):

        self.args = args
        self.opt = {
            'alpha': 1.0,
            'aspect_ratio': 2.0,
            'batchSize': 1,
            'cache_filelist_read': False,
            'cache_filelist_write': False,
            'checkpoints_dir': './checkpoints',
            'contain_dontcare_label': False,
            'croot': './datasets/carcolor',
            'crop_size': 256,
            'dataset_mode': 'norm2diff',
            'display_winsize': 256,
            'gpu_ids': [0],
            'how_many': float('inf'),
            'init_type': 'xavier',
            'init_variance': 0.02,
            'isTrain': False,
            'label_nc': 3,
            'load_from_opt_file': False,
            'load_size': 256,
            'max_dataset_size': 9223372036854775807,
            'model': 'pix2pix',
            'nThreads': 4,
            'name': 'mmis_norm2diff',
            'nef': 16,
            'netG': 'tsit',
            'ngf': 64,
            'no_flip': True,
            'no_instance': True,
            'no_pairing_check': True,
            'no_ss': False,
            'norm_D': 'spectralinstance',
            'norm_E': 'spectralinstance',
            'norm_G': 'spectralfadesyncbatch3x3',
            'norm_S': 'spectralinstance',
            'num_upsampling_layers': 'more',
            'output_nc': 3,
            'phase': 'test',
            'preprocess_mode': 'fixed',
            'results_dir': './results',
            'semantic_nc': 3,
            'serial_batches': True,
            'show_input': True,
            'sroot': '../data/style',
            'task': 'MMIS',
            'test_mode': 'all',
            'use_vae': False,
            'which_epoch': 'latest',
            'z_dim': 256
        }
        self.opt['batchSize'] = self.args.batch_size

        self.domain_list = domain_list
        self.loadpath = loadpath
        self.netG = {}

        for domain in domain_list:
            model_path = os.path.join(loadpath, f"{domain}_net_G.pth") 
            self.netG[domain] = self.loadnetG(model_path)

    def loadnetG(self, save_path):
        
        netG = TSITGenerator(self.opt)
        netG = load_network(netG, save_path)
        netG = netG.cuda(self.args.gpu)
        netG.eval()
        return netG
    
    def __call__(self, content, styles, domain):
        
        augmented_content = {} 

        for aug_domain in self.domain_list:
            with torch.no_grad():
                if aug_domain != domain:
                    augmented_content[aug_domain] = self.netG[domain](content, styles[aug_domain], z=None)
                else:
                    augmented_content[aug_domain] = None
        
        return augmented_content
    
    
if __name__ == "__main__":
    pass