# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn.functional as F

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register('mme')
class MME(AlgorithmBase):
    """
        Minimax Entropy (MME) algorithm for Semi-Supervised Domain Adaptation.
        Based on the paper: "Semi-supervised Domain Adaptation via Minimax Entropy" (ICCV 2019)
        
        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - T (`float`, *optional*, defaults to 0.05):
            Temperature for entropy calculation
        - ent_loss_ratio (`float`, *optional*, defaults to 0.1):
            Weight for adversarial entropy loss
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # MME specified arguments
        self.init(T=args.T, ent_loss_ratio=args.ent_loss_ratio, unsup_warm_up=args.unsup_warm_up)

    def init(self, T=0.05, ent_loss_ratio=0.1, unsup_warm_up=0.4):
        self.T = T
        self.ent_loss_ratio = ent_loss_ratio
        self.unsup_warm_up = unsup_warm_up

    def train_step(self, x_lb, y_lb, x_ulb_w):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w = outputs['logits'][num_lb:]
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w = outputs['feat'][num_lb:]
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
                
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w}

            # Supervised loss
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # Adversarial entropy loss for unlabeled data (MME)
            # This encourages the model to be confident on unlabeled data
            adv_ent_loss = self.adversarial_entropy_loss(logits_x_ulb_w)

            # Warm up for unsupervised loss
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter), a_min=0.0, a_max=1.0)
            
            total_loss = sup_loss + self.ent_loss_ratio * adv_ent_loss * unsup_warmup

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         adv_ent_loss=adv_ent_loss.item(),
                                         total_loss=total_loss.item())
        return out_dict, log_dict

    def adversarial_entropy_loss(self, logits):
        """
        Adversarial entropy loss for MME.
        This loss encourages the model to be confident (low entropy) on unlabeled data.
        """
        # Apply temperature scaling
        logits_scaled = logits / self.T
        probs = F.softmax(logits_scaled, dim=1)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=1)
        
        # Return negative entropy (adversarial entropy)
        # This encourages minimization of entropy (maximization of confidence)
        return torch.mean(-entropy)

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.05, 'Temperature for entropy calculation'),
            SSL_Argument('--ent_loss_ratio', float, 0.1, 'Weight for adversarial entropy loss'),
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
        ]
