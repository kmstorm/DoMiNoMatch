# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument

from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, Bn_Controller


@ALGORITHMS.register('mme')
class MME(AlgorithmBase):
    """
    Minimax Entropy (MME) for Semi-Supervised Domain Adaptation (SSDA).

    This version implements the full C-max / F-min game:
      - C-max: update classifier head (freeze features), minimize entropy (−H)
      - F-min: update feature extractor (freeze head), maximize entropy (+H)

    Falls back to −H if features or head are unavailable.
    
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
        self.ulb_loss_ratio = getattr(args, 'ulb_loss_ratio', 1.0)
        self.init(args,
                  T=getattr(args, 'T', 0.05),
                  ent_loss_ratio=getattr(args, 'ent_loss_ratio', 0.1),
                  unsup_warm_up=getattr(args, 'unsup_warm_up', 0.4))
        super().__init__(args, net_builder, tb_log, logger)

    def init(self, args, T=0.05, ent_loss_ratio=0.1, unsup_warm_up=0.4):
        self.T = T
        self.ent_loss_ratio = ent_loss_ratio
        self.unsup_warm_up = unsup_warm_up
        
    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return
            
        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict['src_lb'] = get_data_loader(self.args,
                                                 self.dataset_dict['src_lb'],
                                                 self.args.batch_size,
                                                 data_sampler=self.args.train_sampler,
                                                 num_iters=self.num_train_iter,
                                                 num_epochs=self.epochs,
                                                 num_workers=self.args.num_workers,
                                                 distributed=self.distributed)
        
        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed)

        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)
        
        if self.dataset_dict['test'] is not None:
            loader_dict['test'] =  get_data_loader(self.args,
                                                   self.dataset_dict['test'],
                                                   self.args.eval_batch_size,
                                                   # make sure data_sampler is None for evaluation
                                                   data_sampler=None,
                                                   num_workers=self.args.num_workers,
                                                   drop_last=False)
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    @staticmethod
    def _entropy_from_logits(logits):
        probs = F.softmax(logits, dim=1)
        return (-(probs * (probs + 1e-5).log()).sum(1)).mean()

    def _warmup(self):
        try:
            return min(float(self.it / (self.unsup_warm_up * self.num_train_iter)), 1.0)
        except Exception:
            return 1.0

    def train_step(self, x_lb_src_w, y_src, x_lb_tgt_w, y_tgt, x_ulb_w):
        """
        MME: Dùng cả labeled + unlabeled data với minimax entropy
        """
        num_lb = y_tgt.shape[0]

        with self.amp_cm():
            # Forward pass
            if self.use_cat:
                x = torch.cat((x_lb_src_w, x_lb_tgt_w, x_ulb_w))
                out = self.model(x)
                logits_x_lb = out['logits'][:num_lb * 2]
                logits_x_ulb_w = out['logits'][num_lb * 2:]
                feats = out['feat']
                feats_x_lb = feats[:num_lb * 2] if feats is not None else None
                feats_x_ulb_w = feats[num_lb * 2:] if feats is not None else None
            else:
                out_lb = self.model( torch.cat((x_lb_src_w, x_lb_tgt_w)) )
                out_ulb = self.model(x_ulb_w)
                logits_x_lb = out_lb['logits']
                logits_x_ulb_w = out_ulb['logits']
                feats_x_lb = out_lb.get('feat', None)
                feats_x_ulb_w = out_ulb.get('feat', None)

            # Supervised loss (source + target labeled)
            if feats_x_lb is not None and hasattr(self.model, 'head') and isinstance(self.model.head, nn.Linear):
                feats_lb_n = F.normalize(feats_x_lb, p=2, dim=1)
                logits_sup = self.model.head(feats_lb_n) / self.T
            else:
                logits_sup = logits_x_lb
            sup_loss = F.cross_entropy(logits_sup, torch.cat((y_src, y_tgt)), reduction='mean')

            # Unsupervised Minimax Entropy
            unsup_loss = torch.tensor(0.0, device=sup_loss.device)
            if feats_x_ulb_w is not None and hasattr(self.model, 'head') and isinstance(self.model.head, nn.Linear):
                feats_ulb_n = F.normalize(feats_x_ulb_w, p=2, dim=1)

                # C-max: update head, freeze features → −H
                logits_c = self.model.head(feats_ulb_n.detach()) / self.T
                H_c = self._entropy_from_logits(logits_c)
                adv = - H_c

                # F-min: update features, freeze head → +H
                W = self.model.head.weight.detach()
                b = self.model.head.bias.detach() if getattr(self.model.head, 'bias', None) is not None else None
                logits_f = F.linear(feats_ulb_n, W, b) / self.T
                H_f = self._entropy_from_logits(logits_f)
                adv += H_f

                unsup_loss = self.ent_loss_ratio * self._warmup() * adv
            else:
                # Fallback: C-max only (không có head để F-min)
                logits_fallback = logits_x_ulb_w / self.T
                H = self._entropy_from_logits(logits_fallback)
                unsup_loss = self.ent_loss_ratio * self._warmup() * (-H)  # maximize entropy (C-max)

            total_loss = sup_loss + self.ulb_loss_ratio * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            total_loss=total_loss.item(),
            adv_coef=float(self.ent_loss_ratio * self._warmup()),
        )
        return out_dict, log_dict

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")

            # Check if S+T mode (baseline)
            if getattr(self.args, 'use_s_t_only', False):
                # S+T MODE: Chỉ dùng source + target labeled
                for src_lb, tgt_lb in zip(self.loader_dict['src_lb'], 
                                         self.loader_dict['train_lb']):
                    
                    # prevent the training iterations exceed args.num_train_iter
                    if self.it >= self.num_train_iter:
                        break

                    self.call_hook("before_train_step")
                    processed_batch = self.process_batch(**src_lb, **tgt_lb)
                    self.out_dict, self.log_dict = result
                    self.call_hook("after_train_step")
                    self.it += 1
                    
            else:
                # MME MODE: Dùng cả unlabeled data
                for src_lb, data_lb, data_ulb in zip( self.loader_dict['src_lb'],
                                             self.loader_dict['train_lb'],
                                             self.loader_dict['train_ulb']):
                    
                    # prevent the training iterations exceed args.num_train_iter
                    if self.it >= self.num_train_iter:
                        break

                    self.call_hook("before_train_step")
                    
                    processed_batch = self.process_batch(**src_lb, **data_lb, **data_ulb)
                    result = self.train_step(**processed_batch)
                    self.out_dict, self.log_dict = result
                    self.call_hook("after_train_step")
                    self.it += 1
                    
            self.call_hook("after_train_epoch")
            
            if self.it >= self.num_train_iter:
                break

        self.call_hook("after_run")


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.05, 'Temperature for entropy calculation'),
            SSL_Argument('--ent_loss_ratio', float, 0.2, 'Weight for adversarial entropy loss'),
            SSL_Argument('--unsup_warm_up', float, 0.2, 'Ramp-up ratio for unsupervised loss'),
        ]
