# Licensed under the MIT License.
import torch
torch.autograd.set_detect_anomaly(True)

from .utils import AdaMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, Bn_Controller


@ALGORITHMS.register('adamatch')
class AdaMatch(AlgorithmBase):
    """
        AdaMatch algorithm (https://arxiv.org/abs/2106.04732).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                momentum for average probability
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(p_cutoff=args.p_cutoff, T=args.T, hard_label=args.hard_label, ema_p=args.ema_p)
    
    def init(self, p_cutoff, T, hard_label=True, ema_p=0.999):
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p

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

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'), 
            "DistAlignHook")
        self.register_hook(AdaMatchThresholdingHook(), "MaskingHook")
        super().set_hooks()


    def train_step(self, x_lb_src_w, x_lb_src_s, y_src, x_lb_tgt_w, x_lb_tgt_s, y_tgt, x_ulb_w, x_ulb_s):
        num_lb = y_tgt.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            
            # Calculate BN logit
            saved_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            bn_inputs = torch.cat((x_lb_src_w, x_lb_src_s, x_lb_tgt_w, x_lb_tgt_s))
            bn_outputs = self.model(bn_inputs)
            # bn_logit_x_lb_src_w, bn_logit_x_lb_src_s, bn_logit_x_lb_tgt_w, bn_logit_x_lb_tgt_s = bn_outputs['logits'].chunk(4)
            bn_logit_x_lb_src_w, bn_logit_x_lb_src_s, bn_logit_x_lb_tgt_w, bn_logit_x_lb_tgt_s = (
                    t.detach() for t in bn_outputs['logits'].chunk(4)
                )
            self.model.load_state_dict(saved_state)
            
            # Calculate logit
            inputs = torch.cat((x_lb_src_w, x_lb_src_s, x_lb_tgt_w, x_lb_tgt_s, x_ulb_w, x_ulb_s))
            outputs = self.model(inputs)
            logit_x_lb_src_w, logit_x_lb_src_s, logit_x_lb_tgt_w, logit_x_lb_tgt_s, logit_x_ulb_w, logit_x_ulb_s = outputs['logits'].chunk(6)

            # Logit Random Interpolation
            rand_w = torch.rand_like(logit_x_lb_src_w)
            rand_s = torch.rand_like(logit_x_lb_src_s)
            rand_tw = torch.rand_like(logit_x_lb_tgt_w)
            rand_ts = torch.rand_like(logit_x_lb_tgt_s)

            logit_x_lb_src_w_interpolated = logit_x_lb_src_w + (bn_logit_x_lb_src_w - logit_x_lb_src_w) * rand_w
            logit_x_lb_src_s_interpolated = logit_x_lb_src_s + (bn_logit_x_lb_src_s - logit_x_lb_src_s) * rand_s
            logit_x_lb_tgt_w_interpolated = logit_x_lb_tgt_w + (bn_logit_x_lb_tgt_w - logit_x_lb_tgt_w) * rand_tw
            logit_x_lb_tgt_s_interpolated = logit_x_lb_tgt_s + (bn_logit_x_lb_tgt_s - logit_x_lb_tgt_s) * rand_ts

            ent_src = self.ce_loss(logit_x_lb_src_w_interpolated, y_src, reduction='mean') + self.ce_loss(logit_x_lb_src_s_interpolated, y_src, reduction='mean')
            ent_tgt = self.ce_loss(logit_x_lb_tgt_w_interpolated, y_tgt, reduction='mean') + self.ce_loss(logit_x_lb_tgt_s_interpolated, y_tgt, reduction='mean')
            sup_loss = (ent_src + ent_tgt) / 2

            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            probs_x_lb = self.compute_prob(logit_x_lb_src_w_interpolated.detach())
            probs_x_ulb_w = self.compute_prob(logit_x_ulb_w.detach())

            # distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_lb=probs_x_lb, logits_x_ulb=probs_x_ulb_w, softmax_x_lb=False, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            # calculate loss
            unsup_loss = self.consistency_loss(logit_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss
        with torch.no_grad():    
            feat_dict = {'x_lb_src_w': outputs['feat'][:num_lb],
                        'x_lb_src_s': outputs['feat'][num_lb:2*num_lb],
                        'x_lb_tgt_w': outputs['feat'][2*num_lb:3*num_lb],
                        'x_lb_tgt_s': outputs['feat'][3*num_lb:4*num_lb],
                        'x_ulb_w': outputs['feat'][4*num_lb:5*num_lb],
                        'x_ulb_s': outputs['feat'][5*num_lb:]}

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
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

        self.call_hook("after_run")


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]