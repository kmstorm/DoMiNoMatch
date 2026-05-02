import torch
import torch.nn.functional as F

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

from semilearn.core.criterions import SoftSupConLoss, CRDLoss
from semilearn.algorithms.adamatch.utils import AdaMatchThresholdingHook

# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

@ALGORITHMS.register('dominomatch')
class DoMiNoMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 

        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        self.lambda_c = args.crossdomain_loss_ratio
        self.lambda_ct = args.contrastive_loss_ratio  
        self.warmup_kl_iter = args.warmup_kl_iter
        self.contrastive_warmup_iter = args.warmup_contrastive_iter
        self.current_iter = 0
        self.contrastive_loss = CRDLoss()
        self.teacher_model = self._load_teacher(args.pretrain_path)
        self.teacher_model.eval()
    
    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        
    def _load_teacher(self, pretrain_path):
        from copy import deepcopy
        teacher = deepcopy(self.model)
        state_dict = torch.load(pretrain_path, map_location='cpu')
        state_dict = state_dict.get('model', state_dict)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        teacher.load_state_dict(state_dict, strict=False)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher.cuda()
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'), 
            "DistAlignHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        # self.register_hook(AdaMatchThresholdingHook(), "MaskingHook")
        super().set_hooks()
        
    def update_lambda_c(self):
        if self.current_iter < self.warmup_kl_iter:
            self.lambda_c = self.lambda_c * (self.current_iter / self.warmup_kl_iter)


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, x_ulb_domain, x_ulb_domain_target):
        
        self.current_iter += 1

        with self.amp_cm():
            # # Student model outputs
            # outs_x_lb = self.model(x_lb)
            # logits_x_lb = outs_x_lb['logits']
            # feats_x_lb = outs_x_lb['feat']
            
            # outs_x_ulb_s = self.model(x_ulb_s)
            # logits_x_ulb_s = outs_x_ulb_s['logits']
            # feats_x_ulb_s = outs_x_ulb_s['feat']
            # # embeds_x_ulb_s = outs_x_ulb_s['proj']
            
            # # Try gradient accumulation here
            # outs_x_ulb_w = self.model(x_ulb_w)
            # logits_x_ulb_w = outs_x_ulb_w['logits']
            # feats_x_ulb_w = outs_x_ulb_w['feat']
            # embeds_x_ulb_w = outs_x_ulb_w['proj']
            
            num_lb = x_lb.size(0)
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s), dim=0)
            outputs = self.model(inputs)
            logits_x_lb = outputs['logits'][:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            
            feats_x_lb = outputs['feat'][:num_lb]
            feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            
            all_embeds = outputs['proj']
            embeds_x_lb = all_embeds[:num_lb]
            embeds_x_ulb_w, embeds_x_ulb_s = all_embeds[num_lb:].chunk(2)

            
            
            with torch.no_grad():    
                # #Teacher embeddings for domain adaptation
                # teacher_outs_ulb_s = self.teacher_model(x_ulb_s)
                # teacher_embeds_ulb_s = teacher_outs_ulb_s['proj']
                # teacher_logits_ulb_s = teacher_outs_ulb_s['logits']
                
                teacher_outs_dm = self.teacher_model(x_ulb_domain)
                teacher_embeds_dm = teacher_outs_dm['proj']
                teacher_logits_dm = teacher_outs_dm['logits']

            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s, 'teacher_ulb_s': teacher_embeds_dm}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_lb = self.compute_prob(logits_x_lb.detach())
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_lb=probs_x_lb, logits_x_ulb=probs_x_ulb_w, softmax_x_lb=False, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)
            
            crossdomain_loss = F.kl_div(
                F.log_softmax(logits_x_ulb_s, dim=-1),
                F.softmax(logits_x_ulb_w.detach() / self.T, dim=-1),
                reduction='none'
            ).sum(dim=1, keepdim=False)
            
            crossdomain_loss = (crossdomain_loss * mask).mean()
            
            # calculate contrastive loss
            mask_bool = mask.bool()
            student_features = embeds_x_ulb_w[mask_bool]  # [N_high, 128]
            # teacher_features = teacher_embeds_ulb_s[mask_bool].detach()  # [N_high, 128]
            teacher_features = teacher_embeds_dm[mask_bool].detach()  # [N_high, 128]
            teacher_probs = torch.softmax(teacher_logits_dm[mask_bool], dim=-1)  # [N_high, num_classes]
            teacher_pseudo_label = teacher_probs.argmax(dim=1)
            
            features = torch.stack([student_features, teacher_features], dim=1)  # [N_high, 2, 128]
            features = F.normalize(features, p=2, dim=2) 
            
            if self.current_iter >= self.contrastive_warmup_iter:
                consup_loss = self.contrastive_loss(student_features, teacher_features)
            else:
                consup_loss = torch.tensor(0.).cuda()

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_c * crossdomain_loss + self.lambda_ct * consup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         crossdomain_loss=crossdomain_loss.item(),
                                         contrastive_distill_loss=consup_loss.item(),
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.8),
            SSL_Argument('--warmup_kl_iter', int, 10000, 'Number of iterations to warm up the KL divergence loss'),
            SSL_Argument('--warmup_contrastive_iter', int, 10000, 'Number of iterations to warm up the contrastive loss'),
            SSL_Argument('--lambda_crossdomain', float, 0.1, 'Weight for the cross-domain KL divergence loss'),
            SSL_Argument('--lambda_contrastive', float, 0.1, 'Weight for the contrastive loss'),
        ]