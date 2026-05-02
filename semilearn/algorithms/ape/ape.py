# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register('ape')
class APE(AlgorithmBase):
    """
    Attract, Perturb, and Explore (APE) – ECCV 2020.

    - Attraction: MMD between labeled (source + labeled target) vs. unlabeled target features.
    - Perturbation: (i) perturb prototypes toward entropy-maximizing direction,
                    (ii) create bounded feature perturbations toward the perturbed prototypes,
                    (iii) regularize with KL divergence.
    - Exploration: low-entropy selective pseudo-labeling via nearest (highest-similarity) prototype.

    Notes:
      * Uses the linear classifier head's weights as trainable class prototypes on a spherical space.
      * Falls back gracefully if 'feat' or a linear 'head' is not available.
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.ulb_loss_ratio = getattr(args, 'ulb_loss_ratio', 1.0)

        # Defaults follow APE usage in later SSDA repros: alpha=10, beta=1, gamma=10
        self.init(
            args,
            T=getattr(args, 'T', 0.05),
            ape_alpha=getattr(args, 'ape_alpha', 10.0),   # weight for Attraction (MMD)
            ape_beta=getattr(args, 'ape_beta', 1.0),      # weight for Perturbation (KL)
            ape_gamma=getattr(args, 'ape_gamma', 10.0),   # weight for Exploration (CE on low-entropy)
            ape_tau=getattr(args, 'ape_tau', 0.5),        # entropy threshold (nats) for Exploration
            ape_eps=getattr(args, 'ape_eps', 1e-2),       # feature perturbation L2 budget
            ape_step=getattr(args, 'ape_step', 1e-2),     # step size for delta update
            ape_proto_step=getattr(args, 'ape_proto_step', 1e-3),  # prototype perturb step
            mmd_sigmas=getattr(args, 'ape_mmd_sigmas', '0.5,1.0,2.0')
        )
        super().__init__(args, net_builder, tb_log, logger)

    def init(self, args, T, ape_alpha, ape_beta, ape_gamma, ape_tau, ape_eps, ape_step, ape_proto_step, mmd_sigmas):
        self.T = float(T)
        self.ape_alpha = float(ape_alpha)
        self.ape_beta = float(ape_beta)
        self.ape_gamma = float(ape_gamma)
        self.ape_tau = float(ape_tau)
        self.ape_eps = float(ape_eps)
        self.ape_step = float(ape_step)
        self.ape_proto_step = float(ape_proto_step)
        # parse sigmas
        self._mmd_sigmas = torch.tensor([float(s) for s in str(mmd_sigmas).split(',') if s], dtype=torch.float32)

    # ------------------------- helpers -------------------------

    @staticmethod
    def _entropy_probs(probs, eps=1e-5):
        return (-(probs * (probs + eps).log()).sum(1))  # per-sample entropy (nats)

    @staticmethod
    def _pairwise_sq_dists(x, y):  # x:[n,d], y:[m,d]
        x2 = (x * x).sum(1, keepdim=True)   # [n,1]
        y2 = (y * y).sum(1, keepdim=True).t()  # [1,m]
        return x2 + y2 - 2 * x @ y.t()

    def _mmd_rbf(self, xa, xb):
        # xa: aligned/labeled feats [na,d], xb: unlabeled feats [nb,d]
        # mixture of RBF kernels with sigmas from args
        if xa.numel() == 0 or xb.numel() == 0:
            return xa.new_zeros([])
        d_aa = self._pairwise_sq_dists(xa, xa)
        d_bb = self._pairwise_sq_dists(xb, xb)
        d_ab = self._pairwise_sq_dists(xa, xb)
        mmd = 0.0
        for sigma in self._mmd_sigmas.to(xa.device):
            gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)
            K_aa = torch.exp(-gamma * d_aa)
            K_bb = torch.exp(-gamma * d_bb)
            K_ab = torch.exp(-gamma * d_ab)
            # unbiased estimate: remove diagonals
            m = xa.size(0); n = xb.size(0)
            if m > 1:
                mmd += (K_aa.sum() - K_aa.diag().sum()) / (m * (m - 1))
            if n > 1:
                mmd += (K_bb.sum() - K_bb.diag().sum()) / (n * (n - 1))
            mmd -= 2.0 * K_ab.mean()
        return mmd / len(self._mmd_sigmas)

    def _get_feats_logits(self, x):
        out = self.model(x)
        return out.get('feat', None), out['logits']

    def _proto_logits(self, feats, use_head=True):
        """
        Compute similarity logits on spherical space using the model head as prototypes.
        """
        if feats is None:
            return None
        z = F.normalize(feats, p=2, dim=1)
        if use_head and hasattr(self.model, 'head') and isinstance(self.model.head, nn.Linear):
            W = self.model.head.weight  # [C, D]
            Wn = F.normalize(W, p=2, dim=1)
            logits = F.linear(z, Wn) / self.T
        else:
            # fall back to whatever logits the model produced (outside caller)
            logits = None
        return logits, z

    # ------------------------- core train step -------------------------

    def train_step(self, x_lb, y_lb, x_ulb_w):
        num_lb = y_lb.shape[0]

        with self.amp_cm():
            # forward (cat path preserves BN stats like USB)
            if self.use_cat:
                x = torch.cat((x_lb, x_ulb_w), dim=0)
                feats_all, logits_all = self._get_feats_logits(x)
                feats_lb = feats_all[:num_lb] if feats_all is not None else None
                feats_ulb = feats_all[num_lb:] if feats_all is not None else None
                logits_lb = logits_all[:num_lb]
                logits_ulb = logits_all[num_lb:]
            else:
                feats_lb, logits_lb = self._get_feats_logits(x_lb)
                feats_ulb, logits_ulb = self._get_feats_logits(x_ulb_w)

            # --- Supervised CE on labeled (source + labeled target) ---
            logits_sup, z_lb = self._proto_logits(feats_lb)
            if logits_sup is None:   # fallback if no feat/head
                logits_sup = logits_lb
                z_lb = None
            # Fix: Ensure y_lb is long type
            y_lb = y_lb.long()
            sup_loss = F.cross_entropy(logits_sup, y_lb, reduction='mean')

            # Prepare unlabeled branch (preds on spherical head)
            logits_ulb_s, z_ulb = self._proto_logits(feats_ulb)
            if logits_ulb_s is None:
                logits_ulb_s = logits_ulb
            probs_ulb = F.softmax(logits_ulb_s, dim=1)
            ent_ulb = self._entropy_probs(probs_ulb)  # [B]

            # --- A) Attraction (MMD between labeled vs. unlabeled features) ---
            attr_loss = torch.tensor(0.0, device=logits_lb.device)
            if z_lb is not None and z_ulb is not None:
                # Fix: Only detach labeled features, let gradient flow into unlabeled features
                attr_loss = self._mmd_rbf(z_lb.detach(), z_ulb)

            # --- B) Perturbation (proto & feature) + KL regularization ---
            pert_loss = torch.tensor(0.0, device=logits_lb.device)
            if z_ulb is not None and hasattr(self.model, 'head') and isinstance(self.model.head, nn.Linear):
                # 1) entropy gradient wrt prototypes
                W = self.model.head.weight  # [C,D], trainable
                Wn = F.normalize(W, p=2, dim=1)
                logits_base = F.linear(z_ulb.detach(), Wn) / self.T
                probs_base = F.softmax(logits_base, dim=1)
                H = self._entropy_probs(probs_base).mean()
                (gradW,) = torch.autograd.grad(H, W, retain_graph=False, create_graph=False)
                # perturb prototypes in entropy-max direction (small step), then renorm
                Wp = W + self.ape_proto_step * gradW
                Wp = F.normalize(Wp, p=2, dim=1).detach()

                # 2) one-step feature perturbation toward perturbed prototypes
                delta = torch.zeros_like(z_ulb, requires_grad=True)
                z_ulb_det = z_ulb.detach()  # Fix: Detach z_ulb for internal computation
                logits_pert = F.linear(F.normalize(z_ulb_det + delta, p=2, dim=1), Wp) / self.T
                # KL(p||p̃) with teacher detached
                with torch.no_grad():
                    p_teacher = F.softmax(F.linear(z_ulb_det, Wn) / self.T, dim=1)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_pert, dim=1),
                    p_teacher,
                    reduction='batchmean'
                )
                # Fix: Use autograd.grad instead of backward()
                (g_delta,) = torch.autograd.grad(loss_kl, delta, retain_graph=False, create_graph=False)
                
                # FGSM-like step & projection to L2 ball
                with torch.no_grad():
                    if g_delta is not None:
                        delta += self.ape_step * g_delta / (g_delta.norm(p=2, dim=1, keepdim=True) + 1e-12)
                        # project to L2 epsilon
                        norm = delta.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                        scale = (self.ape_eps / norm).clamp(max=1.0)
                        delta.mul_(scale)
                # recompute KL with no grad through delta update graph
                delta = delta.detach()
                logits_pert = F.linear(F.normalize(z_ulb + delta, p=2, dim=1), Wp) / self.T
                pert_loss = F.kl_div(
                    F.log_softmax(logits_pert, dim=1),
                    p_teacher,
                    reduction='batchmean'
                )

            # --- C) Exploration (low-entropy pseudo-labeling) ---
            exp_loss = torch.tensor(0.0, device=logits_lb.device)
            with torch.no_grad():
                mask = (ent_ulb < self.ape_tau)  # boolean mask
                pseudo = probs_ulb.argmax(dim=1)
                # Fix: Add debug logging for mask ratio
                if self.it < 10 and hasattr(self, "logger") and self.logger:
                    self.logger.info(f"[APE] mask ratio: {mask.float().mean().item():.3f}")
            if mask.any():
                exp_loss = F.cross_entropy(logits_ulb_s[mask], pseudo[mask], reduction='mean')

            # total
            total_loss = (
                sup_loss
                + self.ulb_loss_ratio * (
                    self.ape_alpha * attr_loss
                    + self.ape_beta * pert_loss
                    + self.ape_gamma * exp_loss
                )
            )

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(
            sup_loss=float(sup_loss),
            attr_loss=float(attr_loss),
            pert_loss=float(pert_loss),
            exp_loss=float(exp_loss),
            total_loss=float(total_loss),
            ape_alpha=float(self.ape_alpha),
            ape_beta=float(self.ape_beta),
            ape_gamma=float(self.ape_gamma),
            tau=float(self.ape_tau),
        )
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.05, 'Temperature for similarity logits'),
            SSL_Argument('--ape_alpha', float, 10.0, 'Weight for Attraction (MMD)'),
            SSL_Argument('--ape_beta', float, 1.0, 'Weight for Perturbation (KL)'),
            SSL_Argument('--ape_gamma', float, 10.0, 'Weight for Exploration (CE)'),
            SSL_Argument('--ape_tau', float, 0.5, 'Entropy threshold (nats) for Exploration'),
            SSL_Argument('--ape_eps', float, 1e-2, 'L2 budget for feature perturbation'),
            SSL_Argument('--ape_step', float, 1e-2, 'Step size for feature perturbation'),
            SSL_Argument('--ape_proto_step', float, 1e-3, 'Step size for prototype perturbation'),
            SSL_Argument('--ape_mmd_sigmas', str, '0.5,1.0,2.0', 'Comma-separated RBF sigmas for MMD'),
        ]