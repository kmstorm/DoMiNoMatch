""" The Code is under Tencent Youtu Public Rule
Part of the code is adopted form SupContrast as in the comment in the class
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, max_probs, labels=None, mask=None, reduction="mean", select_matrix=None):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None and select_matrix is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs, max_probs.T)
            # Some may find that the line 59 is different with eq(6)
            # Acutuall the final mask will set weight=0 when i=j, following Eq(8) in paper
            # For more details, please see issue 9

            # Set diagional to 1 to be same with eq(8) as in issue 9
            # https://github.com/TencentYoutuResearch/Classification-SemiCLS/issues/9
            # Not that our results in paper doesn't have following line and should
            # mathematically be better after adding.
            score_mask = score_mask.fill_diagonal_(1)
            mask = mask.mul(score_mask) * select_matrix

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            #max_probs = max_probs.reshape((batch_size,1))
            max_probs = max_probs.contiguous().view(-1, 1)
            score_mask = torch.matmul(max_probs,max_probs.T)
            # Set diagional to 1 to be same with eq(8) as in issue
            # https://github.com/TencentYoutuResearch/Classification-SemiCLS/issues/9
            # Not that our results in paper doesn't have following line and should
            # mathematically be better after adding.
            score_mask = score_mask.fill_diagonal_(1)
            mask = mask.mul(score_mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)
        
        if reduction == "mean":
            loss = loss.mean()

        return loss
    
class CRDLoss(nn.Module):
    """
    Contrastive Representation Distillation Loss (CRD)
    https://arxiv.org/abs/1910.10699

    Contrast between student and teacher embedding pairs (positive)
    and teacher embeddings of other samples (negatives).
    """
    def __init__(self, temperature=0.07):
        super(CRDLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_feat, teacher_feat):
        """
        Args:
            student_feat: tensor of shape [N, D]
            teacher_feat: tensor of shape [N, D]
        Returns:
            loss (scalar): contrastive distillation loss
        """
        
        # Handle empty batch case
        if student_feat.shape[0] == 0 or teacher_feat.shape[0] == 0:
            return torch.tensor(0.0, device=student_feat.device, requires_grad=True)
        
        # Check for zero vectors and add small noise if needed
        if torch.any(torch.sum(student_feat**2, dim=1) < 1e-8):
            student_feat = student_feat + 1e-8 * torch.randn_like(student_feat)
        if torch.any(torch.sum(teacher_feat**2, dim=1) < 1e-8):
            teacher_feat = teacher_feat + 1e-8 * torch.randn_like(teacher_feat)
            
        # Normalize
        student_feat = F.normalize(student_feat, dim=1)
        teacher_feat = F.normalize(teacher_feat, dim=1)

        batch_size = student_feat.shape[0]
        device = student_feat.device

        # Compute similarity matrix with safe temperature
        temperature = max(self.temperature, 1e-8)
        sim_matrix = torch.matmul(student_feat, teacher_feat.T) / temperature

        # Targets: diagonal (positive pair at same index)
        targets = torch.arange(batch_size).long().to(device)

        # Contrastive loss: student i should match teacher i
        loss = self.criterion(sim_matrix, targets)
        return loss
    
if __name__ == "__main__":
    # # Example usage
    # features = torch.randn(16, 1, 128)  # 16 samples, 1 view, 128-dim features
    # features = torch.nn.functional.normalize(features, p=2, dim=2)
    # max_probs = torch.rand(16, 1)
    # max_probs = max_probs * 0.5 + 0.5 # Random max probabilities for each sample
    # labels = torch.tensor([0, 1, 0, 3, 2, 6, 4, 4, 3, 5, 1, 2, 2, 5, 6, 0])  # Labels for the samples
    # loss_fn = SoftSupConLoss()
    # loss = loss_fn(features, max_probs, labels=labels)
    # print("Loss:", loss.item())
    
    # Create 3 classes, each with 5 samples (total 15)
    num_classes = 3
    samples_per_class = 5
    feature_dim = 128

    features = []
    labels = []
    for cls in range(num_classes):
        # Center for this class
        center = torch.randn(1, feature_dim)
        # Samples are close to the center (add small noise)
        class_features = center + 0.00000 * torch.randn(samples_per_class, feature_dim)
        features.append(class_features)
        labels += [cls] * samples_per_class

    features = torch.stack(features).view(-1, feature_dim)  # [15, 128]
    features = F.normalize(features, p=2, dim=1)
    features = features.unsqueeze(1)  # [15, 1, 128] for compatibility

    labels = torch.tensor(labels)
    max_probs = torch.ones(features.size(0), 1)  # confident

    loss_fn = SoftSupConLoss()
    loss = loss_fn(features, max_probs, labels=labels)
    print("Low-loss batch:", loss.item())