import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)

    def forward (self, logits, targets, reduction='none'):
        """
        focal loss for multi-classification only for one-hot target
    
        Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
        """
        # if targets is not one-hot, convert to one-hot
        if len(targets.size()) != 1:
            targets = torch.eye(logits.size(1))[targets]
        targets = targets.view(-1,1)
        # for one-hot target
        if logits.size(1) < 3:
            targets = targets.view(-1)
            logpt = -F.binary_cross_entropy_with_logits(logits.sum(dim=1), targets.float(), reduction='none')
            pt = torch.exp(logpt)
        else:
            logpt = F.log_softmax(logits, dim=1)
            logpt = logpt.gather(1,targets)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())     #   e^(logpt)
            if self.alpha is not None:
                if self.alpha.type()!=logits.data.type():
                    self.alpha = self.alpha.type_as(logits.data)
                at = self.alpha.gather(0, targets)
                logpt = logpt * Variable(at)
        loss = -1 * (1-pt)**self.gamma * logpt
        if reduction == 'none':
            return loss
        else:
            return loss.mean()