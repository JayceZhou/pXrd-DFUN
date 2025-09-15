import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.1, reduction='mean'):

        super(FocalLossWithLabelSmoothing, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            label_smoothing=self.label_smoothing, 
            reduction='none'
        )
        
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss.unsqueeze(1)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss