import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalDiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Focal LossDice Loss
        
        :
            gamma (float): Focal lossgamma，
            alpha (float): Focal lossalpha，
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, input, target):
        sigmoid_input = torch.sigmoid(input)
        pt = target * sigmoid_input + (1 - target) * (1 - sigmoid_input)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha > 0:
            focal_weight = target * self.alpha * focal_weight + (1 - target) * (1 - self.alpha) * focal_weight
            
        focal_loss = F.binary_cross_entropy_with_logits(
            input, target, weight=focal_weight.detach(), reduction='mean'
        )
        
        smooth = 1e-5
        num = target.size(0)
        sigmoid_input = sigmoid_input.view(num, -1)
        target = target.view(num, -1)
        intersection = (sigmoid_input * target)
        dice = (2. * intersection.sum(1) + smooth) / (sigmoid_input.sum(1) + target.sum(1) + smooth)
        dice_loss = 1 - dice.sum() / num
        
        return 0.5 * focal_loss + 0.5 * dice_loss
