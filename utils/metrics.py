import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

def calculate_metrics(pred_logits, targets, smooth=1e-6):
    preds = (torch.sigmoid(pred_logits) > 0.5).float().view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()
    
    dice = (2. * intersection + smooth) / (total + smooth)
    iou = (intersection + smooth) / (total - intersection + smooth)
    return dice.item(), iou.item()