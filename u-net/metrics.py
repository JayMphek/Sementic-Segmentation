import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)

def compute_miou(predictions, targets, n_classes, ignore_index=-1):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    
    if predictions.ndim == 4:
        predictions = torch.argmax(predictions, dim=1)  # (N, H, W)

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    valid_mask = (targets != ignore_index)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    predictions = torch.clamp(predictions.long(), 0, n_classes - 1)
    targets = targets.long()
    
    iou_per_class = []
    for class_id in range(n_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask.sum() + target_mask.sum() - intersection).float()

        if target_mask.sum() == 0:
            continue
        
        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    if not iou_per_class:
        return 0.0

    miou = torch.stack(iou_per_class).mean()
    return miou.item()  