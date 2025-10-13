#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""

import torch
import torch.nn.functional as F
from skimage.segmentation import find_boundaries

class BoundaryAwareDiceLoss(torch.nn.Module):
    def __init__(self, weight=10.0):
        super(BoundaryAwareDiceLoss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

        target_np = target.cpu().numpy().astype(bool)
        boundary_mask = torch.from_numpy(find_boundaries(target_np, mode='inner')).float().to(pred.device)
        boundary_loss = F.binary_cross_entropy(pred, target) * boundary_mask

        return dice_loss + self.weight * boundary_loss.mean()
