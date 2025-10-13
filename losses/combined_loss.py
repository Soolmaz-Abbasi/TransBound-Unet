#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""


import torch
from boundary_loss import BoundaryAwareDiceLoss
from tversky_loss import TverskyLoss

class CombinedLoss(torch.nn.Module):
    def __init__(self, boundary_weight=0.5, sensitivity_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.sensitivity_weight = sensitivity_weight
        self.boundary_loss = BoundaryAwareDiceLoss()
        self.sensitivity_loss = TverskyLoss(alpha=0.7, beta=0.3)

    def forward(self, pred, target):
        boundary_loss = self.boundary_loss(pred, target)
        sensitivity_loss = self.sensitivity_loss(pred, target)
        return self.boundary_weight * boundary_loss + self.sensitivity_weight * sensitivity_loss
