#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""
import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        TP = (y_true_flat * y_pred_flat).sum()
        FP = ((1 - y_true_flat) * y_pred_flat).sum()
        FN = (y_true_flat * (1 - y_pred_flat)).sum()
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky_index

