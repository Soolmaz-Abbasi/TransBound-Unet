#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""
import torch

def hausdorff_distance_loss(pred, target):
    pred_points = torch.nonzero(pred > 0.5, as_tuple=False).float()
    target_points = torch.nonzero(target > 0.5, as_tuple=False).float()

    if len(pred_points) == 0 or len(target_points) == 0:
        return torch.tensor(1.0, device=pred.device)

    distances_pred_to_target = torch.cdist(pred_points, target_points)
    min_distances_pred_to_target = distances_pred_to_target.min(dim=1)[0]

    distances_target_to_pred = torch.cdist(target_points, pred_points)
    min_distances_target_to_pred = distances_target_to_pred.min(dim=1)[0]

    hausdorff_distance = torch.max(min_distances_pred_to_target.max(), min_distances_target_to_pred.max())
    return hausdorff_distance


