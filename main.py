#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import MedicalDataset  
from Model.transbound_unet import TransBound_UNet 
from losses.boundary_loss import BoundaryAwareDiceLoss  
from train_utils import train_epoch, validate_epoch, test_model, plot_results

from monai.metrics import DiceMetric, HausdorffDistanceMetric 

def main():
    # --- Dataset ---
    image_dir = "AlineBlineSegmentation_Full/Images"
    mask_dir = "AlineBlineSegmentation_Full/AlineBlineSegmentation"

    transform = Compose([Resize((224, 224))])

    dataset = MedicalDataset(image_dir, mask_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


    train_dataset = [(img, mask) for img, mask in train_dataset]
    val_dataset = [(img, mask) for img, mask in val_dataset]
    test_dataset = [(img, mask) for img, mask in test_dataset]

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Model ---
    model = TransBound_UNet(img_size=(224, 224), in_channels=1, out_channels=1).to(device)

    # --- Loss & Optimizer ---
    criterion = BoundaryAwareDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # --- Metrics ---
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, percentile=95)

    # --- Training ---
    epochs = 50


    for epoch in range(epochs):
        train_loss, train_dice, train_hausdorff = train_epoch(
            model, train_loader, criterion, optimizer, device, dice_metric, hausdorff_metric
        )
        val_loss, val_dice, val_hausdorff = validate_epoch(
            model, val_loader, criterion, device, dice_metric, hausdorff_metric
        )

        print(f"Epoch {epoch+1}/{epochs}")
        print(f" Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train Hausdorff: {train_hausdorff:.4f}")
        print(f" Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val Hausdorff: {val_hausdorff:.4f}")



    # --- Test ---
    outputs, ground_truths = test_model(model, test_loader, device, dice_metric, hausdorff_metric, visualize=True)
    # plot_results(outputs, ground_truths, idx=0)

if __name__ == "__main__":
    main()
