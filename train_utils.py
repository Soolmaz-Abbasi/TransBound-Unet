#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.visualization import detect_and_visualize_lines

def train_epoch(model, dataloader, criterion, optimizer, device, metric, hausdorff_metric):
    model.train()
    epoch_loss = 0
    metric.reset()
    
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        masks = (masks > 0).float()

        outputs = torch.sigmoid(model(images))
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        binary_outputs = (outputs > 0.5).float()
        metric(y_pred=binary_outputs, y=masks)
        hausdorff_metric(y_pred=binary_outputs, y=masks)

    mean_dice = metric.aggregate()[0].item()
    hausdorff = hausdorff_metric.aggregate()[0].item()
    return epoch_loss / len(dataloader), mean_dice, hausdorff


def validate_epoch(model, dataloader, criterion, device, metric, hausdorff_metric):
    model.eval()
    epoch_loss = 0
    metric.reset()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            masks = (masks > 0).float()

            outputs = torch.sigmoid(model(images))
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            binary_outputs = (outputs > 0.5).float()
            
            # Optional dilation
            kernel = torch.ones((1, 1, 3, 3), device=binary_outputs.device)
            dilated_output = F.conv2d(binary_outputs, kernel, padding=1)
            binary_outputs = (dilated_output > 0).float()

            metric(y_pred=binary_outputs, y=masks)
            hausdorff_metric(y_pred=binary_outputs, y=masks)

    mean_dice = metric.aggregate()[0].item()
    hausdorff = hausdorff_metric.aggregate()[0].item()
    return epoch_loss / len(dataloader), mean_dice, hausdorff


def test_model(model, dataloader, device, metric, hausdorff_metric, visualize):
    all_outputs = []
    all_ground_truths = []
    total_TP = total_TN = total_FP = total_FN = 0
    model.eval()
    metric.reset()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            masks = (masks > 0).float()

            outputs = torch.sigmoid(model(images))

            binary_outputs = (outputs > 0.5).float()

            metric(y_pred=binary_outputs, y=masks)
            hausdorff_metric(y_pred=binary_outputs, y=masks)

            all_outputs.append(outputs.cpu())
            all_ground_truths.append(masks.cpu())
            if visualize:

                for i in range(3):
                    img = images[i].cpu().squeeze(0) 
                    img = img.cpu().squeeze(0)
                    mask = masks[i].cpu().squeeze(0)  
                    pred = (outputs[i].cpu().squeeze(0) > 0.5).float()

                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                    axs[0].imshow(img, cmap='gray')
                    axs[0].set_title("Original Image")
                    axs[0].axis("off")

                    axs[1].imshow(mask, cmap='gray')
                    axs[1].set_title("Ground Truth Segmentation")
                    axs[1].axis("off")

                    axs[2].imshow(pred, cmap='gray')
                    axs[2].set_title("Predicted Segmentation")
                    axs[2].axis("off")

                    a_lines, b_lines, result_image = detect_and_visualize_lines(pred)

                    if isinstance(result_image, torch.Tensor):
                        result_image = result_image.detach().cpu().numpy()

                    if result_image.dtype != np.uint8:
                        result_image = (result_image * 255).astype(np.uint8)

                    if result_image.ndim == 3 and result_image.shape[2] == 3:
                        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                    axs[3].imshow(result_image, cmap='gray')
                    axs[3].set_title(f"Output (A-lines: {a_lines}, B-lines: {b_lines})")
                    axs[3].axis('off')

                    plt.show()

                visualize = True
            metric(y_pred=binary_outputs, y=masks)
            hausdorff_metric(y_pred=binary_outputs, y=masks)

            TP = ((binary_outputs == 1) & (masks == 1)).sum().item()
            TN = ((binary_outputs == 0) & (masks == 0)).sum().item()
            FP = ((binary_outputs == 1) & (masks == 0)).sum().item()
            FN = ((binary_outputs == 0) & (masks == 1)).sum().item()

            total_TP += TP
            total_TN += TN
            total_FP += FP
            total_FN += FN

            metric(y_pred=binary_outputs, y=masks)
            hausdorff_metric(y_pred=binary_outputs, y=masks)

            # plot_batch(images, masks, binary_outputs)
    sensitivity = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    specificity = total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0
    mean_dice = metric.aggregate()[0].item()  # Get the mean Dice score from the tuple
    hausdorff = hausdorff_metric.aggregate()[0].item()
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    print(f"Test Dice Score: {mean_dice}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(hausdorff)
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1_score}")
    print(f"IoU: {iou}")
    return all_outputs, all_ground_truths 




def plot_results(predictions, ground_truths, idx=5):
    pred = predictions[idx].cpu().numpy()
    binary_pred = (pred > 0).astype(np.uint8)

    true = ground_truths[idx].cpu().numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(true[0], cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binary_pred[0], cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.show()