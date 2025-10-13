#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils.preprocessing import preprocess_ultrasound  

class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        image = preprocess_ultrasound(image)

        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

if __name__ == "__main__":
    dataset = MedicalDataset(
        image_dir=os.path.join("..", "data", "images"),
        mask_dir=os.path.join("..", "data", "masks")
    )
    image, mask = dataset[0]
    print("Image shape:", image.shape, "Mask shape:", mask.shape)
