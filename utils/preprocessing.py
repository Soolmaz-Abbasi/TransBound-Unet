#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""

import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import gaussian_filter

def preprocess_ultrasound(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses and enhances an ultrasound image.
    """
    if len(image.shape) == 3:  # RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))  # Adjusted for newer versions
    denoised = denoise_nl_means(
        image,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=3,
        channel_axis=None,  
    )
    denoised = (denoised * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(denoised)


    blurred = gaussian_filter(enhanced_contrast, sigma=1)
    unsharp_masked = cv2.addWeighted(enhanced_contrast, 1.5, blurred, -0.5, 0)


    normalized = unsharp_masked / 255.0

    return normalized


if __name__ == "__main__":
    import os
    from PIL import Image
    img_path = os.path.join("..", "data", "images", "sample_image.png")
    image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
    processed = preprocess_ultrasound(image)
    print("Processed image shape:", processed.shape)
