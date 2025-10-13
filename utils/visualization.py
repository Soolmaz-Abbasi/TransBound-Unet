#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: soolmaz
"""

import numpy as np
import cv2
import cv2
import numpy as np
import torch

def detect_and_visualize_lines(image):

    gray = image.detach().cpu().numpy()
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    dilation_kernel_size = (7, 7)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones(dilation_kernel_size, np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    a_line_count = 0
    b_line_count = 0

    color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w > h * 1.5:
            a_line_count += 1
            cv2.drawContours(color_image, [contour], -1, (255, 0, 0), 2)  # blue line

        elif w * 1.5 < h:
            b_line_count += 1
            cv2.drawContours(color_image, [contour], -1, (0, 0, 255), 2)  # red line

    return a_line_count, b_line_count, color_image

