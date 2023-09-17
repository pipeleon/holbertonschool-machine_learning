#!/usr/bin/env python3
"""Task 0 Convolutions and Pooling"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images"""
    m = images.shape[0]
    k1 = kernel.shape[0]
    k2 = kernel.shape[1]
    new_dimX = images.shape[1] - k1 + 1
    new_dimY = images.shape[2] - k2 + 1

    new_image = np.zeros((m, new_dimX, new_dimY))

    for i in range(2):
        for j in range(new_image.shape[1]):
            for k in range(new_image.shape[2]):
                new_value = np.sum(images[i, j:j+k1, k:k+k2] * kernel)
                new_image[i, j, k] = new_value

    return new_image
