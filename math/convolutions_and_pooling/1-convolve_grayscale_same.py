#!/usr/bin/env python3
"""Task 1 Convolutions and Pooling"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m = images.shape[0]
    k1 = kernel.shape[0]
    k2 = kernel.shape[1]
    p1 = int((k1 - 1) / 2)
    p2 = int((k2 - 1) / 2)
    if k1 % 2 == 0:
        p1 += 1
    if k2 % 2 == 0:
        p2 += 1
    new_dimX = images.shape[1]
    new_dimY = images.shape[2]

    images = np.pad(images, ((0, 0), (p1, p1), (p2, p2)), mode="constant")

    new_image = np.zeros((m, new_dimX, new_dimY))

    for j in range(new_image.shape[1]):
        for k in range(new_image.shape[2]):
            new_value = np.sum(images[:, j:j+k1, k:k+k2] * kernel, axis=(1, 2))
            new_image[:, j, k] = new_value

    return new_image
