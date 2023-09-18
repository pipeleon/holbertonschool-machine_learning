#!/usr/bin/env python3
"""Task 6 Convolutions and Pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs a convolution on images with channels"""
    m = images.shape[0]
    k1 = kernel_shape[0]
    k2 = kernel_shape[1]
    s1 = stride[0]
    s2 = stride[1]

    new_dimX = int(1 + (images.shape[1] - k1) / s1)
    new_dimY = int(1 + (images.shape[2] - k2) / s2)

    new_image = np.zeros((m, new_dimX, new_dimY, images.shape[3]))

    for j in range(new_image.shape[1]):
        for k in range(new_image.shape[2]):
            if mode == "max":
                new_value = np.max(
                    images[:, j*s1:j*s1+k1, k*s2:k*s2+k2, :], axis=(1, 2))
            else:
                new_value = np.average(
                    images[:, j*s1:j*s1+k1, k*s2:k*s2+k2, :], axis=(1, 2))
            new_image[:, j, k, :] = new_value

    return new_image
