#!/usr/bin/env python3
"""Task 5 Convolutions and Pooling"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels"""
    m = images.shape[0]
    k1 = kernels.shape[0]
    k2 = kernels.shape[1]
    s1 = stride[0]
    s2 = stride[1]
    if type(padding) is str:
        if padding == 'same':
            p1 = int((images.shape[1] * (s1 - 1) + k1 - s1) / 2)
            p2 = int((images.shape[2] * (s2 - 1) + k2 - s2) / 2)
            if k1 % 2 == 0:
                p1 += 1
            if k2 % 2 == 0:
                p2 += 1
        else:
            p1 = 0
            p2 = 0
    else:
        p1 = padding[0]
        p2 = padding[1]

    new_dimX = int(1 + (images.shape[1] - k1 + 2 * p1) / s1)
    new_dimY = int(1 + (images.shape[2] - k2 + 2 * p2) / s2)

    images = np.pad(
        images, ((0, 0), (p1, p1), (p2, p2), (0, 0)), mode="constant")

    new_image = np.zeros((m, new_dimX, new_dimY, kernels.shape[3]))

    for j in range(new_image.shape[1]):
        for k in range(new_image.shape[2]):
            for i in range(kernels.shape[3]):
                new_value = np.sum(
                    images[:, j*s1:j*s1+k1, k*s2:k*s2+k2, :] *
                    kernels[i], axis=(1, 2, 3))
                new_image[:, j, k, i] = new_value

    return new_image
