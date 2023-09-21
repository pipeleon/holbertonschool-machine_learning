#!/usr/bin/env python3
"""Task 6 Convolutions and Pooling"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs a convolution on A_prev with channels"""
    m = A_prev.shape[0]
    k1 = kernel_shape[0]
    k2 = kernel_shape[1]
    s1 = stride[0]
    s2 = stride[1]

    new_dimX = int(1 + (A_prev.shape[1] - k1) / s1)
    new_dimY = int(1 + (A_prev.shape[2] - k2) / s2)

    A = np.zeros((m, new_dimX, new_dimY, A_prev.shape[3]))

    for j in range(A.shape[1]):
        for k in range(A.shape[2]):
            if mode == "max":
                new_value = np.max(
                    A_prev[:, j*s1:j*s1+k1, k*s2:k*s2+k2, :], axis=(1, 2))
            else:
                new_value = np.average(
                    A_prev[:, j*s1:j*s1+k1, k*s2:k*s2+k2, :], axis=(1, 2))
            A[:, j, k, :] = new_value

    return A
