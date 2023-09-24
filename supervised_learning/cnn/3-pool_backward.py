#!/usr/bin/env python3
"""Task 3 Convolutional Neural Networks"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural network"""
    m = A_prev.shape[0]
    k1 = kernel_shape[0]
    k2 = kernel_shape[1]
    s1 = stride[0]
    s2 = stride[1]

    dA_prev = np.zeros(A_prev.shape)

    for set in range(m):
        for j in range(dA.shape[1]):
            for k in range(dA.shape[2]):
                for c in range(dA.shape[3]):
                    if mode == "max":
                        a_slice = A_prev[set, k*s2:k*s2+k2, j*s1:j*s1+k1, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[set, j*s1:j*s1+k1,
                                k*s2:k*s2+k2, c] += (mask * dA[set, j, k, c])
                    else:
                        ones = np.ones((k1, k2))
                        A = dA[set, j, k, c] / (k1 * k2)
                        dA_prev[set, j*s1:j*s1+k1, k*s2:k*s2+k2, c] += ones * A

    return dA_prev[0]
