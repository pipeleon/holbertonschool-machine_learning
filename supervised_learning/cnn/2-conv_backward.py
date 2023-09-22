#!/usr/bin/env python3
"""Task 0 Convolutional Neural Networks"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    """
    m = A_prev.shape[0]
    k1 = W.shape[0]
    k2 = W.shape[1]
    s1 = stride[0]
    s2 = stride[1]

    if type(padding) is str:
        if padding == 'same':
            p1 = int((A_prev.shape[1] * (s1 - 1) + k1 - s1) / 2)
            p2 = int((A_prev.shape[2] * (s2 - 1) + k2 - s2) / 2)
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

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for set in range(m):
        for j in range(dZ.shape[1]):
            for k in range(dZ.shape[2]):
                for i in range(W.shape[3]):
                    Z = dZ[set, j, k, i]
                    dA_prev[
                        set,
                        j*s1:j*s1+k1,
                        k*s2:k*s2+k2,
                        :] += Z * W[:, :, :, i]
                    dW[:, :, :, i] += A_prev[set,
                                             j*s1:j*s1+k1,
                                             k*s2:k*s2+k2, :] * Z

    return dA_prev, dW, db
