#!/usr/bin/env python3
"""Task 0 Convolutional Neural Networks"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
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

    new_dimX = int(1 + (A_prev.shape[1] - k1 + 2 * p1) / s1)
    new_dimY = int(1 + (A_prev.shape[2] - k2 + 2 * p2) / s2)

    A_prev = np.pad(
        A_prev, ((0, 0), (p1, p1), (p2, p2), (0, 0)), mode="constant")

    Z = np.zeros((m, new_dimX, new_dimY, W.shape[3]))

    for j in range(Z.shape[1]):
        for k in range(Z.shape[2]):
            for i in range(W.shape[3]):
                print(A_prev.shape)
                print(W.shape)
                print(Z.shape)
                print("************")
                print(A_prev[:, j*s1:j*s1+k1, k*s2:k*s2+k2, :].shape)
                print(W[:, :, :, i].shape)
                print(Z[:, j, k, i].shape)
                new_value = np.sum(
                    A_prev[:, j*s1:j*s1+k1, k*s2:k*s2+k2, :] *
                    W[:, :, :, i], axis=(1, 2, 3))
                Z[:, j, k, i] = new_value

    Z += b

    A = activation(Z)

    return A
