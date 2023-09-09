#!/usr/bin/env python3
"""Task 1 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network
    using gradient descent with L2 regularization
    """
    m = Y.shape[1]
    limit = L
    w_aux = []

    while limit > 0:
        A = cache['A' + str(limit)]
        if limit == L:
            dz = A - Y
        else:
            dz = np.matmul(w_aux.T, dz) * (1 - A**2)
        l2_reg_w = lambtha * weights['W' + str(limit)]
        dw = (1 / m) * (np.matmul(dz, cache['A' + str(limit-1)].T) + l2_reg_w)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        w_aux = weights["W" + str(limit)].copy()
        weights["W" + str(limit)] -= alpha * dw
        weights["b" + str(limit)] -= alpha * db
        limit -= 1
