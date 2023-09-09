#!/usr/bin/env python3
"""Task 4 Regularization"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X

    for i in range(L):
        if i == 0:
            z = np.matmul(weights['W1'], X) + weights['b1']
        else:
            W = weights['W' + str(i+1)]
            A = cache['A' + str(i)]
            z = np.matmul(W, A) + weights['b' + str(i+1)]

        if i + 1 == L:
            t = np.exp(z)
            suma_t = np.sum(t, axis=0, keepdims=True)
            cache['A' + str(i+1)] = t / suma_t
        else:
            cache['A' + str(i+1)] = np.tanh(z)

        if i != (L - 1):
            A = cache['A' + str(i+1)]
            d = np.random.rand(A.shape[0], A.shape[1])
            cache['D' + str(i+1)] = np.where(d < keep_prob, 1, 0)
            cache['A' + str(i+1)] = np.multiply(A, cache['D' + str(i+1)])
            cache['A' + str(i+1)] /= keep_prob

    return cache
