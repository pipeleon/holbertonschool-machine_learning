#!/usr/bin/env python3
"""Task 0 Regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization"""
    sum_weights = 0

    for i in range(1, L + 1):
        sum_weights += np.linalg.norm(weights['W'+str(i)])

    new_cost = cost + lambtha * sum_weights / (2 * m)

    return new_cost
