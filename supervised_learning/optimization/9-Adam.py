#!/usr/bin/env python3
"""Task 9 Optimization"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm:
    """
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * grad**2

    v_correct = v_new / (1 - beta1**t)
    s_correct = s_new / (1 - beta2**t)

    new_var = var - alpha * v_correct / (np.sqrt(s_correct) + epsilon)

    return new_var, v_new, s_new
