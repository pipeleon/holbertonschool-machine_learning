#!/usr/bin/env python3
"""Task 6 Optimization"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm
    """
    new_s = beta2 * s + (1 - beta2) * grad**2
    new_var = var - alpha * grad / (np.sqrt(new_s) + epsilon)

    return new_var, new_s
