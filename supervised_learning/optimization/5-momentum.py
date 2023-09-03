#!/usr/bin/env python3
"""Task 5 Optimization"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient
    descent with momentum optimization algorithm
    """
    momentum = beta1 * v + (1 - beta1) * grad
    new_var = var - alpha * momentum

    return new_var, momentum
