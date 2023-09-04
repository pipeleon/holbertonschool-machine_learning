#!/usr/bin/env python3
"""Task 13 Optimization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a
    neural network using batch normalization
    """
    m = Z.shape[0]
    mean = np.sum(Z, axis=0) / m
    variance = (np.sum((Z - mean)**2, axis=0) / m)

    Z_norm = (Z - mean) / (variance + epsilon)**(1 / 2)

    return gamma * Z_norm + beta
