#!/usr/bin/env python3
"""Task 0 Optimization"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix"""
    m = X.shape[0]
    mean = np.sum(X, axis=0) / m
    standard_dev = (np.sum((X - mean)**2, axis=0) / m)**(1 / 2)

    return mean, standard_dev
