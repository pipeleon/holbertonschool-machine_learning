#!/usr/bin/env python3
"""Task 2 Optimization"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way"""
    divisor = X.shape[1]
    temp_concat = np.concatenate((X, Y), axis=1)
    shuffle = np.random.permutation(temp_concat)

    X_shuffled = shuffle[:, :divisor]
    Y_shuffled = shuffle[:, divisor:]

    return X_shuffled, Y_shuffled
