#!/usr/bin/env python3
"""Task 24 Classification"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    matrix = np.zeros((classes, len(Y)))

    for i in range(len(Y)):
        if Y[i] > classes:
            return None
        matrix[Y[i]][i] = 1

    return matrix
