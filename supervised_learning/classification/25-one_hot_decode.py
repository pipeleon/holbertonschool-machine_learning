#!/usr/bin/env python3
"""Task 25 Classification"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray:
        return None
    classes = one_hot.shape[0]
    lenY = one_hot.shape[1]
    matrix = np.zeros((lenY, ))

    for i in range(lenY):
        for j in range(classes):
            if one_hot[j][i]:
                matrix[i] = j

    return matrix.astype(int)
