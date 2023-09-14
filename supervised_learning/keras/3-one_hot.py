#!/usr/bin/env python3
"""Task 3 Keras"""
import numpy as np


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""
    classes = np.max(labels) + 1
    matrix = np.zeros((classes, len(labels)))

    for i in range(len(labels)):
        if labels[i] > classes:
            return None
        matrix[i][labels[i]] = 1

    return matrix
