#!/usr/bin/env python3
"""Task 0 Error analylis"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    classes = one_hot.shape[1]
    lenY = one_hot.shape[0]
    matrix = np.zeros((lenY, ))

    for i in range(lenY):
        for j in range(classes):
            if one_hot[i][j]:
                matrix[i] = j

    return matrix.astype(int)


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix"""
    classes = labels.shape[1]
    m = labels.shape[0]

    labels_decode = one_hot_decode(labels)
    logits_decode = one_hot_decode(logits)

    matrix = np.zeros((classes, classes))

    for i in range(m):
        matrix[labels_decode[i], logits_decode[i]] += 1

    return matrix
