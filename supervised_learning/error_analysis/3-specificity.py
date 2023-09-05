#!/usr/bin/env python3
"""Task 3 Error analylis"""
import numpy as np


def specificity(confusion):
    """Calculates the precision for each class in a confusion matrix"""
    classes = confusion.shape[0]
    specificity_matrix = np.zeros((classes, ))

    calculated_true = np.diagonal(confusion)
    actual_true = np.sum(confusion, axis=1)
    predicted_true = np.sum(confusion, axis=0)

    for i in range(classes):
        false_positive = predicted_true[i] - calculated_true[i]
        actual_false = np.sum(actual_true) - actual_true[i]
        specificity_matrix[i] = false_positive / actual_false

    return 1 - specificity_matrix
