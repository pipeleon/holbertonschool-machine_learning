#!/usr/bin/env python3
"""Task 2 Error analylis"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix"""
    calculated_true = np.diagonal(confusion)
    predicted_true = np.sum(confusion, axis=0)

    return calculated_true / predicted_true
