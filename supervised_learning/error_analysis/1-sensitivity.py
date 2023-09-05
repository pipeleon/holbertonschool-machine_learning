#!/usr/bin/env python3
"""Task 1 Error analylis"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix"""
    calculated_true = np.diagonal(confusion)
    actual_true = np.sum(confusion, axis=1)

    return calculated_true / actual_true
