#!/usr/bin/env python3
"""Task 4 Error analylis"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score of a confusion matrix"""
    s = sensitivity(confusion)
    p = precision(confusion)

    return 2 * s * p / (s + p)
