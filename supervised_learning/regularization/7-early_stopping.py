#!/usr/bin/env python3
"""Task 7 Regularization"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early
    """
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0

    if count == patience:
        return True, count
    else:
        return False, count
