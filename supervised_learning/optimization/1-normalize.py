#!/usr/bin/env python3
"""Task 1 Optimization"""


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix"""
    X = (X - m) / s

    return X
