#!/usr/bin/env python3
"""Task 13"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Funtion tha return a matrix Transpose"""
    return np.concatenate((mat1, mat2), axis)
