#!/usr/bin/env python3
"""Task 12"""


def np_elementwise(mat1, mat2):
    """Funtion tha return a matrix Transpose"""
    return [
        mat1 + mat2,
        mat1 - mat2,
        mat1 * mat2,
        mat1 / mat2,
    ]
