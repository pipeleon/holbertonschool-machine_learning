#!/usr/bin/env python3
"""Task 12"""


def np_elementwise(mat1, mat2):
    """Funtion tha return a matrix Transpose"""
    return {
        'add': mat1 + mat2,
        'sub': mat1 - mat2,
        'mul': mat1 * mat2,
        'div': mat1 / mat2,
    }
