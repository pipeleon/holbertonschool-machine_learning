#!/usr/bin/env python3
"""Task 4"""


def recursive_look(matrix, list_shape):
    """Auxiliar recursive funtion"""
    if type(matrix) == list:
        list_shape.append(len(matrix))
        if len(matrix) > 0:
            return recursive_look(matrix[0], list_shape)
        else:
            return list_shape
    else:
        return list_shape


def matrix_shape(matrix):
    """Funtion to get the shape of a matrix"""
    shape = []

    return recursive_look(matrix, shape)


def cat_arrays(arr1, arr2):
    """Funtion to sum arrays"""
    new_array = arr1.copy()

    if len(arr1) == 0 or len(arr2) == 0:
        return new_array

    for i in range(matrix_shape(arr2)[0]):
        new_array.append(arr2[i])

    return new_array
