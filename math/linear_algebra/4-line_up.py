#!/usr/bin/env python3
"""Task 4"""


def recursive_look(matrix, list_shape):
    """Auxiliar recursive funtion"""
    if type(matrix) == list:
        list_shape.append(len(matrix))
        return recursive_look(matrix[0], list_shape)
    else:
        return list_shape


def matrix_shape(matrix):
    """Funtion to get the shape of a matrix"""
    shape = []

    return recursive_look(matrix, shape)


def add_arrays(arr1, arr2):
    """Funtion to sum arrays"""
    new_array = []

    if matrix_shape(arr1) != matrix_shape(arr2):
        return None

    for i in range(matrix_shape(arr1)[0]):
        new_array.append(arr1[i] + arr2[i])

    return new_array
