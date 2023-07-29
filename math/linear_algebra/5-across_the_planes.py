#!/usr/bin/env python3
"""Task 4"""


def recursive_look(matrix, list_shape):
    """Auxiliar recursive funtion"""
    if type(matrix) == list:
        list_shape.append(len(matrix))
        if matrix[0]:
            return recursive_look(matrix[0], list_shape)
        else:
            return list_shape
    else:
        return list_shape


def matrix_shape(matrix):
    """Funtion to get the shape of a matrix"""
    shape = []

    return recursive_look(matrix, shape)


def add_matrices2D(mat1, mat2):
    """Funtion to sum 2D arrays"""
    new_matrix = []

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    for i in range(matrix_shape(mat1)[0]):
        temp = []
        for j in range(matrix_shape(mat1)[1]):
            temp.append(mat1[i][j] + mat2[i][j])
        new_matrix.append(temp)

    return new_matrix
