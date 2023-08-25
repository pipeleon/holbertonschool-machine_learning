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


def cat_matrices2D(mat1, mat2, axis=0):
    """Funtion to concat 2D arrays"""
    new_matrix = []
    for row in mat1:
        new_matrix.append(row.copy())

    if axis == 0:
        if matrix_shape(mat1)[1] != matrix_shape(mat2)[1]:
            return None

        for i in range(len(mat2)):
            new_matrix.append(mat2[i])
    elif axis == 1:
        if matrix_shape(mat1)[0] != matrix_shape(mat2)[0]:
            return None

        for i in range(matrix_shape(mat2)[0]):
            for j in range(matrix_shape(mat2)[1]):
                new_matrix[i].append(mat2[i][j])

    return new_matrix
