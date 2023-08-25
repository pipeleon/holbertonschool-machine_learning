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


def mat_mul(mat1, mat2):
    """Funtion to multiplicate 2D arrays"""
    new_matrix = []

    if matrix_shape(mat1)[1] != matrix_shape(mat2)[0]:
        return None

    for i in range(matrix_shape(mat1)[0]):
        temp = []
        for j in range(matrix_shape(mat2)[1]):
            temp2 = 0
            for k in range(matrix_shape(mat2)[0]):
                temp2 += mat1[i][k] * mat2[k][j]
            temp.append(temp2)
        new_matrix.append(temp)

    return new_matrix
