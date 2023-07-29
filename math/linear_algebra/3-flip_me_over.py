#!/usr/bin/env python3
"""Task 3"""


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


def matrix_transpose(matrix):
    """Funtion to transpose a 2D matrix"""
    shape_matrix = matrix_shape(matrix)
    new_matrix = []

    for axis1 in range(shape_matrix[1]):
        temp = []
        for axis2 in range(shape_matrix[0]):
            temp.append(0)
        new_matrix.append(temp)

    for i in range(shape_matrix[0]):
        for j in range(shape_matrix[1]):
            new_matrix[j][i] = matrix[i][j]

    return new_matrix
