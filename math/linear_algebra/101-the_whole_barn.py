#!/usr/bin/env python3
"""Task 100"""


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


def add_matrices(mat1, mat2):
    """Funtion to sum 2D arrays"""
    new_matrix = []

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None  

    if len(matrix_shape(mat1)) == 1:
        for i in range(matrix_shape(mat1)[0]):
            new_matrix.append(mat1[i] + mat2[i])
    elif len(matrix_shape(mat1)) == 2:
        for i in range(matrix_shape(mat1)[0]):
            temp = []
            for j in range(matrix_shape(mat1)[1]):
                temp.append(mat1[i][j] + mat2[i][j])
            new_matrix.append(temp)
    elif len(matrix_shape(mat1)) == 3:
        for i in range(matrix_shape(mat1)[0]):
            temp = []
            for j in range(matrix_shape(mat1)[1]):
                temp2 = []
                for k in range(matrix_shape(mat1)[2]):
                    temp2.append(mat1[i][j][k] + mat2[i][j][k])
                temp.append(temp2)
            new_matrix.append(temp)
    elif len(matrix_shape(mat1)) == 4:
        for i in range(matrix_shape(mat1)[0]):
            temp = []
            for j in range(matrix_shape(mat1)[1]):
                temp2 = []
                for k in range(matrix_shape(mat1)[2]):
                    temp3 = []
                    for i2 in range(matrix_shape(mat1)[3]):
                        temp3.append(mat1[i][j][k][i2] + mat2[i][j][k][i2])
                    temp2.append(temp3)
                temp.append(temp2)
            new_matrix.append(temp)

    return new_matrix
