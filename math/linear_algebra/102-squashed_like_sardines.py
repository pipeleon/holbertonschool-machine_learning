#!/usr/bin/env python3
"""Task 102"""


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


def m_sh(matrix):
    """Funtion to get the shape of a matrix"""
    shape = []

    return recursive_look(matrix, shape)


def cat_matrices(mat1, mat2, axis=0):
    """Funtion to concat arrays"""

    if len(m_sh(mat1)) == 1 and len(m_sh(mat2)) == 1:
        new_matrix = mat1.copy()

        for i in range(m_sh(mat2)[0]):
            new_matrix.append(mat2[i])
    elif len(m_sh(mat1)) == 2 and len(m_sh(mat2)) == 2:
        new_matrix = []
        for row in mat1:
            new_matrix.append(row.copy())

        if axis == 0:
            if m_sh(mat1)[1] != m_sh(mat2)[1]:
                return None

            for i in range(len(mat2)):
                new_matrix.append(mat2[i])
        elif axis == 1:
            if m_sh(mat1)[0] != m_sh(mat2)[0]:
                return None

            for i in range(m_sh(mat2)[0]):
                for j in range(m_sh(mat2)[1]):
                    new_matrix[i].append(mat2[i][j])
    elif len(m_sh(mat1)) == 3 and len(m_sh(mat2)) == 3:
        new_matrix = []
        for ax0 in mat1:
            temp = []
            for ax1 in ax0:
                temp.append(ax1.copy())
            new_matrix.append(temp)

        if axis == 0:
            if m_sh(mat1)[1] != m_sh(mat2)[1] and m_sh(mat1)[2] != m_sh(mat2)[2]:
                return None

            for i in range(len(mat2)):
                new_matrix.append(mat2[i])
        elif axis == 1:
            if m_sh(mat1)[0] != m_sh(mat2)[0] and m_sh(mat1)[2] != m_sh(mat2)[2]:
                return None

            for i in range(m_sh(mat2)[0]):
                for j in range(m_sh(mat2)[1]):
                    new_matrix[i].append(mat2[i][j])
        elif axis == 2:
            if m_sh(mat1)[0] != m_sh(mat2)[0] and m_sh(mat1)[1] != m_sh(mat2)[1]:
                return None

            for i in range(m_sh(mat2)[0]):
                for j in range(m_sh(mat2)[1]):
                    for k in range(m_sh(mat2)[2]):
                        new_matrix[i][j].append(mat2[i][j][k])
    elif len(m_sh(mat1)) == 4 and len(m_sh(mat2)) == 4:
        new_matrix = []
        for ax0 in mat1:
            temp = []
            for ax1 in ax0:
                temp2 = []
                for ax2 in ax1:
                    temp2.append(ax2.copy())
                temp.append(temp2)
            new_matrix.append(temp)

        if axis == 0:
            if m_sh(mat1)[1] != m_sh(mat2)[1] and m_sh(mat1)[2] != m_sh(mat2)[2] and m_sh(mat1)[3] != m_sh(mat2)[3]:
                return None

            for i in range(len(mat2)):
                new_matrix.append(mat2[i])
        elif axis == 1:
            if m_sh(mat1)[0] != m_sh(mat2)[0] and m_sh(mat1)[2] != m_sh(mat2)[2] and m_sh(mat1)[3] != m_sh(mat2)[3]:
                return None

            for i in range(m_sh(mat2)[0]):
                for j in range(m_sh(mat2)[1]):
                    new_matrix[i].append(mat2[i][j])
        elif axis == 2:
            if m_sh(mat1)[0] != m_sh(mat2)[0] and m_sh(mat1)[1] != m_sh(mat2)[1] and m_sh(mat1)[3] != m_sh(mat2)[3]:
                return None

            for i in range(m_sh(mat2)[0]):
                for j in range(m_sh(mat2)[1]):
                    for k in range(m_sh(mat2)[2]):
                        new_matrix[i][j].append(mat2[i][j][k])
        elif axis == 3:
            if m_sh(mat1)[0] != m_sh(mat2)[0] and m_sh(mat1)[1] != m_sh(mat2)[1] and m_sh(mat1)[2] != m_sh(mat2)[2]:
                return None

            for i in range(m_sh(mat2)[0]):
                for j in range(m_sh(mat2)[1]):
                    for k in range(m_sh(mat2)[2]):
                        for i2 in range(m_sh(mat2)[3]):
                            new_matrix[i][j][k].append(mat2[i][j][k][i2])
    else:
        return None

    return new_matrix
