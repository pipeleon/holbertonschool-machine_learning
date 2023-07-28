"""Task 2"""

def recursive_look(matrix, list_shape):
    """Auxiliar funtion"""
    if type(matrix) == list:
        list_shape.append(len(matrix))
        return recursive_look(matrix[0], list_shape)
    else:
        return list_shape

def matrix_shape(matrix):
    """Funtion"""
    shape = []

    return recursive_look(matrix, shape)
