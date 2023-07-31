#!/usr/bin/env python3
"""Task 100"""


def np_slice(matrix, axes={}):
    """Funtion to slice arrays"""
    shape = matrix.shape

    new_matrix = []

    if len(shape) >= 3:
        new_matrix = matrix[:,:,:]
        slice2 = axes.get(2)
        if slice2:
            if len(slice2) == 1:
                new_matrix = new_matrix[:, :, :slice2[0]]
            elif len(slice2) == 2:
                new_matrix = new_matrix[:, :, slice2[0]:slice2[1]]
            elif len(slice2) == 3:
                new_matrix = new_matrix[:, :, slice2[0]:slice2[1]:slice2[2]]
    if len(shape) >= 2:
        if len(new_matrix) == 0: new_matrix = matrix[:,:]
        slice1 = axes.get(1)
        if slice1:
            if len(slice1) == 1:
                new_matrix = new_matrix[:, :slice1[0]]
            elif len(slice1) == 2:
                new_matrix = new_matrix[:, slice1[0]:slice1[1]]
            elif len(slice1) == 3:
                new_matrix = new_matrix[:, slice1[0]:slice1[1]:slice1[2]]
    if len(shape) >= 1:
        if len(new_matrix) == 0: new_matrix = matrix[:]
        slice0 = axes.get(0)
        if slice0:
            if len(slice0) == 1:
                new_matrix = new_matrix[:slice0[0]]
            elif len(slice0) == 2:
                new_matrix = new_matrix[slice0[0]:slice0[1]]
            elif len(slice0) == 3:
                new_matrix = new_matrix[slice0[0]:slice0[1]:slice0[2]]
    
    return new_matrix
    