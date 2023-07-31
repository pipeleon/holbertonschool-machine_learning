#!/usr/bin/env python3
"""Task 100"""


def np_slice(matrix, axes={}):
    """Funtion to slice arrays"""
    shape = matrix.shape

    new_matrix = []

    if len(shape) >= 6:
        new_matrix = matrix[:, :, :, :, :, :]
        sl5 = axes.get(5)
        if sl5:
            if len(sl5) == 1:
                new_matrix = new_matrix[:, :, :, :, :, :sl5[0]]
            elif len(sl5) == 2:
                new_matrix = new_matrix[:, :, :, :, :, sl5[0]:sl5[1]]
            elif len(sl5) == 3:
                new_matrix = new_matrix[:, :, :, :, :, sl5[0]:sl5[1]:sl5[2]]
    if len(shape) >= 5:
        if len(new_matrix) == 0:
            new_matrix = matrix[:, :, :, :, :]
        sl4 = axes.get(4)
        if sl4:
            if len(sl4) == 1:
                new_matrix = new_matrix[:, :, :, :, :sl4[0]]
            elif len(sl4) == 2:
                new_matrix = new_matrix[:, :, :, :, sl4[0]:sl4[1]]
            elif len(sl4) == 3:
                new_matrix = new_matrix[:, :, :, :, sl4[0]:sl4[1]:sl4[2]]
    if len(shape) >= 4:
        if len(new_matrix) == 0:
            new_matrix = matrix[:, :, :, :]
        slice3 = axes.get(3)
        if slice3:
            if len(slice3) == 1:
                new_matrix = new_matrix[:, :, :, :slice3[0]]
            elif len(slice3) == 2:
                new_matrix = new_matrix[:, :, :, slice3[0]:slice3[1]]
            elif len(slice3) == 3:
                new_matrix = new_matrix[:, :, :, slice3[0]:slice3[1]:slice3[2]]
    if len(shape) >= 3:
        if len(new_matrix) == 0:
            new_matrix = matrix[:, :, :]
        slice2 = axes.get(2)
        if slice2:
            if len(slice2) == 1:
                new_matrix = new_matrix[:, :, :slice2[0]]
            elif len(slice2) == 2:
                new_matrix = new_matrix[:, :, slice2[0]:slice2[1]]
            elif len(slice2) == 3:
                new_matrix = new_matrix[:, :, slice2[0]:slice2[1]:slice2[2]]
    if len(shape) >= 2:
        if len(new_matrix) == 0:
            new_matrix = matrix[:, :]
        slice1 = axes.get(1)
        if slice1:
            if len(slice1) == 1:
                new_matrix = new_matrix[:, :slice1[0]]
            elif len(slice1) == 2:
                new_matrix = new_matrix[:, slice1[0]:slice1[1]]
            elif len(slice1) == 3:
                new_matrix = new_matrix[:, slice1[0]:slice1[1]:slice1[2]]
    if len(shape) >= 1:
        if len(new_matrix) == 0:
            new_matrix = matrix[:]
        slice0 = axes.get(0)
        if slice0:
            if len(slice0) == 1:
                new_matrix = new_matrix[:slice0[0]]
            elif len(slice0) == 2:
                new_matrix = new_matrix[slice0[0]:slice0[1]]
            elif len(slice0) == 3:
                new_matrix = new_matrix[slice0[0]:slice0[1]:slice0[2]]

    return new_matrix
