#!/usr/bin/env python3

cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

""" mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat4)
print(mat5)
mat1[0] = [9, 10]
mat1[1].append(5)
print(mat1)
print(mat4)
print(mat5) """
m1 = [[4, -7, 56, 2], [5, 106, 7, 2]]
m2 = [[2, -6, 3, 23], [0, -6, 3, 42], [73, 8, 2, 99]]
m = cat_matrices2D(m1, m2)
print(m)