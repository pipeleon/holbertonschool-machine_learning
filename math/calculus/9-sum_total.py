#!/usr/bin/env python3
"""Task 9"""


def summation_i_squared(n, sum=0):
    """Function that calculates the sume of n sqare numbers"""

    if type(n) is not int:
        return None

    if n == 0:
        return sum
    else:
        sum += n * n
        return summation_i_squared(n - 1, sum)
