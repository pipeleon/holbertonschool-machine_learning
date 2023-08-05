#!/usr/bin/env python3
"""Task 9"""


def summation_i_squared(n, sum=0):
    """Function that calculates the sume of n sqare numbers"""

    if type(n) is not int:
        return None
    else:
        return int(n**3/3 + n**2/2 + n/6)
