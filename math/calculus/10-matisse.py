#!/usr/bin/env python3
"""Task 10"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial"""

    if type(poly) != list:
        return None

    if len(poly) == 1:
        return [0]

    dev_poly = []

    for i in range(len(poly) - 1):
        dev_poly.append(poly[i + 1] * (i + 1))

    return dev_poly
