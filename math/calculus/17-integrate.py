#!/usr/bin/env python3
"""Task 17"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""

    if type(poly) != list or len(poly) == 0 or type(C) != int:
        return None

    int_poly = [C]

    if poly == [0]:
        return int_poly

    for i in range(len(poly)):
        if poly[i] % (i + 1) == 0:
          int_poly.append(int(poly[i] / (i + 1)))
        else:
            int_poly.append((poly[i] / (i + 1)))

    return int_poly
