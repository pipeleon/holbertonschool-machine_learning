#!/usr/bin/env python3
"""Task 4 Optimization"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set"""
    new_data = []
    v = [0]

    for i in range(len(data)):
        new_v = beta * v[i] + (1 - beta) * data[i]
        v.append(new_v)
        new_data.append(new_v / (1 - beta**(i + 1)))

    return new_data
