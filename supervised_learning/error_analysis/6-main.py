#!/usr/bin/env python3

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision
f1_score = __import__('4-f1_score').f1_score

if __name__ == '__main__':
    train_matrix = np.matrix([[54, 2, 4], [5, 52, 3], [15, 4, 41]])
    dev_matrix = np.matrix([[53, 1, 6], [7, 51, 2], [15, 5, 40]])

    print((train_matrix))
    print((dev_matrix))
    print(precision(train_matrix/1))
    print(precision(dev_matrix/1))