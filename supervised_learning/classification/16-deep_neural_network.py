#!/usr/bin/env python3
"""Task 16 Classification"""
import numpy as np


class DeepNeuralNetwork ():
    """Deep Neural Network performing binary classification"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        tmp_w = {}
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            ly = layers
            if i == 0:
                st = np.sqrt(2/nx)
                tmp_w['W' + str(i+1)] = np.random.normal(0, st, (ly[i], nx))
            else:
                st = np.sqrt(2/layers[i - 1])
                tmp_w['W' + str(i+1)] = np.random.normal(0, st, (ly[i], ly[i-1]))
            tmp_w['b' + str(i+1)] = np.zeros((layers[i], 1))

        self.L = len(layers)
        self.cache = {}
        self.weights = tmp_w
