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
                key = 'W' + str(i+1)
                tmp_w[key] = np.random.normal(0, st, (ly[i], ly[i-1]))
            tmp_w['b' + str(i+1)] = np.zeros((layers[i], 1))

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = tmp_w

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X

        for i in range(self.__L):
            if i == 0:
                z = np.matmul(self.__weights['W1'], X) + self.__weights['b1']
            else:
                W = self.__weights['W' + str(i+1)]
                A = self.__cache['A' + str(i)]
                z = np.matmul(W, A) + self.__weights['b' + str(i+1)]

            self.__cache['A' + str(i+1)] = 1/(1 + np.exp(-z))

        return self.__cache['A' + str(i+1)], self.__cache
