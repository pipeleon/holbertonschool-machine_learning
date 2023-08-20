#!/usr/bin/env python3

import numpy as np

Neuron = __import__('5-neuron').Neuron

""" lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
neuron.gradient_descent(X, Y, A, 0.5)
print(neuron.W)
print(neuron.b) """

np.random.seed(6)
nx, m = np.random.randint(100, 1000, 2).tolist()
nn = Neuron(nx)
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
A = np.random.uniform(size=(1, m))
print(nn.W)
nn.gradient_descent(X, Y, A)
print(nn.W)
nn.gradient_descent(X, Y, A, alpha=0.5)
print(nn.W)
try:
    nn.W = 10
    print('Fail: private attribute W overwritten as a public attribute')
except:
    pass