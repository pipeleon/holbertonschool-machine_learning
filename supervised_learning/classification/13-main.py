#!/usr/bin/env python3

import numpy as np

NN = __import__('13-neural_network').NeuralNetwork

""" lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A1, A2 = nn.forward_prop(X)
nn.gradient_descent(X, Y, A1, A2, 0.5)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
 """

np.random.seed(13)
nx, l, m = np.random.randint(100, 1000, 3).tolist()
nn = NN(nx, l)
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
A1 = np.random.uniform(size=(l, m))
A2 = np.random.uniform(size=(1, m))
print(nn.W1)
nn.gradient_descent(X, Y, A1, A2)
print(nn.W1)
nn.gradient_descent(X, Y, A1, A2, alpha=0.5)
print(nn.W1)
try:
    nn.W1 = 10
    print('Fail: private attribute W1 is overwritten as a public attribute')
except:
    pass