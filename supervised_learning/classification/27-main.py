#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('27-deep_neural_network').DeepNeuralNetwork
one_hot_encode = __import__('24-one_hot_encode').one_hot_encode
one_hot_decode = __import__('25-one_hot_decode').one_hot_decode

lib= np.load('../data/MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_valid_3D = lib['X_valid']
Y_valid = lib['Y_valid']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1)).T
print(Y_train)
print(Y_valid)
print(Y_train.shape)
print(Y_valid.shape)
Y_train_one_hot = one_hot_encode(Y_train, 10)
Y_valid_one_hot = one_hot_encode(Y_valid, 10)
print(Y_train_one_hot)
print(Y_valid_one_hot)
print(Y_train_one_hot.shape)
print(Y_valid_one_hot.shape)

deep = Deep.load('27-saved.pkl')
print(deep.L)
print("A0: " + str(deep.cache.get('A0').shape))
print("A1: " + str(deep.cache.get('A1').shape))
print("A2: " + str(deep.cache.get('A2').shape))
print("A3: " + str(deep.cache.get('A3').shape))
print("b1: " + str(deep.weights.get('b1').shape))
print("b2: " + str(deep.weights.get('b2').shape))
print("b3: " + str(deep.weights.get('b3').shape))
print("W1: " + str(deep.weights.get('W1').shape))
print("W2: " + str(deep.weights.get('W2').shape))
print("W3: " + str(deep.weights.get('W3').shape))

print(deep.cost(Y_train_one_hot, deep.cache.get('A3')))

A_one_hot, cost = deep.train(X_train, Y_train_one_hot, iterations=100,
                             step=10, graph=False)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_train == A) / Y_train.shape[0] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

A_one_hot, cost = deep.evaluate(X_valid, Y_valid_one_hot)
A = one_hot_decode(A_one_hot)
accuracy = np.sum(Y_valid == A) / Y_valid.shape[0] * 100
print("Validation cost:", cost)
print("Validation accuracy: {}%".format(accuracy))

deep.save('27-output')

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_valid_3D[i])
    plt.title(A[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig("task-27.png")