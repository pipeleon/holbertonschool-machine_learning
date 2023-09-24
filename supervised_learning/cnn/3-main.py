#!/usr/bin/env python3

""" import numpy as np
pool_backward = __import__('3-pool_backward').pool_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_a = X_train[:10].reshape((-1, h, w, 1))
    X_train_b = 1 - X_train_a
    X_train_c = np.concatenate((X_train_a, X_train_b), axis=3)

    dA = np.random.randn(10, h // 3, w // 3, 2)
    print(dA.shape)
    print(X_train_c.shape)
    print(pool_backward(dA, X_train_c, (3, 3), stride=(3, 3), mode="")) """

import numpy as np
pool_backward = __import__('3-pool_backward').pool_backward

np.random.seed(6)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
c = np.random.randint(2, 5)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 5, 2)).tolist()

X = np.random.uniform(0, 1, (m, h, w, c))
dA = np.random.uniform(0, 1, (m, (h - fh) // sh + 1, (w - fw) // sw + 1, c))
Y = pool_backward(dA, X, (fh, fw), stride=(sh, sw), mode='max')
print(Y)
print(Y.shape)
