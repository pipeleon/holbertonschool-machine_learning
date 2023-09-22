#!/usr/bin/env python3

""" import numpy as np
conv_backward = __import__('2-conv_backward').conv_backward

if __name__ == "__main__":
    np.random.seed(0)
    lib = np.load('../data/MNIST.npz')
    X_train = lib['X_train']
    _, h, w = X_train.shape
    X_train_c = X_train[:10].reshape((-1, h, w, 1))

    W = np.random.randn(3, 3, 1, 2)
    b = np.random.randn(1, 1, 1, 2)

    dZ = np.random.randn(10, h - 2, w - 2, 2)
    print(conv_backward(dZ, X_train_c, W, b, padding="valid"))
 """
import numpy as np
conv_backward = __import__('2-conv_backward').conv_backward

np.random.seed(5)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
cin = np.random.randint(2, 5)
cout = np.random.randint(5, 10)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 4, 2)).tolist()

X = np.random.uniform(0, 1, (m, h, w, cin))
W = np.random.uniform(0, 1, (fh, fw, cin, cout))
b = np.random.uniform(0, 1, (1, 1, 1, cout))
dZ = np.random.uniform(0, 1, (m, h, w, cout))
print("X: {}".format(X.shape))
print("W: {}".format(W.shape))
print("b: {}".format(b.shape))
print("dZ: {}".format(dZ.shape))
print("sh, sw: {}, {}".format(sh, sw))
dA, dW, db = conv_backward(dZ, X, W, b, padding="same", stride=(sh, sw))
np.set_printoptions(threshold=np.inf)
""" print(dA)print(dW)print(db) """
print(dA.shape)

print(dW.shape)

print(db.shape)