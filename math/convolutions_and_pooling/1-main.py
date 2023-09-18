#!/usr/bin/env python3

""" import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.savefig("1-normal.png")
    plt.imshow(images_conv[0], cmap='gray')
    plt.savefig("2-alter.png") """

import numpy as np
convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same

np.random.seed(1)
m = np.random.randint(1000, 2000)
h, w = np.random.randint(100, 200, 2).tolist()
fh, fw = (np.random.randint(1, 5, 2) * 2).tolist()

images = np.random.randint(0, 256, (m, h, w))
kernel = np.random.randint(0, 10, (fh, fw))
conv_ims = convolve_grayscale_same(images, kernel)
print(conv_ims)
print(conv_ims.shape)