#!/usr/bin/env python3

""" import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.savefig("3-normal.png")
    plt.imshow(images_conv[0], cmap='gray')
    plt.savefig("3-alter.png") """
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale

np.random.seed(3)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 4, 2)).tolist()

images = np.random.randint(0, 256, (m, h, w))
kernel = np.random.randint(0, 10, (fh, fw))

print(images.shape)
print(kernel.shape)
print(sh)
print(sw)
conv_ims = convolve_grayscale(images, kernel, stride=(sh, sw))
np.set_printoptions(threshold=np.inf)
print(conv_ims)
print(conv_ims.shape)