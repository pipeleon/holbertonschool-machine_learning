#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    """ test = np.array([[2 ,0, -2], [3, 0, -3], [0, 0, -1]])
    print(np.sum(kernel * test)) """
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.savefig("0-normal.png")
    plt.imshow(images_conv[0], cmap='gray')
    plt.savefig("0-alter.png")