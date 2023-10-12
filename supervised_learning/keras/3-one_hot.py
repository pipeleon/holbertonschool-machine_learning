#!/usr/bin/env python3
"""Task 3 Keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""
    matrix = K.utils.to_categorical(labels, classes)

    return matrix
