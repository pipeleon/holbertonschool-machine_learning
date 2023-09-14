#!/usr/bin/env python3
"""Task 4 Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent:"""
    return network.fit(data, labels, batch_size,
                       epochs, verbose, shuffle=shuffle,
                       validation_data=validation_data)
