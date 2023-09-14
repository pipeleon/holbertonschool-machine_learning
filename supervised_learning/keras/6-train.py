#!/usr/bin/env python3
"""Task 6 Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent:"""
    return network.fit(data, labels, batch_size,
                       epochs, verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=K.callbacks.EarlyStopping(patience=patience))
