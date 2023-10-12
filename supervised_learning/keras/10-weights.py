#!/usr/bin/env python3
"""Task 9 Keras"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Saves a model's weights"""
    network.save_weights(filename)

    return None


def load_weights(network, filename):
    """Loads a model's weights"""
    network.load_weights(filename)

    return None
