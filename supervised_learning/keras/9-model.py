#!/usr/bin/env python3
"""Task 9 Keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model"""
    network.save(filename)

    return None


def load_model(filename):
    """Loads an entire model"""
    return K.models.load_model(filename)
