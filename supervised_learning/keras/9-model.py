#!/usr/bin/env python3
"""Task 9 Keras"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model"""
    K.saving.save_model(network, filename)

    return None


def load_model(filename):
    """Loads an entire model"""
    return K.saving.load_model(filename)
