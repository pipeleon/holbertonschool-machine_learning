#!/usr/bin/env python3
"""Task 11 Keras"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format"""
    config = network.to_json()

    with open(filename, "w") as saver:
        saver.write(config)
    return None


def load_config(filename):
    """Loads a model's configuration in JSON format"""
    with open(filename, "r") as loader:
        model = K.models.model_from_json(loader.read())

    return model
