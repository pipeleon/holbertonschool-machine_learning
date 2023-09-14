#!/usr/bin/env python3
"""Task 2 Keras"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Builds a neural network with the Keras library"""
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss=K.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

    return None
