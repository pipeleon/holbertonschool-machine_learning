#!/usr/bin/env python3
"""Task 1 Keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    input = K.Input((nx, ))
    regularizer = K.regularizers.L2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            prev = K.layers.Dense(layers[i], activations[i],
                                  kernel_regularizer=regularizer)(input)
        else:
            prev = (K.layers.Dense(layers[i], activations[i],
                                   kernel_regularizer=regularizer))(prev)
        if i != len(layers) - 1:
            prev = K.layers.Dropout(1 - keep_prob)(prev)

    model = K.Model(inputs=input, outputs=prev)

    return model
