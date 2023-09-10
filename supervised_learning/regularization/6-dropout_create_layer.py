#!/usr/bin/env python3
"""Task 6 Regularization"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    weights_init = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode=("fan_avg"))
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=weights_init
    )
    dropout = tf.layers.Dropout(keep_prob)

    return dropout(layer(prev))
