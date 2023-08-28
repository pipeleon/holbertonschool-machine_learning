#!/usr/bin/env python3
"""Task 1 Tensorflow"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Returns: layer for x, n and activation funtion"""
    weights_init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        units=n, 
        activation= activation, 
        name="layer", 
        kernel_initializer=weights_init
    )

    return layer(prev)
