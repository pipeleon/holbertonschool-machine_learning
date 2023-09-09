#!/usr/bin/env python3
"""Task 2 Regularization"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization"""
    weights_init = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode=("fan_avg"))
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        name="layer",
        kernel_initializer=weights_init,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )

    return layer(prev)
