#!/usr/bin/env python3
"""Task 14 Optimization"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer
    for a neural network in tensorflow
    """
    weights_init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    return tf.nn.batch_normalization()
