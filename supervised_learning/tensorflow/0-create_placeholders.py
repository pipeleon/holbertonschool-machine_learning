#!/usr/bin/env python3
"""Task 0 Tensorflow"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """Returns: placeholders x and y"""
    x = tf.placerfolder(tf.float32, (None, nx))
    y = tf.placerfolder(tf.float32, (None, classes))

    return x, y
