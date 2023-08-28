#!/usr/bin/env python3
"""Task 3 Tensorflow"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Returns: a tensor containing the decimal accuracy of the prediction"""
    comparation = tf.math.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(comparation, tf.float32))

    return accuracy
