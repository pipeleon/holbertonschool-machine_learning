#!/usr/bin/env python3
"""Task 4 Tensorflow"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Returns: a tensor containing the loss of the prediction"""

    lose = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=y_pred
    )

    return lose
