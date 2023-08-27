#!/usr/bin/env python3
"""Task 0 Tensorflow"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """Returns: placeholders x and y"""
    x = tf.placerholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placerholder(tf.float32, shape=(None, classes), name="y")

    return x, y
