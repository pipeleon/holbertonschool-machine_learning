#!/usr/bin/env python3
"""Task 4 Convolutional Neural Networks"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow
    """
    weights_init = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding='same',
        activation="relu",
        kernel_initializer=weights_init
    )(x)

    pool_1 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv_1)

    conv_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding='valid',
        activation="relu",
        kernel_initializer=weights_init
    )(pool_1)

    pool_2 = tf.layers.MaxPooling2D(
        pool_size=2,
        strides=2
    )(conv_2)

    flat = tf.layers.Flatten()(pool_2)

    layer_1 = tf.layers.Dense(
        units=120,
        activation="relu",
        name="layer",
        kernel_initializer=weights_init
    )(flat)

    layer_2 = tf.layers.Dense(
        units=84,
        activation="relu",
        name="layer",
        kernel_initializer=weights_init
    )(layer_1)

    output = tf.layers.Dense(
        units=10,
        activation=None,
        name="layer",
        kernel_initializer=weights_init
    )(layer_2)

    losses = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=output
    )

    comparation = tf.math.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(comparation, tf.float32))

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(losses)

    return output, train, losses, accuracy
