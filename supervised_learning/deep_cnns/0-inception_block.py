#!/usr/bin/env python3
"""Task 0 Deep Convolutional Architectures"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block
    """
    initializer = K.initializers.HeNormal()

    conv_1 = K.layers.Conv2D(
        filters=filters[0],
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(A_prev)

    conv_31 = K.layers.Conv2D(
        filters=filters[1],
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(A_prev)

    conv_33 = K.layers.Conv2D(
        filters=filters[2],
        kernel_size=3,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(conv_31)

    conv_51 = K.layers.Conv2D(
        filters=filters[3],
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(A_prev)

    conv_55 = K.layers.Conv2D(
        filters=filters[4],
        kernel_size=5,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(conv_51)

    pool_3 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding="same"
    )(A_prev)

    conv_31 = K.layers.Conv2D(
        filters=filters[5],
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(pool_3)

    filter_concat = K.layers.concatenate([conv_1, conv_33, conv_55, conv_31])

    return filter_concat
