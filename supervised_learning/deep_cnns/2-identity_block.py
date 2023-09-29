#!/usr/bin/env python3
"""Task 2 Deep Convolutional Architectures"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)
    """
    initializer = K.initializers.HeNormal()

    conv_1 = K.layers.Conv2D(
        filters=filters[0],
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(A_prev)

    batch_n1 = K.layers.BatchNormalization()(conv_1)
    activation_1 = K.layers.Activation('relu')(batch_n1)

    conv_2 = K.layers.Conv2D(
        filters=filters[1],
        kernel_size=3,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(activation_1)

    batch_n2 = K.layers.BatchNormalization()(conv_2)
    activation_2 = K.layers.Activation('relu')(batch_n2)

    conv_3 = K.layers.Conv2D(
        filters=filters[2],
        kernel_size=1,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(activation_2)

    batch_n3 = K.layers.BatchNormalization()(conv_3)

    add = K.layers.Add()([batch_n3, A_prev])

    activation_3 = K.layers.Activation('relu')(add)

    return activation_3
