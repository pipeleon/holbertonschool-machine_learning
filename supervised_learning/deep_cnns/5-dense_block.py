#!/usr/bin/env python3
"""Task 5 Deep Convolutional Architectures"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    Densely Connected Convolutional Networks
    """
    initializer = K.initializers.HeNormal()
    concat = X

    for i in range(layers):
        batch_n1 = K.layers.BatchNormalization()(concat)
        activation_1 = K.layers.Activation('relu')(batch_n1)

        conv_1 = K.layers.Conv2D(
            filters=4*growth_rate,
            kernel_size=1,
            padding="same",
            strides=1,
            kernel_initializer=initializer,
            activation="linear"
        )(activation_1)

        batch_n2 = K.layers.BatchNormalization()(conv_1)
        activation_2 = K.layers.Activation('relu')(batch_n2)

        conv_2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding="same",
            strides=1,
            kernel_initializer=initializer,
            activation="linear"
        )(activation_2)

        concat = K.layers.concatenate([concat, conv_2])

    return concat, (nb_filters + layers*growth_rate)
