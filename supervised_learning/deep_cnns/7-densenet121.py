#!/usr/bin/env python3
"""Task 7 Deep Convolutional Architectures"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)
    """
    initializer = K.initializers.HeNormal()

    X = K.Input(shape=(224, 224, 3))
    batch_n1 = K.layers.BatchNormalization()(X)
    activation_1 = K.layers.Activation('relu')(batch_n1)

    conv_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="same",
        kernel_initializer=initializer,
        activation="linear"
    )(activation_1)

    pool_1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
    )(conv_1)

    dense_block1, nb_filters = dense_block(
        pool_1, 64, growth_rate, 6)

    transition_1, nb_filters = transition_layer(
        dense_block1, nb_filters, compression)

    dense_block2, nb_filters = dense_block(
        transition_1, nb_filters, growth_rate, 12)

    transition_2, nb_filters = transition_layer(
        dense_block2, nb_filters, compression)

    dense_block3, nb_filters = dense_block(
        transition_2, nb_filters, growth_rate, 24)

    transition_3, nb_filters = transition_layer(
        dense_block3, nb_filters, compression)

    dense_block4, nb_filters = dense_block(
        transition_3, nb_filters, growth_rate, 16)

    pool_avg = K.layers.AveragePooling2D(
            pool_size=(7, 7),
            strides=(1, 1)
    )(dense_block4)

    Y = K.layers.Dense(
        1000,
        "softmax",
        kernel_initializer=initializer,
    )(pool_avg)

    model = K.models.Model(inputs=X, outputs=Y)

    return model
