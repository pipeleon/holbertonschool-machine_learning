#!/usr/bin/env python3
"""Task 1 Deep Convolutional Architectures"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described
    in Going Deeper with Convolutions (2014)
    """
    initializer = K.initializers.HeNormal()

    X = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(X)

    pool_1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
    )(conv_1)

    conv_2 = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(pool_1)

    conv_3 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(conv_2)

    pool_2 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
    )(conv_3)

    layer_3a = inception_block(pool_2, [64, 96, 128, 16, 32, 32])

    layer_3b = inception_block(layer_3a, [128, 128, 192, 32, 96, 64])

    pool_3 = K.layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding="same"
        )(layer_3b)

    layer_4a = inception_block(pool_3, [192, 96, 208, 16, 48, 64])

    layer_4b = inception_block(layer_4a, [160, 112, 224, 24, 64, 64])

    layer_4c = inception_block(layer_4b, [128, 128, 256, 24, 64, 64])

    layer_4d = inception_block(layer_4c, [112, 144, 288, 32, 64, 64])

    layer_4e = inception_block(layer_4d, [256, 160, 320, 32, 128, 128])

    pool_4 = K.layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding="same"
        )(layer_4e)

    layer_5a = inception_block(pool_4, [256, 160, 320, 32, 128, 128])

    layer_5b = inception_block(layer_5a, [384, 192, 384, 48, 128, 128])

    flat = K.layers.Flatten()(layer_5b)

    pool_5 = K.layers.AveragePooling2D(
            pool_size=(7, 7),
            strides=(1, 1)
    )(layer_5b)

    dropout = K.layers.Dropout(0.4)(pool_5)

    Y = K.layers.Dense(
        1000,
        "softmax",
        kernel_initializer=initializer,
    )(dropout)

    model = K.models.Model(inputs=X, outputs=Y)

    return model
