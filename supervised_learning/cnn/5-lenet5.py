#!/usr/bin/env python3
"""Task 5 Convolutional Neural Networks"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras
    """
    initializer = K.initializers.HeNormal()

    conv_1 = K.layers.Conv2D(
        filters=6,
        kernel_size=5,
        padding="same",
        kernel_initializer=initializer,
        activation="relu"
    )(X)

    pool_1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv_1)

    conv_2 = K.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="valid",
        kernel_initializer=initializer,
        activation="relu"
    )(pool_1)

    pool_2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv_2)

    flat = K.layers.Flatten()(pool_2)

    layer_1 = K.layers.Dense(120, "relu",
                             kernel_initializer=initializer)(flat)
    layer_2 = K.layers.Dense(84, "relu",
                             kernel_initializer=initializer)(layer_1)
    layer_3 = K.layers.Dense(10, "softmax",
                             kernel_initializer=initializer)(layer_2)

    model = K.Model(inputs=X, outputs=layer_3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    return model
