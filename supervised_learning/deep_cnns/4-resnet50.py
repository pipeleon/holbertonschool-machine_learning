#!/usr/bin/env python3
"""Task 4 Deep Convolutional Architectures"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)
    """
    initializer = K.initializers.HeNormal()

    X = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="same",
        kernel_initializer=initializer,
        activation="linear"
    )(X)

    batch_n1 = K.layers.BatchNormalization()(conv_1)
    activation_1 = K.layers.Activation('relu')(batch_n1)

    pool_1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
    )(activation_1)

    project_1 = projection_block(pool_1, [64, 64, 256], 1)

    identity_1 = identity_block(project_1, [64, 64, 256])

    identity_2 = identity_block(identity_1, [64, 64, 256])

    project_2 = projection_block(identity_2, [128, 128, 512])

    identity_3 = identity_block(project_2, [128, 128, 512])

    identity_4 = identity_block(identity_3, [128, 128, 512])

    identity_5 = identity_block(identity_4, [128, 128, 512])

    project_3 = projection_block(identity_5, [256, 256, 1024])

    identity_6 = identity_block(project_3, [256, 256, 1024])

    identity_7 = identity_block(identity_6, [256, 256, 1024])

    identity_8 = identity_block(identity_7, [256, 256, 1024])

    identity_9 = identity_block(identity_8, [256, 256, 1024])

    identity_10 = identity_block(identity_9, [256, 256, 1024])

    project_4 = projection_block(identity_10, [512, 512, 2048])

    identity_11 = identity_block(project_4, [512, 512, 2048])

    identity_12 = identity_block(identity_11, [512, 512, 2048])

    pool_avg = K.layers.AveragePooling2D(
            pool_size=(7, 7),
            strides=(1, 1)
    )(identity_12)

    Y = K.layers.Dense(
        1000,
        "softmax",
        kernel_initializer=initializer,
    )(pool_avg)

    model = K.models.Model(inputs=X, outputs=Y)

    return model
