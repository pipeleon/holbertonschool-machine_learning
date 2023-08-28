#!/usr/bin/env python3
"""Task 2 Tensorflow"""
import tensorflow.compat.v1 as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Returns: the prediction of the network in tensor form"""
    prev = x

    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])

    return prev
