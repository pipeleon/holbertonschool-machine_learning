#!/usr/bin/env python3
"""Task 8 Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent:"""
    callbacks = []

    def scheduler(epoch, lr):
        return alpha / (1 + decay_rate * epoch)

    if early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))

    if learning_rate_decay:
        callbacks.append(K.callbacks.LearningRateScheduler(scheduler, 1))

    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath,
                                                     save_best_only=True))

    return network.fit(data, labels, batch_size,
                       epochs, verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
