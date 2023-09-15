#!/usr/bin/env python3
"""Task 7 Keras"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent:"""
    def scheduler(epoch, lr):
        return lr / (1 + decay_rate * epoch)

    return network.fit(data, labels, batch_size,
                       epochs, verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=[K.callbacks.EarlyStopping(patience=patience),
                                  K.callbacks.LearningRateScheduler(scheduler,
                                                                    1)])
