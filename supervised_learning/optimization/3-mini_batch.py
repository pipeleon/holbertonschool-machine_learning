#!/usr/bin/env python3
"""Task 3 Optimization"""
import numpy as np
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent
    """
    with tf.Session() as sess:
        model = tf.train.import_meta_graph(load_path + '.meta')
        model.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]

        for i in range(epochs + 1):
            train_cost, train_accuracy = sess.run(
                        [loss, accuracy], feed_dict={x: X_train, y: Y_train}
                    )
            validation_cost, validation_accuracy = sess.run(
                        [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
                    )

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(validation_cost))
            print("\tValidation Accuracy: {}".format(validation_accuracy))

            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            if i < epochs:
                for j in range(m // batch_size):
                    end = (j+1)*batch_size if (j+1)*batch_size < m else m
                    X_batch = X_shuffled[(j*batch_size):end]
                    Y_batch = Y_shuffled[(j*batch_size):end]

                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if j % 100 == 0 and j > 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X_batch, y: Y_batch}
                        )
                        print("\tStep {}:".format(j))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))
        return model.save(sess, save_path)
