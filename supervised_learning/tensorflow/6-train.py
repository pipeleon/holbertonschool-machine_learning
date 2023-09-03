#!/usr/bin/env python3
"""Task 6 Tensorflow"""
import tensorflow.compat.v1 as tf


create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """builds, trains, and saves a neural network classifier"""

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            print(i)
            """ _, loss_value = sess.run((train_op, loss))
            if i == iterations or i % 100 == 0:
                print("After {i} iterations:")
                print("\tTraining Cost: {}".format(loss_value))
                print("\tTraining Accuracy: {}".format(calculate_accuracy(y, y_pred))) """
            train_cost, train_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_train, y: Y_train}
                )
            validation_cost, validation_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
                )

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(validation_cost))
                print("\tValidation Accuracy: {}".format(validation_accuracy))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
