"""Provides interface for meta-ops that deal with TensorFlow and the model."""

import os

import tensorflow as tf


def predict_binary(output):
    """
    Converts a sigmoidal tensor to a binary tensor.

    # Parameters
        output (tensor): A tensor of values put through a logistic-sigmoid function.
    # Returns
        A binary version of it (half-rounded-up).
    """
    halves = tf.constant(0.5, dtype=tf.float32, shape=output.shape.as_list())
    return tf.cast(tf.greater_equal(output, halves), tf.float32)


def restore_model(session, name='saved_model'):
    """
    Restores the model to the session of choice.

    # Parameters
        session (tf.Session): Session to restore to.
        name (str): Name of the `.meta` and `checkpoint` files.
    """
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(os.getcwd(), '{}/{}'.format(name, name)))


def save_model(session, name='saved_model'):
    """
    Saves the model in the current directory.

    # Parameters
        session (tf.Session): A session to save.
        name (str): Name of the save-dir.
    """
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), '{}/{}'.format(name, name)))
