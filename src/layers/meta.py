"""Provides interface for meta-ops that deal with TensorFlow and the model."""

import os

import tensorflow as tf


def restore_model(session, name='saved_model'):
    """
    Restores the model to the session of choice.

    # Parameters
        session (tf.Session): Session to restore to.
        name (str): Name of the `.meta` and `checkpoint` files.
    """
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(os.getcwd(), name))


def save_model(session, name='saved_model'):
    """
    Saves the model in the current directory.

    # Parameters
        session (tf.Session): A session to save.
        name (str): Name of the saved files.
    """
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), name))
