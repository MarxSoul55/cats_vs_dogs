"""Provides interface for meta-functions for use in tandem with the training-process."""

import tensorflow as tf


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
