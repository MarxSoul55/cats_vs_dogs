"""Provides interface for meta-functions for use in tandem with the training-process."""

import os

import tensorflow as tf


def restore_model(session, savedir='saved_model'):
    """
    Restores the model to the session of choice.

    # Parameters
        session (tf.Session): Session to restore to.
        savedir (str): Name of the save-dir.
    """
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(os.getcwd(), '{}/{}'.format(savedir, savedir)))


def save_model(session, savedir='saved_model'):
    """
    Saves the model in the current directory, along with global-step.

    # Parameters
        session (tf.Session): A session to save.
        savedir (str): Name of the save-dir.
    """
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), '{}/{}'.format(savedir, savedir)))
