"""Provides interface for meta-functions for use in tandem with the training-process."""

import os
import shutil

import tensorflow as tf


def restore_variables(session, savedir='saved_model'):
    """
    Restores variables of the graph (whose data is in `savedir`).
    Note that the model (plus any namescopes it's associated with)...
    ...must be redeclared with standard API commands before this function.

    # Parameters
        session (tf.Session): Session to restore to.
        savedir (str): Name of the save-dir.
    """
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(os.getcwd(), '{}/{}'.format(savedir, savedir)))


def save_variables(session, savedir='saved_model'):
    """
    Saves variables of the current graph being used by the sesion.

    # Parameters
        session (tf.Session): A session to save.
        savedir (str): Name of the save-dir.
    """
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), '{}/{}'.format(savedir, savedir)))


def tensorboard_writer(logdir='tensorboard'):
    """
    Writer for tensorboard.

    # Parameters
        logdir (str): Name of the directory to store the logs.
    # Returns
        A `FileWriter` object.
    """
    if logdir in os.listdir():
        shutil.rmtree(logdir)
    return tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
