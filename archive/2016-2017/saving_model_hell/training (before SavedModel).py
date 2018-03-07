"""Provides interface for meta-functions for use in tandem with the training-process."""

import os
import shutil

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
    Saves the model in the current directory.

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
