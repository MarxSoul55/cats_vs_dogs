"""Provides interface for meta-functions for use in tandem with the training-process."""

import os
import shutil

import tensorflow as tf


def restore_model(session, tag='default', savedir='saved_model'):
    """
    Restores the model to the session of choice.

    # Parameters
        session (tf.Session): Session to restore to.
        tag (str): An ID for the model, e.g. "training" or "serving".
        savedir (str): Name of the save-dir.
    """
    tf.saved_model.loader.load(session, [tag], savedir)


def save_model(session, tag='default', savedir='saved_model'):
    """
    Saves the model to a directory.

    # Parameters
        session (tf.Session): A session to save.
        tag (str): An ID for the model, e.g. "training" or "serving".
        savedir (str): Name of the save-dir.
    """
    if savedir in os.listdir():
        shutil.rmtree(savedir)
    saver = tf.saved_model.builder.SavedModelBuilder(savedir)
    saver.add_meta_graph_and_variables(session, [tag])
    saver.save()


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
