"""Provides interface for meta-ops that deal with TensorFlow and the model."""

import os
import shutil

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


def restore_model(session, tag, savedir='saved_model'):
    """
    Restores the model to the session of choice.

    # Parameters
        session (tf.Session): Session to restore to.
        tag (str): An ID for the model.
        savedir (str): Name of the save-dir.
    """
    tf.saved_model.loader.load(session, [tag], savedir)


def save_model(session, tag, savedir='saved_model'):
    """
    Saves the model to a directory.

    # Parameters
        session (tf.Session): A session to save.
        tag (str): An ID for the model.
        savedir (str): Name of the save-dir.
    """
    if savedir in os.listdir():
        shutil.rmtree(savedir)
    saver = tf.saved_model.builder.SavedModelBuilder(savedir)
    saver.add_meta_graph_and_variables(session, [tag])
    saver.save()
