"""Provides interface for computing accuracies."""

import tensorflow as tf


def binary_accuracy_reporter(labels, output):  # TODO: This is fucked. Gotta change it.
    """
    Generates a tensor for reporting accuracy.
    Assumes binary encoding of the labels.
    Rules for elements in `output`:
    If `output` < 0, rounds to 0.
    If `output` > 1, rounds to 1.
    If 0 <= `output` <= 1, rounds half-up.

    # Parameters
        labels (tensor): Tensor for holding labels.
        output (tensor): Output of the model.
    # Returns
        An accuracy-tensor.
    """
    bools_labels = tf.cast(labels, tf.bool)
    halves = tf.constant(0.5, dtype=tf.float32, shape=output.shape.as_list())
    bools_output = tf.greater_equal(output, halves)
    bools_total = tf.equal(bools_labels, bools_output)
    return tf.reduce_mean(tf.cast(bools_total, tf.float32))


def categorical_accuracy_reporter(labels, output):
    """
    Generates a tensor for reporting accuracy.
    Assumes one-hot encoding of labels.
    Argmaxes for the corresponding result in `output`.

    # Parameters
        labels (tensor): Tensor for holding labels.
        output (tensor): Output of the model.
    # Returns
        An accuracy-tensor.
    """
    bools = tf.equal(tf.argmax(labels, axis=1), tf.argmax(output, axis=1))
    return tf.reduce_mean(tf.cast(bools, tf.float32))
