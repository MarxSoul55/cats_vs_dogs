"""Provides interface for objective functions."""

import tensorflow as tf


def mean_absolute_error(labels, predictions):
    """
    Calculates the mean absolute error.

    # Parameters
        labels (tensor): Tensor for the labels.
        predictions (tensor): Tensor for predictions.
    # Returns
        The (scalar) mean result from the elementwise errors.
    """
    return tf.losses.absolute_difference(labels, predictions, reduction=tf.losses.Reduction.MEAN)


def mean_binary_entropy(labels, logits):
    """
    Calculates the mean binary entropy.

    # Parameters
        labels (tensor): Tensor for the labels.
        logits (tensor): The (raw) output of the model, or unscaled log-probabilities.
    # Returns
        The (scalar) mean result from the elementwise errors.
    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))


def mean_squared_error(labels, predictions):
    """
    Calculates the mean squared error.

    # Parameters
        labels (tensor): Tensor for the labels.
        predictions (tensor): Tensor for predictions.
    # Returns
        The (scalar) mean result from the elementwise errors.
    """
    return tf.losses.mean_squared_error(labels, predictions, reduction=tf.losses.Reduction.MEAN)


def root_mean_squared_error(labels, predictions):
    """
    Calculates the root mean squared error.

    # Parameters
        labels (tensor): Tensor for the labels.
        predictions (tensor): Tensor for the predictions.
    # Returns
        The (scalar) mean result from the elementwise errors.
    """
    return tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, predictions)))
