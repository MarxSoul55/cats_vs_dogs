"""Provides interface for activation-functions. Arranged in alphanumeric order."""

import tensorflow as tf


def elu(input_):
    """
    Applies the exponential-linear-unit (ELU) on the input, element-wise.

    # Parameters
        input_ (tensor): The tensor to be ELU'd.
    # Returns
        The resulting tensor.
    """
    return tf.nn.elu(input_)


def relu(input_):
    """
    Applies rectified-linear-unit (ReLU) on the input, element-wise.

    # Parameters
        input_ (tensor): The tensor to be ReLU'd.
    # Returns
        The resulting tensor.
    """
    return tf.nn.relu(input_)


def sigmoid(input_):
    """
    Applies sigmoid function to the input, element-wise.

    # Parameters
        input_ (tensor): The tensor to be sigmoid'd.
    # Returns
        The resulting tensor.
    """
    return tf.nn.sigmoid(input_)


def softmax(input_):
    """
    Applies softmax function to the input.

    # Parameters
        input_ (tensor): The tensor to be softmax'd.
    # Returns
        The resulting tensor.
    """
    return tf.nn.softmax(input_)


def tanh(input_):
    """
    Applies hyperbolic-tangent to the input.

    # Parameters
        input_ (tensor): The tensor to be tanh'd.
    # Returns
        The resulting tensor.
    """
    return tf.tanh(input_)
