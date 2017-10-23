"""Provides interface for activation-functions. Arranged in alphanumeric order."""

import tensorflow as tf


def crelu(input_):
    """
    Applies the concatenated rectified-linear-unit (CReLU) on the input, element-wise.

    # Parameters
        input_ (tensor): The tensor to be CReLU'd.
    # Returns
        The resulting tensor.
    """
    return tf.nn.crelu(input_)


def elu(input_):
    """
    Applies the exponential-linear-unit (ELU) on the input, element-wise.

    # Parameters
        input_ (tensor): The tensor to be ELU'd.
    # Returns
        The resulting tensor.
    """
    return tf.nn.elu(input_)


def prelu(input_, looks_linear=False):
    """
    Applies the parametric linear-unit (PReLU) on the input, element-wise.

    # Parameters
        input_ (tensor): The tensor to be PReLU'd.
        looks_linear (bool): Whether `alpha` is initialized to 1 or not.
                             See https://arxiv.org/pdf/1702.08591.pdf for details.
    # Returns
        The resulting tensor.
    """
    if looks_linear:
        alpha = tf.Variable(tf.constant(1, dtype=tf.float32, shape=input_.shape.as_list()[-1]))
    else:
        alpha = tf.Variable(tf.constant(0, dtype=tf.float32, shape=input_.shape.as_list()[-1]))
    return tf.maximum(0.0, input_) + alpha * tf.minimum(0.0, input_)


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
