"""Provides interface for core-functions. Arranged in alphanumeric order."""

import tensorflow as tf


def dense(input_, output_col):
    """
    A fully-connected layer of neurons.
    Weights are initialized orthogonally from [-1, 1].
    Biases initialized to 0.

    # Parameters
        input_ (tensor): A rank-2 tensor of shape [samples, columns].
        output_col (int): Output neurons; length of the output vector.
    # Returns
        A tensor.
    """
    input_col = input_.shape.as_list()[1]
    weight_shape = [input_col, output_col]
    initializer = tf.orthogonal_initializer(gain=1.0, dtype=tf.float32)
    weight = tf.Variable(initializer(weight_shape, dtype=tf.float32))
    bias = tf.Variable(tf.constant(0.0, shape=[output_col]))
    return tf.matmul(input_, weight) + bias


def residual(prev_layer, layer):
    """
    Given some tensor, adds the output from a previous tensor to it.
    Use to implement residual neural-networks.
    Requires that both `prev_layer` and `layer` are of the same shape.

    # Parameters
        prev_layer (tensor): The output of the previous layer in question.
        layer (tensor): The new tensor that `prev_layer` will be added to.
    # Raises
        ValueError: if `prev_layer` and `layer` are different shapes.
    """
    prev_shape = prev_layer.shape.as_list()
    shape = layer.shape.as_list()
    if prev_shape != shape:
        raise ValueError('`prev_layer` and `layer` must be the same shape.')
    return tf.add(prev_layer, layer)
