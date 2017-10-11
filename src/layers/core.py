"""Provides interface for core-functions. Arranged in alphanumeric order."""

import tensorflow as tf


def dense(input_, output_col):
    """
    A fully-connected layer of neurons.

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
