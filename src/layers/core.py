"""Provides interface for core-functions. Arranged in alphanumeric order."""

import tensorflow as tf


def dense(input_, output_col):
    """
    A fully-connected layer of neurons.
    Initializes weights from a normal distribution with mean 0 and STD 0.01.
    A bias-tensor (initialized to 0) is added to the resulting tensor.

    # Parameters
        input_ (tensor): A rank-2 tensor of shape [samples, columns].
        output_col (int): Output neurons; length of the output vector.
    # Returns
        A tensor.
    """
    input_col = input_.shape.as_list()[1]
    weight = tf.Variable(tf.random_normal([input_col, output_col], mean=0.0, stddev=0.05))
    bias = tf.Variable(tf.constant(0.0, shape=[output_col]))
    return tf.matmul(input_, weight) + bias
