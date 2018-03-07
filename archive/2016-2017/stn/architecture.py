"""Defines the architecture of the model and functions for STNs."""

import tensorflow as tf

from licensed.transformer import spatial_transformer


def localization_network(input_):
    """
    Given a tensor, apply an affine transformation and give the result.
    Assumes `input_` is a square tensor of shape [a, b, b, c].

    # Parameters
        input_ (tensor): The input.
    # Returns
        A composite, TensorFlow operation.
    """
    if input_.shape.as_list()[1] != input_.shape.as_list()[2]:
        raise IndexError('Expected a square-like input, was disappointed.')
    ln0 = tf.layers.conv2d(input_, 8, 3, padding='same',
                           kernel_initializer=tf.orthogonal_initializer())
    ln1 = tf.nn.elu(ln0)
    ln2 = tf.layers.max_pooling2d(ln1, 2, 2)
    ln3 = tf.layers.conv2d(ln2, 16, 3, padding='same',
                           kernel_initializer=tf.orthogonal_initializer())
    ln4 = tf.nn.elu(ln3)
    ln5 = tf.layers.max_pooling2d(ln4, 2, 2)
    ln6 = tf.reshape(ln5, [1, (ln5.shape.as_list()[1] ** 2) * ln5.shape.as_list()[3]])
    ln7 = tf.layers.dense(ln6, 6)
    ln7_rows, ln7_cols = ln7.shape.as_list()
    w_ln8 = tf.Variable(tf.zeros([ln7_rows, 6]))
    b_ln8 = tf.Variable(initial_value=[[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
    ln8 = tf.matmul(tf.zeros([ln7_rows, ln7_cols]), w_ln8) + b_ln8
    return ln8


def spatial_transformer_network(input_):
    """
    Interface for spatial-transformer networks.

    # Parameters
        input_ (tensor): Tensor that was inputted into the localization network.
    # Returns
        A TensorFlow operation.
    """
    transformation_matrix = localization_network(input_)
    return spatial_transformer(input_, transformation_matrix)


def model(input_):
    """
    Defines the model's architecture.

    # Parameters
        input_ (tf.placeholder): Placeholder for the input data.
    # Returns
        The output of the model.
    """
    output = tf.layers.conv2d(input_, 8, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output, 16, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 16, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output, 32, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 32, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 32, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output, 64, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 64, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 64, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 64, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.conv2d(output, 128, 3, padding='same',
                              kernel_initializer=tf.orthogonal_initializer())
    output = tf.nn.elu(output)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.average_pooling2d(output, output.shape.as_list()[1], 1)
    output = tf.reshape(output, [1, output.shape.as_list()[3]])
    output = tf.layers.dense(output, 2, kernel_initializer=tf.orthogonal_initializer())
    return output
