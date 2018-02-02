"""Defines the architecture of the model via a genetic algorithm."""

import tensorflow as tf


def convolution_elu(input_, filters):
    """
    Bootstrapping function for the genetic algorithm.

    # Parameters
        input_ (tensor): The tensor that will serve as the input.
        filters (int): However many filters the layer will have.
    # Returns
        The ELU'd result of the convolution.
    """
    return tf.nn.elu(tf.layers.conv2d(input_, filters, 3, padding='same',
                                      kernel_initializer=tf.orthogonal_initializer()))


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
