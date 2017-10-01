"""
Provides interface for convolutional operations. Arranged in alphanumeric order.

`_Nd` indicates dimensionality of the data, NOT the input-tensor itself! For example, if the
input-tensor is of shape [samples, rows, columns, channels], the dimensionality of the data is 2,
since there are rows and columns. Channels don't count; although the data is spread out over the
channels, the channels aren't an intrinsic dimension of the data.
"""

import tensorflow as tf


def averagepooling_2d(input_, filter_size=2, strides=2, padding='VALID'):
    """
    Pools `input_` by its rows and columns over each channel; replaces with average value.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
        filter_size (int): Height and width of the filter.
        strides (int): Amount of steps to jump for the filter.
        padding (str): Either 'SAME' or 'VALID'.
    # Returns
        A tensor.
    """
    return tf.nn.avg_pool(input_, [1, filter_size, filter_size, 1], [1, strides, strides, 1],
                          padding)


def convolution_2d(input_, output_chan, filter_size=3, strides=1, padding='SAME'):
    """
    Performs convolution on rows, columns, and channels of `input_`.
    Initializes weights from a normal distribution with mean 0 and STD 0.01.
    A bias-tensor (initialized to 0) is added to the resulting tensor.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
        output_chan (int): Amount of channels in output; AKA number of filters.
        filter_size (int): Width and height of each filter.
        strides (int): Amount of steps to jump for each filter.
        padding (str): Either 'SAME' or 'VALID'. Whether or not to use zero-padding.
    # Returns
        A `batch_size` x `output_dim` x `output_dim` x `output_chan` tensor.
    """
    input_chan = input_.shape.as_list()[3]
    weight = tf.Variable(tf.random_normal([filter_size, filter_size, input_chan, output_chan],
                                          mean=0.0, stddev=0.01))
    bias = tf.Variable(tf.constant(0.0, shape=[output_chan]))
    return tf.nn.conv2d(input_, weight, [1, strides, strides, 1], padding) + bias


def deconvolution_2d(input_, output_dim, output_chan, filter_size=3, strides=1, padding='SAME'):
    """
    Performs deconvolution on rows, columns, and channels of `input_`.
    Initializes weights from a normal distribution with mean 0 and STD 0.01.
    A bias-tensor (initialized to 0) is added to the resulting tensor.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
        output_dim (int): Height and width of the resulting tensor.
        output_chan (int): Amount of channels in output; AKA number of filters.
        filter_size (int): Width and height of each filter.
        strides (int): Amount of steps to jump for each filter.
        padding (str): Either 'SAME' or 'VALID'. Whether or not to use zero-padding.
    # Returns
        A `batch_size` x `output_dim` x `output_dim` x `output_chan` tensor.
    """
    batch_size = input_.shape.as_list(0)
    input_chan = input_.shape.as_list(3)
    weight = tf.Variable(tf.random_normal([filter_size, filter_size, output_chan, input_chan],
                                          mean=0.0, stddev=0.01))
    bias = tf.Variable(tf.constant(0.0, shape=[output_chan]))
    return tf.nn.conv2d_transpose(input_, weight,
                                  [batch_size, output_dim, output_dim, output_chan],
                                  [1, strides, strides, 1], padding=padding) + bias


def flatten_2d(input_):
    """
    Flattens `input_` from its multiple rows, columns, and channels into a rank-2 tensor with shape
    [samples, rows * columns * channels].

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
    # Returns
        A flattened tensor.
    """
    input_row = input_.shape.as_list()[1]
    input_col = input_.shape.as_list()[2]
    input_chan = input_.shape.as_list()[3]
    output_col = input_row * input_col * input_chan
    return tf.reshape(input_, [-1, output_col])


def globalaveragepooling_2d(input_):
    """
    Globally pools `input_` by its rows and columns over each channel; replaces with average value.
    Does not flatten into vector(s).

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
    """
    input_dim = input_.shape.as_list()[1]
    return tf.nn.avg_pool(input_, [1, input_dim, input_dim, 1], [1, 1, 1, 1], 'VALID')


def maxpooling_2d(input_, filter_size=2, strides=2, padding='VALID'):
    """
    Pools `input_` by its rows and columns over each channel; replaces with maximum value.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
        filter_size (int): Width and height of the filter.
        strides (int): Amount of steps to jump for the filter.
        padding (str): Either 'SAME' or 'VALID'.
    # Returns
        The resulting tensor.
    """
    return tf.nn.max_pool(input_, [1, filter_size, filter_size, 1], [1, strides, strides, 1],
                          padding)


def zeropadding_2d(input_, padding):
    """
    Pads `input_` by its rows and columns over each channel with zeros.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
        padding (int): Amount of padding to use.
    # Returns
        The resulting tensor.
    """
    return tf.pad(input_, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                  mode='CONSTANT')
