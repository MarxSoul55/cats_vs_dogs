"""
Provides interface for convolutional operations. Arranged in alphanumeric order.

`_Nd` indicates dimensionality of the data, NOT the input-tensor itself! For example, if the
input-tensor is of shape [samples, rows, columns, channels], the dimensionality of the data is 2,
since there are rows and columns. Channels don't count; although the data is spread out over the
channels, the channels aren't an intrinsic dimension of the data.
"""

import tensorflow as tf


def avg_pool_2d(input_, filter_size=2, strides=2, padding='VALID'):
    """
    Pools `input_` by its rows and columns over each channel.
    Replaces its window with the average value.
    Akin to "blurring" out a feature-map.

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


def conv_2d(input_, output_chan, filter_size=3, strides=1, padding='SAME'):
    """
    Performs convolution on rows, columns, and channels of `input_`.
    Weights of the filter are initialized orthogonally from [-1, 1].
    Adds a bias-parameter after the merge; initial value is 0.

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
    initializer = tf.orthogonal_initializer(gain=1.0, dtype=tf.float32)
    weight = tf.Variable(initializer([filter_size, filter_size, input_chan, output_chan],
                                     dtype=tf.float32))
    bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[output_chan]))
    return tf.nn.conv2d(input_, weight, [1, strides, strides, 1], padding) + bias


def transposed_conv_2d(input_, output_dim, output_chan, filter_size=3, strides=1, padding='SAME'):
    """
    Performs transposed convolution on rows, columns, and channels of `input_`.
    Weights of the filter are initialized orthogonally from [-1, 1].
    Adds a bias-parameter after the merge; initial value is 0.

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
    input_chan = input_.shape.as_list(3)
    initializer = tf.orthogonal_initializer(gain=1.0, dtype=tf.float32)
    weight = tf.Variable(initializer([filter_size, filter_size, output_chan, input_chan],
                                     dtype=tf.float32))
    bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[output_chan]))
    batch_size = input_.shape.as_list()[0]
    if batch_size != 1:
        raise IndexError('Transposed convolution only works with a constant batch-size of 1.')
    return tf.nn.conv2d_transpose(input_, weight,
                                  [batch_size, output_dim, output_dim, output_chan],
                                  [1, strides, strides, 1],
                                  padding=padding) + bias


def depthwise_separable_conv_2d(input_, output_chan, filter_size=3, strides=1,
                                padding='SAME'):
    """
    Performs depthwise, separable convolution on rows, columns, and channels of `input_`.
    i.e. Applies one filter without merging channels, then does pointwise convolution to merge.
    Used mainly due to efficiency; has fewer parameters than normal convolution.
    Weights of each filter are initialized orthogonally from [-1, 1].
    Adds a bias-parameter after the merging of channels; initial value is 0.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
        output_chan (int): Amount of channels in output; AKA number of filters.
        filter_size (int): Size of the depthwise filter.
        strides (int): Amount of steps to jump for each filter.
        padding (str): Either 'SAME' or 'VALID'. Whether or not to use zero-padding.
    # Returns
        A `batch_size` x `output_dim` x `output_dim` x `output_chan` tensor.
    """
    input_chan = input_.shape.as_list()[3]
    initializer = tf.orthogonal_initializer(gain=1.0, dtype=tf.float32)
    depthwise_shape = [filter_size, filter_size, input_chan, 1]
    pointwise_shape = [1, 1, input_chan, output_chan]
    depthwise_weight = tf.Variable(initializer(depthwise_shape, dtype=tf.float32))
    pointwise_weight = tf.Variable(initializer(pointwise_shape, dtype=tf.float32))
    bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[output_chan]))
    return tf.nn.separable_conv2d(input_, depthwise_weight, pointwise_weight,
                                  [1, strides, strides, 1], padding) + bias


def flatten_2d(input_):
    """
    Flattens `input_` from its multiple rows, columns, and channels into a rank-2 tensor.
    Shape will be [samples, rows * columns * channels].

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


def global_avg_pool_2d(input_):
    """
    Globally pools `input_` by its rows and columns over each channel.
    Replaces the entire feature-map with the average value.
    Does not flatten into vector(s).
    Used mainly as a substitute for dense layers (besides output-layer) to save parameters.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
    """
    input_rows = input_.shape.as_list()[1]
    input_columns = input_.shape.as_list()[2]
    return tf.nn.avg_pool(input_, [1, input_rows, input_columns, 1], [1, 1, 1, 1], 'VALID')


def max_pool_2d(input_, filter_size=2, strides=2, padding='VALID'):
    """
    Pools `input_` by its rows and columns over each channel.
    Replaces its window with the maximum value.
    Akin to picking out features that were detected strongly and leaving out the rest.

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


def zero_pad_2d(input_, padding):
    """
    Goes across the channels of `input_` and...
    Adds `padding` rows and columns of zeros.
    Akin to fitting a square of zeros around each feature-map.

    # Parameters
        input_ (tensor): A tensor of shape [samples, rows, columns, channels].
        padding (int): Amount of padding to use.
    # Returns
        The resulting tensor.
    """
    return tf.pad(input_, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
                  mode='CONSTANT')
