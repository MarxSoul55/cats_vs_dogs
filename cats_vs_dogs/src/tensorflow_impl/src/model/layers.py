"""
Provides computational layers for assembling a model's architecture.
Arranged in alphabetical order.
"""

import tensorflow as tf


def averagepooling_2d(input_,
                      filter_size=2,
                      strides=2,
                      padding='VALID',
                      name=None):
    """
    Pools `input_` by its rows and columns over each channel.
    Replaces its window with the average value.

    Parameters:
        - input_ (tensor)
            - The input tensor.
            - Must be in NHWC format.
        - filter_size (int)
            - Width and height of each filter.
            - e.g. 2 specifies a 2x2 filter through with the window is pooled.
        - strides (int)
            - Amount of steps to jump for each pass of the filter.
        - padding (str)
            - Either 'SAME' or 'VALID'.
            - Controls whether or not to use zero-padding to preserve the dimensions.
        - name (str)
            - Name scope for this TF operation.
    Returns:
        - The resulting tensor whose shape depends on `padding`.
    """
    with tf.name_scope(name):
        output = tf.nn.avg_pool(input_, [1, filter_size, filter_size, 1], [1, strides, strides, 1],
                                padding, data_format='NHWC', name='output')
        return output


def convolution_2d(input_,
                   input_chan,
                   output_chan,
                   filter_size=3,
                   strides=1,
                   dilation=1,
                   padding='SAME',
                   activation=None,
                   dtype=tf.float32,
                   name=None):
    """
    Performs convolution on rows, columns, and channels of `input_`.
    Weights of the filter are initialized orthogonally from [-1, 1].
    Adds a bias-parameter after the merge; initial value is 0.

    Parameters:
        - input_ (tf.placeholder)
            - The input tensor.
            - Must be in NHWC format.
        - input_chan (int)
            - Number of channels in the input.
        - output_chan (int)
            - Amount of channels in output.
            - AKA the amount of filters/kernels.
        - filter_size (int)
            - Width and height of each filter.
            - e.g. 3 specifies a 3x3 filter.
        - strides (int)
            - Amount of steps to jump for each pass of the filter.
        - dilation (int)
            - Dilation factor for each filter.
        - padding (str)
            - Either 'SAME' or 'VALID'.
            - Controls whether or not to use zero-padding to preserve the dimensions.
        - activation (tf.something)
            - Additional activation that may be applied.
            - e.g. activation=tf.nn.elu
            - `None` specifies a linear activation.
        - dtype (tf-supported dtype)
            - What datatype the parameters of the convolution will use.
        - name (str)
            - Name scope for this TF operation.
    Returns:
        - The resulting tensor whose shape depends on the `padding` argument.
    """
    with tf.name_scope(name):
        weight_shape = [filter_size, filter_size, input_chan, output_chan]
        weight_init = tf.orthogonal_initializer(gain=1.0, dtype=dtype)
        weight = tf.Variable(initial_value=weight_init(weight_shape, dtype=dtype), name='weight')
        bias = tf.Variable(initial_value=tf.constant(0, dtype=dtype, shape=[output_chan]),
                           name='bias')
        conv = tf.nn.conv2d(input_, weight, [1, strides, strides, 1], padding,
                            data_format='NHWC', dilations=[1, dilation, dilation, 1]) + bias
        if activation is None:
            output = tf.identity(conv, name='output')
        else:
            output = activation(conv, name='output')
        return output


def dense(input_,
          input_col,
          output_col,
          activation=None,
          dtype=tf.float32,
          name=None):
    """
    A fully-connected layer of neurons.
    Weights are initialized orthogonally from [-1, 1].
    Adds biases which are initialized to 0.

    Parameters:
        - input_ (tensor)
            - A rank-2 tensor of shape [samples, columns].
        - input_col (int)
            - Columns of the input vector; AKA number of input neurons.
        - output_col (int)
            - Output neurons.
            - AKA length of the output vector.
        - activation (tf.something)
            - Additional activation that may be applied.
            - e.g. activation=tf.nn.elu
            - `None` specifies a linear activation.
        - dtype (tf-supported dtype)
            - What datatype the parameters of the dense layer will use.
        - name (str)
            - Name scope for this TF operation.
    Returns:
        - A resulting tensor with shape [batch_size, output_col].
    """
    with tf.name_scope(name):
        weight_shape = [input_col, output_col]
        weight_init = tf.orthogonal_initializer(gain=1.0, dtype=dtype)
        weight = tf.Variable(initial_value=weight_init(weight_shape, dtype=dtype), name='weight')
        bias = tf.Variable(initial_value=tf.constant(0, dtype=dtype, shape=[output_col]),
                           name='bias')
        dense = tf.matmul(input_, weight) + bias
        if activation is None:
            output = tf.identity(dense, name='output')
        else:
            output = activation(dense, name='output')
        return output


def maxpooling_2d(input_,
                  filter_size=2,
                  strides=2,
                  padding='VALID',
                  name=None):
    """
    Pools `input_` by its rows and columns over each channel.
    Replaces its window with the maximum value in the window.

    Parameters:
        - input_ (tensor)
            - The input tensor.
            - Must be in NHWC format.
        - filter_size (int)
            - Width and height of each filter.
            - e.g. 2 specifies a 2x2 filter through with the window is pooled.
        - strides (int)
            - Amount of steps to jump for each pass of the filter.
        - padding (str)
            - Either 'SAME' or 'VALID'.
            - Controls whether or not to use zero-padding to preserve the dimensions.
        - name (str)
            - Name scope for this TF operation.
    Returns:
        - The resulting tensor whose shape depends on `padding`.
    """
    with tf.name_scope(name):
        output = tf.nn.max_pool(input_, [1, filter_size, filter_size, 1], [1, strides, strides, 1],
                                padding, data_format='NHWC', name='output')
        return output
