"""
Provides interface for model operations, arranged in alpha-dimensional order.

If an operation is dimension-specific, then it is specified with an `_Nd` at the end,
where `N` is the amount of dimensions that operation supports.

All operations assume a tensor-shape of [samples, ...] where `...` can be height, width, depth, and
any other combination of dimensions. See `input_` descriptions in docstrings for required shape.
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
        An `output_chan`-dimensional tensor.
    """
    input_chan = input_.shape.as_list()[3]
    weight = tf.Variable(tf.random_normal([filter_size, filter_size, input_chan, output_chan],
                                          mean=0.0, stddev=0.05))
    bias = tf.Variable(tf.constant(0.0, shape=[output_chan]))
    return tf.nn.conv2d(input_, weight, [1, strides, strides, 1], padding) + bias


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


def elu(input_):
    """
    Applies the exponential linear unit (ELU) on the input, element-wise.

    # Parameters
        input_ (tensor): The tensor to apply the activation on.
    # Returns
        The resulting tensor.
    """
    return tf.nn.elu(input_)


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
        input_ (tensor): The tensor to be softmaxed.
    # Returns
        The resulting tensor.
    """
    return tf.nn.softmax(input_)
