"""Provides interface for the definition of the model's architecture."""

import tensorflow as tf


def averagepooling_2d(input_, filter_size=2, strides=2, padding='VALID', name=None):
    """
    Pools `input_` by its rows and columns over each channel.
    Replaces its window with the average value.

    # Parameters
        input_ (tensor):
            - The input tensor.
            - Must have shape [samples, rows, columns, channels].
        filter_size (int):
            - Width and height of each filter.
            - e.g. 2 specifies a 2x2 filter through with the window is pooled.
        strides (int):
            - Amount of steps to jump for each pass of the filter.
        padding (str):
            - Either 'SAME' or 'VALID'.
            - Controls whether or not to use zero-padding to preserve the dimensions.
        name (str):
            - Name scope for this TF operation.
    # Returns
        A tensor.
    """
    with tf.name_scope(name):
        output = tf.nn.avg_pool(input_, [1, filter_size, filter_size, 1], [1, strides, strides, 1],
                                padding, name='output')
        return output


def convolution_2d(input_, output_chan, filter_size=3, strides=1, padding='SAME', activation=None,
                   dtype=tf.float32, name=None):
    """
    Performs convolution on rows, columns, and channels of `input_`.
    Weights of the filter are initialized orthogonally from [-1, 1].
    Adds a bias-parameter after the merge; initial value is 0.

    # Parameters
        input_ (tf.placeholder):
            - The input tensor.
            - Must have shape [samples, rows, columns, channels].
        output_chan (int):
            - Amount of channels in output.
            - AKA the amount of filters/kernels.
        filter_size (int):
            - Width and height of each filter.
            - e.g. 3 specifies a 3x3 filter.
        strides (int):
            - Amount of steps to jump for each pass of the filter.
        padding (str):
            - Either 'SAME' or 'VALID'.
            - Controls whether or not to use zero-padding to preserve the dimensions.
        activation (tf.something):
            - Additional activation that may be applied.
            - e.g. activation=tf.nn.elu
            - `None` specifies a linear activation.
        dtype (tf-supported dtype):
            - What datatype the parameters of the convolution will use.
        name (str):
            - Name scope for this TF operation.
    # Returns
        - The resulting tensor whose shape depends on the `padding` argument.
    """
    with tf.name_scope(name):
        weight_shape = [filter_size, filter_size, input_.shape.as_list()[3], output_chan]
        weight_init = tf.orthogonal_initializer(gain=1.0, dtype=dtype)
        weight = tf.Variable(initial_value=weight_init(weight_shape, dtype=dtype), name='weight')
        bias = tf.Variable(initial_value=tf.constant(0, dtype=dtype, shape=[output_chan]),
                           name='bias')
        conv = tf.add(tf.nn.conv2d(input_, weight, [1, strides, strides, 1], padding), bias)
        if activation is None:
            output = tf.identity(conv, name='output')
        else:
            output = activation(conv, name='output')
        return output


def dense(input_, output_col, dtype=tf.float32, name=None):
    """
    A fully-connected layer of neurons.
    Weights are initialized orthogonally from [-1, 1].
    Adds biases which are initialized to 0.

    # Parameters
        input_ (tensor):
            - A rank-2 tensor of shape [samples, columns].
        output_col (int):
            - Output neurons
            - AKA length of the output vector.
        dtype (tf-supported dtype):
            - What datatype the parameters of the dense layer will use.
    # Returns
        - A resulting tensor with shape [batch_size, output_col].
    """
    with tf.name_scope(name):
        weight_shape = [input_.shape.as_list()[1], output_col]
        weight_init = tf.orthogonal_initializer(gain=1.0, dtype=dtype)
        weight = tf.Variable(initial_value=weight_init(weight_shape, dtype=dtype), name='weight')
        bias = tf.Variable(initial_value=tf.constant(0, dtype=dtype, shape=[output_col]),
                           name='bias')
        output = tf.add(tf.matmul(input_, weight), bias, name='output')
        return output


def model(input_, name=None):
    """
    Builds the model's architecture on the graph.

    # Parameters
        input_ (tf.placeholder):
            - Placeholder for the input data.
        name (str):
            - The name of the scope of the model's operations.
    # Returns
        - The output of the model.
    """
    with tf.name_scope(name):
        skip = tf.layers.conv2d(input_, 16, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(skip, 16, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(x, 16, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(skip, 32, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(skip, 64, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(x, 64, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 128, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(skip, 128, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(x, 128, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        skip = tf.layers.conv2d(x, 256, 3, padding='same', activation=tf.nn.elu,
                                kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(skip, 256, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = tf.layers.conv2d(x, 256, 3, padding='same', activation=tf.nn.elu,
                             kernel_initializer=tf.orthogonal_initializer())
        x = x + skip
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.average_pooling2d(x, x.shape.as_list()[1], 1)
        x = tf.reshape(x, [1, x.shape.as_list()[3]], name='distilled')
        x = tf.layers.dense(x, 2, kernel_initializer=tf.orthogonal_initializer())
        x = tf.identity(x, name='output')
        return x
