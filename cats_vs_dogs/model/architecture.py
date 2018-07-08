"""Provides interface for the definition of the model's architecture."""

import tensorflow as tf

from layers import averagepooling_2d, convolution_2d, dense, maxpooling_2d


def primary(input_,
            name=None):
    """
    Builds the model's architecture on the graph.

    Parameters:
        - input_ (tf.placeholder)
            - Placeholder for the input data.
        - name (str)
            - The name of the scope of the model's operations.
    Returns:
        - The output of the model.
    """
    with tf.name_scope(name):
        skip = convolution_2d(input_, 32, activation=tf.nn.elu, name='conv1')
        x = convolution_2d(skip, 32, activation=tf.nn.elu, name='conv2')
        x = convolution_2d(x, 32, activation=tf.nn.elu, name='conv3')
        x = tf.add(x, skip, name='res1')
        x = maxpooling_2d(x, name='pool1')
        skip = convolution_2d(x, 64, activation=tf.nn.elu, name='conv4')
        x = convolution_2d(skip, 64, activation=tf.nn.elu, name='conv5')
        x = convolution_2d(x, 64, activation=tf.nn.elu, name='conv6')
        x = tf.add(x, skip, name='res2')
        x = maxpooling_2d(x, name='pool2')
        skip = convolution_2d(x, 128, activation=tf.nn.elu, name='conv7')
        x = convolution_2d(skip, 128, activation=tf.nn.elu, name='conv8')
        x = convolution_2d(x, 128, activation=tf.nn.elu, name='conv9')
        x = tf.add(x, skip, name='res3')
        x = maxpooling_2d(x, name='pool3')
        skip = convolution_2d(x, 256, activation=tf.nn.elu, name='conv10')
        x = convolution_2d(skip, 256, activation=tf.nn.elu, name='conv11')
        x = convolution_2d(x, 256, activation=tf.nn.elu, name='conv12')
        x = tf.add(x, skip, name='res4')
        x = maxpooling_2d(x, name='pool4')
        x = averagepooling_2d(x, filter_size=x.shape.as_list()[1], strides=1, name='pool6')
        x = tf.reshape(x, [1, x.shape.as_list()[3]], name='distilled')
        x = dense(x, 2, name='output')
        return x
