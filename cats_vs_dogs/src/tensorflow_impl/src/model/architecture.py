"""Provides interface for the definition of the model's architecture."""

import tensorflow as tf

from .layers import averagepooling_2d, convolution_2d, dense, maxpooling_2d


def baby_resnet(input_,
                name=None):
    """
    Builds the model's architecture on the graph.

    Parameters:
        - input_ (tf.placeholder)
            - This model is designed to take in tensors of shape [1, 128, 128, 3] (NHWC).
        - name (str)
            - The name of the scope of the model's operations.
    Returns:
        - When used with `sess.run`:
            - The output of the model.
            - A [1, 2] shape tensor.
    """
    with tf.name_scope(name):
        skip = convolution_2d(input_, 3, 32, activation=tf.nn.elu, name='conv1')
        x = convolution_2d(skip, 32, 32, activation=tf.nn.elu, name='conv2')
        x = convolution_2d(x, 32, 32, activation=tf.nn.elu, name='conv3')
        x = tf.add(x, skip, name='res1')
        x = maxpooling_2d(x, name='pool1')
        skip = convolution_2d(x, 32, 64, activation=tf.nn.elu, name='conv4')
        x = convolution_2d(skip, 64, 64, activation=tf.nn.elu, name='conv5')
        x = convolution_2d(x, 64, 64, activation=tf.nn.elu, name='conv6')
        x = tf.add(x, skip, name='res2')
        x = maxpooling_2d(x, name='pool2')
        skip = convolution_2d(x, 64, 128, activation=tf.nn.elu, name='conv7')
        x = convolution_2d(skip, 128, 128, activation=tf.nn.elu, name='conv8')
        x = convolution_2d(x, 128, 128, activation=tf.nn.elu, name='conv9')
        x = tf.add(x, skip, name='res3')
        x = maxpooling_2d(x, name='pool3')
        skip = convolution_2d(x, 128, 256, activation=tf.nn.elu, name='conv10')
        x = convolution_2d(skip, 256, 256, activation=tf.nn.elu, name='conv11')
        x = convolution_2d(x, 256, 256, activation=tf.nn.elu, name='conv12')
        x = tf.add(x, skip, name='res4')
        x = maxpooling_2d(x, name='pool4')
        x = averagepooling_2d(x, filter_size=8, strides=1, name='pool6')
        x = tf.reshape(x, [1, 256], name='distilled')
        x = dense(x, 256, 2, name='output')
        return x
